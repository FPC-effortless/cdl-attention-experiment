"""Late-acquisition language adapters on the frozen COG-LM-001 recurrent models."""
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'cog_lm_001'))
from model import Config, LM, Successors, mix, successor_ids
from vendor.mamba2 import InferenceCache, ssd


class RevisionBank(nn.Module):
    """Route each block to one distribution over competing records. Revise matched records."""
    def __init__(self, width, slots=8, dim=32, period=16):
        super().__init__()
        self.period = period
        self.anchors = nn.Parameter(torch.randn(slots, dim) * .1)
        self.key = nn.Linear(width, dim)
        self.query = nn.Linear(width, dim)
        self.value = nn.Linear(width, dim)
        self.salience = nn.Linear(width, 1)
        self.strength = nn.Linear(width, 1)
        self.readout = nn.Linear(dim, width, bias=False)

    def forward(self, hidden, state=None):
        batch, length, _ = hidden.shape
        if state is None:
            keys = self.anchors.expand(batch, -1, -1)
            values = torch.zeros_like(keys)
            pending = hidden[:, :0]
        else:
            keys, values, pending = state
        output = []
        for part in torch.split(torch.cat((pending, hidden), 1), self.period, dim=1):
            current = part[:, pending.size(1):]
            query = F.normalize(self.query(current), dim=-1)
            addresses = F.normalize(keys, dim=-1).transpose(1, 2)
            read = (8 * (query @ addresses)).softmax(-1) @ values
            output.append(self.readout(read))
            pending = part.new_empty(batch, 0, hidden.size(-1))
            if part.size(1) < self.period:
                pending = part.contiguous().clone()
                break
            pooled = (self.salience(part).softmax(1) * part).sum(1, keepdim=True)
            key = F.normalize(self.key(pooled), dim=-1)
            route = (8 * (key @ addresses)).softmax(-1).transpose(1, 2)
            rate = route * self.strength(pooled).sigmoid()
            keys = torch.lerp(keys, key.expand_as(keys), rate)
            values = torch.lerp(values, self.value(pooled).tanh().expand_as(values), rate)
        return torch.cat(output, 1), (keys, values, pending)


class Readout(nn.Module):
    def __init__(self, width, revision):
        super().__init__()
        self.down = nn.Linear(width, 64)
        self.up = nn.Linear(64, width)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.gate = nn.Linear(width, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -1.5)
        self.bank = RevisionBank(width) if revision else None

    def forward(self, h, index, embedding, bank=None, disabled=False, last_only=False, mask_read=False):
        h = h + self.up(F.gelu(self.down(h)))
        if self.bank is not None and not disabled:
            read, bank = self.bank(h, bank)
            h = h + (read * 0 if mask_read else read)
        if last_only:
            h, index = h[:, -1:], index[:, -1:]
        logp = F.log_softmax(F.linear(h, embedding), -1)
        return mix(logp, index, self.gate(h).squeeze(-1)), bank


def mamba_chunk(m, u, cache):
    """Continue the upstream SSD block path with the live convolution and SSM state."""
    c = m.args
    z, raw, dt = m.in_proj(u).split((c.d_inner, c.d_inner + 2*c.d_state, c.nheads), -1)
    raw = torch.cat((cache.conv_state[:, :, 1:], raw.transpose(1, 2)), -1)
    filtered = F.silu(F.conv1d(raw, m.conv1d.weight, m.conv1d.bias, groups=raw.size(1))).transpose(1, 2)
    x, b, k = filtered.split((c.d_inner, c.d_state, c.d_state), -1)
    x = x.unflatten(-1, (c.nheads, c.headdim))
    dt = F.softplus(dt + m.dt_bias)
    y, state = ssd(x * dt.unsqueeze(-1), -m.A_log.exp() * dt, b.unsqueeze(2), k.unsqueeze(2),
                   c.chunk_size, initial_states=cache.ssm_state.unsqueeze(1))
    out = m.out_proj(m.norm((y + x * m.D[:, None]).flatten(2), z))
    return out, InferenceCache(raw[:, :, -c.d_conv:].contiguous(), state.contiguous())


def features(model, x, states=None):
    h = model.emb(x)
    new = []
    for layer, state in zip(model.blocks, states or [None]*len(model.blocks)):
        if model.c.arm == 'mamba2' and state is not None and x.size(1) > 1:
            y, state = mamba_chunk(layer.mixer, layer.norm(h), state)
            h = h + y
        else:
            h, state = layer(h, state)
        new.append(state)
    return model.norm(h), new


def backbone(path):
    packet = torch.load(path, map_location='cpu', weights_only=True)
    model = LM(Config(**packet['config']))
    model.load_state_dict(packet['state'])
    if model.c.arm not in ('local', 'mamba2'):
        raise ValueError('expected the memory-free pretrained sequence model')
    return model.eval().requires_grad_(False)
