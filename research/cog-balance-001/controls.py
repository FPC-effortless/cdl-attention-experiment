"""COG-BALANCE controls for the 288-symbol whole-model experiment.

Native HH, Unified U and GQA use the 1.08M WholeCOG source. Recurrent
controls replace the two GQA sequence mixers, preserving Local, the tied
readout and residual FF topology. All PyTorch reference kernels use fp32.

Mamba-2 follows state-spaces/mamba mamba2.py blob 36b16d4715ed2d753697167d65da5e6abba22017.
Gated DeltaNet follows fla-org/flash-linear-attention gated_deltanet.py blob
8264bda8a9c639c3103fda40317aebee67d5e944. These are independent CPU
implementations of the recurrences, not optimized upstream kernel builds.
"""
import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

BASE = Path(__file__).resolve().parents[1] / "unified-search-002"
sys.path.insert(0, str(BASE))
from integrated import WholeCOG
from model import D, count


class GRUSeq(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(D, D, batch_first=True)

    def forward(self, x, state=None):
        return self.gru(x, state)


def depthwise(module, x, history):
    # The history contains pre-convolution activations in chronological order.
    x = x.transpose(1, 2)
    if history is None:
        history = x.new_zeros(x.size(0), x.size(1), module.kernel_size[0] - 1)
    packed = torch.cat((history, x), dim=-1)
    return F.silu(module(packed)).transpose(1, 2), packed[..., -3:].clone()


def initial_dt(heads):
    dt = torch.empty(heads).uniform_(math.log(0.001), math.log(0.1)).exp()
    return nn.Parameter(dt + torch.log(-torch.expm1(-dt)))


def decay_kernel(log_decay):
    # [batch, heads, T, T], zero above the diagonal; mask before exp.
    prefix = log_decay.cumsum(-1)
    differences = prefix.unsqueeze(-1) - prefix.unsqueeze(-2)
    future = torch.ones_like(differences[0, 0], dtype=torch.bool).triu(1)
    return prefix.exp(), differences.masked_fill(future, -torch.inf).exp()


class Mamba2Seq(nn.Module):
    """Single-group SSD, learned dt and A, gated RMS, 16-dimensional state."""
    heads, head_dim, state_dim = 4, 64, 16

    def __init__(self):
        super().__init__()
        inner = self.heads * self.head_dim
        channels = inner + 2 * self.state_dim
        self.in_proj = nn.Linear(D, inner + channels + self.heads, bias=False)
        self.conv = nn.Conv1d(channels, channels, 4, groups=channels)
        self.dt_bias = initial_dt(self.heads)
        self.log_a = nn.Parameter(torch.rand(self.heads).mul_(15).add_(1).log())
        self.skip = nn.Parameter(torch.ones(self.heads))
        self.norm = nn.RMSNorm(inner, eps=1e-5)
        self.out = nn.Linear(inner, D, bias=False)

    def forward(self, x, state=None):
        inner = self.heads * self.head_dim
        width = self.state_dim
        gate, raw, dt = self.in_proj(x).split((inner, inner + 2 * width, self.heads), -1)
        features, tail = depthwise(self.conv, raw, None if state is None else state[1])
        v, b, c = features.split((inner, width, width), -1)
        v = v.unflatten(-1, (self.heads, self.head_dim)).transpose(1, 2)
        dt = F.softplus(dt + self.dt_bias).transpose(1, 2)
        log_decay = -self.log_a.exp()[None, :, None] * dt
        memory = x.new_zeros(len(x), self.heads, width, self.head_dim) if state is None else state[0]
        outputs = []
        for start in range(0, x.size(1), 32):
            sl = slice(start, start + 32)
            b_chunk, c_chunk = b[:, None, sl], c[:, None, sl]
            p, decay = decay_kernel(log_decay[:, :, sl])
            u = v[:, :, sl] * dt[:, :, sl, None]
            y = (c_chunk @ memory) * p[..., None] + ((c_chunk @ b_chunk.transpose(-1, -2)) * decay) @ u
            memory = memory * p[:, :, -1, None, None] + b_chunk.transpose(-1, -2) @ (u * decay[:, :, -1, :, None])
            outputs.append(y + v[:, :, sl] * self.skip[None, :, None, None])
        mixed = torch.cat(outputs, dim=2).transpose(1, 2).flatten(-2)
        return self.out(self.norm(mixed * F.silu(gate))), (memory, tail)


class GatedDeltaSeq(nn.Module):
    """L2-normalized keys/queries, short conv, scalar decay and delta write."""
    heads, key_dim, value_dim =  3, 32, 64

    def __init__(self):
        super().__init__()
        k, v = self.heads * self.key_dim, self.heads * self.value_dim
        self.qkv = nn.Linear(D, 2 * k + v, bias=False)
        self.conv = nn.Conv1d(2 * k + v, 2 * k + v, 4, groups=2 * k + v, bias=False)
        self.rates = nn.Linear(D, 2 * self.heads, bias=False)
        self.gate = nn.Linear(D, v, bias=False)
        self.norm = nn.RMSNorm(self.value_dim, eps=1e-5)
        self.out = nn.Linear(v, D, bias=False)
        self.dt_bias = initial_dt(self.heads)
        self.log_a = nn.Parameter(torch.rand(self.heads).clamp_min(1e-6).mul_(16).log())

    def forward(self, x, state=None):
        features, tail = depthwise(self.conv, self.qkv(x), None if state is None else state[1])
        k_width, v_width = self.heads * self.key_dim, self.heads * self.value_dim
        q, k, v = features.split((k_width, k_width, v_width), dim=-1)
        q = F.normalize(q.unflatten(-1, (self.heads, self.key_dim)), dim=-1).transpose(1, 2) / math.sqrt(self.key_dim)
        k = F.normalize(k.unflatten(-1, (self.heads, self.key_dim)), dim=-1).transpose(1, 2)
        v = v.unflatten(-1, (self.heads, self.value_dim)).transpose(1, 2)
        dt, beta = self.rates(x).chunk(2, -1)
        beta = beta.sigmoid().transpose(1, 2)
        log_decay = (-self.log_a.exp() * F.softplus(dt + self.dt_bias)).transpose(1, 2)
        memory = x.new_zeros(len(x), self.heads, self.key_dim, self.value_dim) if state is None else state[0]
        outputs = []
        for start in range(0, x.size(1), 16):
            sl = slice(start, start + 16)
            qc, kc = q[:, :, sl], k[:, :, sl]
            p, decay = decay_kernel(log_decay[:, :, sl])
            beta_chunk = beta[:, :, sl, None]
            correction = (beta_chunk * decay * (kc @ kc.transpose(-1, -2))).tril(-1)
            rhs = beta_chunk * (v[:, :, sl] - p[..., None] * (kc @ memory))
            update = torch.linalg.solve_triangular(correction, rhs, upper=False, unitriangular=True)
            outputs.append(p[..., None] * (qc @ memory) + (decay * (qc @ kc.transpose(-1, -2))) @ update)
            memory = memory * p[:, :, -1, None, None] + kc.transpose(-1, -2) @ (update * decay[:, :, -1, :, None])
        y = self.norm(torch.cat(outputs, dim=2).transpose(1, 2)).flatten(-2)
        return self.out(y * F.silu(self.gate(x))), (memory, tail)


class MixerLayer(nn.Module):
    def __init__(self, old, seq):
        super().__init__()
        self.seq = seq
        self.norm, self.norm2 = old.norm, old.norm2
        extra = count(seq) - count(old.qkv) - count(old.out)
        width = old.ff[0].out_features - round(extra / (2 * D + 1))
        if not 0 < width <= old.ff[0].out_features:
            raise ValueError(f"Cannot match width {width} with parent initialization")
        self.ff = nn.Sequential(nn.Linear(D, width), nn.GELU(), nn.Linear(width, D))
        with torch.no_grad():
            self.ff[0].weight.copy_(old.ff[0].weight[:width])
            self.ff[0].bias.copy_(old.ff[0].bias[:width])
            self.ff[2].weight.copy_(old.ff[2].weight[:, :width])
            self.ff[2].bias.copy_(old.ff[2].bias)

    def forward(self, h, state=None, side=None):
        mixed, new_state = self.seq(self.norm(h), state)
        h = h + mixed
        return h + self.ff(self.norm2(h)), new_state


def build_control(arm, seed):
    parents = {"A": "NN_native_native", "B": "BB_native_native", "GQA": "GQA"}
    if arm in parents:
        return WholeCOG(parents[arm], seed)
    mixers = {"GRU": GRUSeq, "GATED_DELTA": GatedDeltaSeq, "MAMBA2": Mamba2Seq}
    if arm not in mixers:
        raise ValueError(f"Unknown control {arm!r}")
    m = WholeCOG("GQA", seed)
    for index, block in enumerate(m.parent.core.blocks):
        torch.manual_seed(seed + 60013 + index)
        m.parent.core.blocks[index] = MixerLayer(block, mixers[arm]())
    if abs(count(m) - 1077488) / 1077488 > 0.002:
        raise ValueError(f"Control parameter mismatch: {count(m)}")
    return m
