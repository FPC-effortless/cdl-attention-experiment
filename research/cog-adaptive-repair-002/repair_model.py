"""A bounded, contextual byte-copy memory on the canonical slow-state C model.

The inherited C state dict can initialize the backbone exactly. The new
streaming state is (original_C_state, (history_ids, slot_by_byte_matrix)).
Start a fresh stream when switching from C, since an old C state cannot
reconstruct the new copy memory. Causal sequence evaluation uses 24-token
blocks; the persistent state has no dependency on document length.
"""
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "unified-search-002"))
from integrated import WholeCOG, VOCAB
from model import D


class ContextualCopy(nn.Module):
    history_length = 23
    channels = 12
    slots = 12

    def __init__(self):
        super().__init__()
        self.byte_features = nn.Embedding(VOCAB, self.channels)
        self.local_context = nn.Linear((self.history_length + 1) * self.channels, 32)
        self.backbone_context = nn.Linear(D, 32, bias=False)
        self.key = nn.Linear(32, self.slots)
        self.query = nn.Linear(32, self.slots)
        self.write = nn.Linear(32, 1)
        self.read_gain = nn.Linear(D, 1)
        nn.init.zeros_(self.read_gain.weight)
        nn.init.zeros_(self.read_gain.bias)

    def forward(self, ids, features, state=None):
        batch = ids.shape[0]
        if state is None:
            history = ids.new_zeros(batch, self.history_length)
            memory = features.new_zeros(batch, self.slots, VOCAB)
        else:
            history, memory = state
        packed = torch.cat((history, ids), dim=1)
        emb = self.byte_features(packed)
        windows = emb.unfold(1, self.history_length + 1, 1).flatten(-2)
        context = (self.local_context(windows) + self.backbone_context(features)).tanh()
        writes = self.key(context).softmax(-1) * self.write(context).sigmoid()
        # The factor bounds reciprocal prefix products within each short block.
        writes = 0.8 * writes
        queries = self.query(context).softmax(-1)
        outputs = []
        for tokens, write, query in zip(ids.split(24, 1), writes.split(24, 1), queries.split(24, 1)):
            # M_t = (1-w_t) * M_(t-1) + w_t * one_hot(byte_t).
            # Expand a causal block analytically so training can batch the reads.
            prefix = (1 - write).cumprod(1)
            prior_weight = query * prefix
            current_weight = write / prefix
            current_attention = (prior_weight @ current_weight.transpose(1, 2)).tril()
            value = F.one_hot(tokens, VOCAB).to(features.dtype)
            mass = prior_weight @ memory + current_attention @ value
            outputs.append(mass / mass.sum(-1, keepdim=True).clamp_min(1e-6))
            last = prefix[:, -1]
            memory = last.unsqueeze(-1) * memory + (
                last.unsqueeze(1) * current_weight
            ).transpose(1, 2) @ value
        copy_probs = torch.cat(outputs, 1)
        return self.read_gain(features) * copy_probs, (
            packed[:, -self.history_length:].clone(), memory,
        )


class RepairModel(WholeCOG):
    def __init__(self, seed, parent_state=None):
        super().__init__("SS_native_native", seed)
        if parent_state is not None:
            self.load_state_dict(parent_state, strict=True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 310919)
            self.copy_memory = ContextualCopy()
        self.disable_read = False

    def forward(self, x, state=None, carry=False, features=False):
        core_state, copy_state = (None, None) if state is None else state
        logits, core_state, hidden = super().forward(x, core_state, features=True)
        residual, copy_state = self.copy_memory(x, hidden, copy_state)
        if not self.disable_read:
            logits = logits + residual
        next_state = (core_state, copy_state)
        if features:
            return logits, next_state, hidden
        return (logits, next_state) if carry else logits


def build_repair_model(seed, parent_state=None):
    return RepairModel(seed, parent_state)
