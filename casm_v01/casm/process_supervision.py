from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import PAD
from .recurrent_model import CASMRecurrent


class ProcessHead(nn.Module):
    """Training-only decoder from recurrent latent state to a fixed process code."""

    def __init__(self, d_model: int, code_dim: int = 64):
        super().__init__()
        self.code_dim = int(code_dim)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, self.code_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def fixed_trace_code(text: str, dim: int = 64) -> torch.Tensor:
    """Order-sensitive deterministic target vector with no learned parameters."""
    if dim < 4 or dim % 2:
        raise ValueError("trace code dimension must be even and >= 4")
    raw = text.encode("utf-8", errors="replace")[:192]
    v = torch.zeros(dim, dtype=torch.float32)
    half = dim // 2
    if not raw:
        v[0] = 1.0
        return v
    for pos, byte in enumerate(raw):
        p = float(pos + 1)
        b = float(byte + 1)
        for j in range(half):
            freq = math.exp(-math.log(10000.0) * j / max(1, half - 1))
            angle = p * b * freq
            v[2 * j] += math.sin(angle)
            v[2 * j + 1] += math.cos(angle)
    return F.normalize(v, dim=0)


def _capture_reasoning_states(
    model: CASMRecurrent,
    tokens: torch.Tensor,
    anchors: torch.Tensor,
) -> torch.Tensor:
    """Return post-retrieval recurrent states at each answer anchor.

    Shape: [batch, reasoning_steps, d_model]. The anchor is the last byte of
    the literal ``answer `` marker, so local causal attention cannot see any
    gold answer byte at the supervised location.
    """
    b, t = tokens.shape
    if anchors.shape != (b,):
        raise ValueError("anchors must have shape [batch]")
    x_in = tokens[:, :-1]
    total_len = x_in.shape[1]
    ring, ring_valid, state = model._init_memory(
        b, tokens.device, model.embed.weight.dtype
    )
    captured: List[List[torch.Tensor | None]] = [
        [None for _ in range(model.reasoning_steps)] for _ in range(b)
    ]

    for start in range(0, total_len, model.cfg.chunk_size):
        end = min(start + model.cfg.chunk_size, total_len)
        ids = x_in[:, start:end]
        h = model.embed(ids)
        for block in model.blocks:
            h = block(h)
        h = model.norm(h)

        candidates, cand_valid = model._candidates(ring, ring_valid, state)
        if model.cfg.use_memory:
            final_h, pre_states, _, _ = model.reason(
                h, candidates, cand_valid, capture_states=True
            )
            post_states = list(pre_states[1:]) + [final_h]
        else:
            final_h = h + model.memory_ffn(model.memory_ffn_norm(h))
            post_states = [final_h for _ in range(model.reasoning_steps)]

        for bi in range(b):
            a = int(anchors[bi])
            if start <= a < end:
                local = a - start
                for ri, z in enumerate(post_states):
                    captured[bi][ri] = z[bi, local]

        valid_tok = ids != PAD
        last_idx = valid_tok.long().sum(dim=1).clamp_min(1) - 1
        summary = final_h[torch.arange(b, device=tokens.device), last_idx]
        state = model.state.update(state, summary)
        new_mem = torch.tanh(model.memory_in(summary))
        ring = torch.cat([ring[:, 1:], new_mem[:, None, :]], dim=1)
        ring_valid = torch.cat(
            [
                ring_valid[:, 1:],
                torch.ones(b, 1, device=tokens.device, dtype=torch.bool),
            ],
            dim=1,
        )

    missing = [
        (bi, ri)
        for bi in range(b)
        for ri in range(model.reasoning_steps)
        if captured[bi][ri] is None
    ]
    if missing:
        raise RuntimeError(f"failed to capture process states: {missing[:4]}")
    return torch.stack(
        [torch.stack([x for x in row if x is not None], dim=0) for row in captured],
        dim=0,
    )


def process_alignment_loss(
    model: CASMRecurrent,
    head: ProcessHead,
    tokens: torch.Tensor,
    anchors: torch.Tensor,
    traces: Sequence[Sequence[str]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    states = _capture_reasoning_states(model, tokens, anchors)
    if len(traces) != states.shape[0]:
        raise ValueError("trace batch size mismatch")
    if any(len(row) != states.shape[1] for row in traces):
        raise ValueError("every trace row must match reasoning_steps")

    pred = F.normalize(head(states), dim=-1)
    target = torch.stack(
        [
            torch.stack([fixed_trace_code(txt, head.code_dim) for txt in row], dim=0)
            for row in traces
        ],
        dim=0,
    ).to(device=pred.device, dtype=pred.dtype)
    cosine = (pred * target).sum(dim=-1)
    loss = (1.0 - cosine).mean()
    return loss, cosine.mean().detach()
