"""Experimental decoder shell. No upstream trainer or engine compatibility implied.

The attention-free mixer is injected per layer. Shared decoder conventions follow
karpathy/nanochat (MIT), pinned in README.md.
"""
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor


def norm(x: Tensor) -> Tensor:
    return F.rms_norm(x, (x.size(-1),))


@dataclass(frozen=True)
class Config:
    vocab_size: int
    width: int = 64
    layers: int = 2
    vocab_alignment: int = 64

    def __post_init__(self) -> None:
        if any(type(x) is not int or x <= 0 for x in
               (self.vocab_size, self.width, self.layers, self.vocab_alignment)):
            raise ValueError("decoder geometry must use positive integers")
        if self.width < 24:
            raise ValueError("the upstream smear gate requires width >= 24")


@dataclass(frozen=True)
class DecodeState:
    mixer_states: tuple[Any, ...]
    previous_embedding: Tensor
    tokens_seen: int


class LocalOnly(nn.Module):
    """Trainable causal-convolution reference with no communication state."""
    def __init__(self, width: int, lookback: int = 3) -> None:
        super().__init__()
        if lookback < 1:
            raise ValueError("lookback must be positive")
        self.width, self.lookback = width, lookback
        self.conv = nn.Conv1d(width, width, lookback, groups=width, bias=False)
        self.out = nn.Linear(width, width, bias=False)

    def forward(self, x: Tensor, state: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        n = self.lookback - 1
        if state is None:
            state = x.new_zeros((x.size(0), n, self.width))
        if state.shape != (x.size(0), n, self.width) or state.device != x.device or state.dtype != x.dtype:
            raise ValueError("local state batch, device, or dtype mismatch")
        joined = torch.cat((state, x), dim=1)
        y = self.conv(joined.transpose(1, 2)).transpose(1, 2)
        # A view of joined would keep all chunk activations alive in inference.
        next_state = joined[:, -n:].clone() if n else x.new_empty((x.size(0), 0, self.width))
        return self.out(y), next_state


class Decoder(nn.Module):
    def __init__(self, cfg: Config, make_mixer: Callable[[int], nn.Module] = LocalOnly) -> None:
        super().__init__()
        self.cfg = cfg
        padded = ((cfg.vocab_size + cfg.vocab_alignment - 1) // cfg.vocab_alignment) * cfg.vocab_alignment
        self.embedding = nn.Embedding(padded, cfg.width)
        self.mixers = nn.ModuleList([make_mixer(cfg.width) for _ in range(cfg.layers)])
        self.mlps = nn.ModuleList([nn.ModuleList([
            nn.Linear(cfg.width, 4 * cfg.width, bias=False),
            nn.Linear(4 * cfg.width, cfg.width, bias=False),
        ]) for _ in range(cfg.layers)])
        self.output = nn.Linear(cfg.width, padded, bias=False)
        self.resid = nn.Parameter(torch.ones(cfg.layers))
        self.x0_gain = nn.Parameter(torch.zeros(cfg.layers))
        self.smear_gate = nn.Linear(24, 1, bias=False)
        self.smear_gain = nn.Parameter(torch.zeros(()))
        self.backout = nn.Parameter(torch.tensor(0.2))

    def forward(self, tokens: Tensor, state: Optional[DecodeState] = None,
                targets: Optional[Tensor] = None, *, loss_reduction: str = "mean"
                ) -> tuple[Tensor, DecodeState]:
        """Consume only given tokens and return logits (or CE) with a new state.

        None starts a fresh stream. Batches must have aligned stream boundaries.
        Caller owns document resets. A state is only for its originating model
        and batch order; cross-model/batch reuse is unsupported.
        """
        if tokens.ndim != 2 or tokens.dtype != torch.long or min(tokens.shape) == 0:
            raise ValueError("tokens must be a nonempty [B,T] long tensor")
        if targets is not None and (targets.shape != tokens.shape or targets.dtype != torch.long):
            raise ValueError("targets must match token geometry and dtype")
        if ((tokens < 0) | (tokens >= self.cfg.vocab_size)).any():
            raise ValueError("tokens outside the configured vocabulary")
        if targets is not None and loss_reduction == "mean" and not targets.ne(-1).any():
            raise ValueError("mean loss has no observed target")
        e = norm(self.embedding(tokens))
        if state is None:
            previous = e.new_zeros(e.size(0), 1, e.size(-1))
            mixer_states = (None,) * self.cfg.layers
            seen = 0
        else:
            previous, mixer_states, seen = state.previous_embedding, state.mixer_states, state.tokens_seen
            if (len(mixer_states) != self.cfg.layers or type(seen) is not int or seen <= 0
                or previous.shape != (e.size(0), 1, e.size(-1))
                or previous.dtype != e.dtype or previous.device != e.device):
                raise ValueError("decoder state does not match batch/model geometry")
        prev = torch.cat((previous, e[:, :-1]), dim=1)
        x = e + self.smear_gain * torch.sigmoid(self.smear_gate(e[..., :24])) * prev
        x0 = x
        backout = None
        updated = []
        for i, (mixer, (fc, proj), layer_state) in enumerate(zip(self.mixers, self.mlps, mixer_states)):
            x = self.resid[i] * x + self.x0_gain[i] * x0
            delta, next_state = mixer(norm(x), layer_state)
            if delta.shape != x.shape:
                raise ValueError("interaction returned invalid residual geometry")
            x = x + delta
            x = x + proj(F.relu(fc(norm(x))).square())
            if i == self.cfg.layers // 2:
                backout = x
            updated.append(next_state)
        if backout is None:
            raise RuntimeError("no backout layer")
        x = x - self.backout * backout
        logits = self.output(norm(x))[..., :self.cfg.vocab_size].float()
        logits = 15.0 * torch.tanh(logits / 15.0)
        out = logits if targets is None else F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
            ignore_index=-1, reduction=loss_reduction)
        return out, DecodeState(tuple(updated), e[:, -1:].clone(), seen + tokens.size(1))
