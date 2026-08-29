from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CASMConfig:
    vocab_size: int = 260
    d_model: int = 192
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int = 2
    d_ff: int = 512
    chunk_size: int = 32
    memory_slots: int = 8
    state_slots: int = 2
    memory_dim: int = 96
    dropout: float = 0.0
    rope_base: float = 10000.0
    mtp_horizons: int = 3
    compression_future_tokens: int = 16
    compression_temperature: float = 0.5
    compression_loss_weight: float = 0.15
    compression_predictor_loss_weight: float = 0.10
    mtp_loss_weight: float = 0.15
    verifier_loss_weight: float = 0.10
    use_compression_score: bool = True
    use_memory: bool = True

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        scale = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_float * scale).to(dtype) * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.repeat_interleave(freqs, 2, dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + rotate_half(x) * sin


class LocalGQAAttention(nn.Module):
    def __init__(self, cfg: CASMConfig):
        super().__init__()
        assert cfg.n_heads % cfg.n_kv_heads == 0
        self.cfg = cfg
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.rope_base)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.cfg.n_heads, self.cfg.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.cfg.n_kv_heads, self.cfg.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.cfg.n_kv_heads, self.cfg.head_dim).transpose(1, 2)
        cos, sin = self.rope(t, x.device, q.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        repeat = self.cfg.n_heads // self.cfg.n_kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).contiguous().view(b, t, self.cfg.d_model)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: CASMConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: CASMConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = LocalGQAAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class CompressionRouter(nn.Module):
    def __init__(self, cfg: CASMConfig):
        super().__init__()
        d = cfg.memory_dim
        self.q = nn.Linear(cfg.d_model, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.score_mlp = nn.Sequential(nn.Linear(4 * d, d), nn.SiLU(), nn.Linear(d, 1, bias=False))
        self.value = nn.Linear(d, cfg.d_model, bias=False)
        self.base_future = nn.Linear(cfg.d_model, cfg.compression_future_tokens * cfg.vocab_size, bias=False)
        self.future_delta = nn.Sequential(
            nn.Linear(cfg.d_model + d, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.compression_future_tokens * cfg.vocab_size, bias=False),
        )
        self.cfg = cfg

    def scores(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        q = F.normalize(self.q(query), dim=-1)
        k = F.normalize(self.k(memory), dim=-1)
        if q.ndim == 2:
            dot = torch.einsum("bd,bmd->bm", q, k)
            if not self.cfg.use_compression_score:
                return dot
            qe = q[:, None, :].expand_as(k)
            feat = torch.cat([qe, k, qe * k, (qe - k).abs()], dim=-1)
        else:
            dot = torch.einsum("btd,bmd->btm", q, k)
            if not self.cfg.use_compression_score:
                return dot
            qe = q[:, :, None, :].expand(-1, -1, k.shape[1], -1)
            ke = k[:, None, :, :].expand(-1, q.shape[1], -1, -1)
            feat = torch.cat([qe, ke, qe * ke, (qe - ke).abs()], dim=-1)
        return dot + self.score_mlp(feat).squeeze(-1)

    def retrieve(self, query: torch.Tensor, memory: torch.Tensor, valid_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.scores(query, memory)
        if scores.ndim == 2:
            weights = F.softmax(scores.masked_fill(~valid_mask, -1e9), dim=-1)
            retrieved_mem = torch.einsum("bm,bmd->bd", weights, memory)
        else:
            mask = valid_mask[:, None, :].expand_as(scores)
            weights = F.softmax(scores.masked_fill(~mask, -1e9), dim=-1)
            retrieved_mem = torch.einsum("btm,bmd->btd", weights, memory)
        return self.value(retrieved_mem), scores

    def compression_target(self, query: torch.Tensor, memory: torch.Tensor, future_tokens: torch.Tensor, valid_mask: torch.Tensor):
        b, m, _ = memory.shape
        k = min(self.cfg.compression_future_tokens, future_tokens.shape[1])
        if k <= 0:
            z = torch.zeros((b, m), device=query.device, dtype=query.dtype)
            return F.softmax(z, dim=-1), z, z.mean()
        fut = future_tokens[:, :k]
        token_mask = (fut != 256).to(query.dtype)
        denom = token_mask.sum(dim=-1).clamp_min(1.0)
        base_logits = self.base_future(query).view(b, self.cfg.compression_future_tokens, self.cfg.vocab_size)[:, :k]
        base_tok_nll = F.cross_entropy(base_logits.transpose(1, 2), fut, reduction="none", ignore_index=256)
        base_nll = (base_tok_nll * token_mask).sum(dim=-1) / denom
        qe = query[:, None, :].expand(b, m, query.shape[-1])
        cond_in = torch.cat([qe, memory], dim=-1)
        delta_logits = self.future_delta(cond_in).view(b, m, self.cfg.compression_future_tokens, self.cfg.vocab_size)[:, :, :k]
        cond_logits = base_logits[:, None, :, :] + delta_logits
        targets = fut[:, None, :].expand(b, m, k)
        cond_tok_nll = F.cross_entropy(cond_logits.reshape(b * m * k, self.cfg.vocab_size), targets.reshape(-1), reduction="none", ignore_index=256).view(b, m, k)
        cond_nll = (cond_tok_nll * token_mask[:, None, :]).sum(dim=-1) / denom[:, None]
        gain = (base_nll[:, None] - cond_nll).masked_fill(~valid_mask, -1e9)
        target = F.softmax(gain.detach() / self.cfg.compression_temperature, dim=-1)
        valid_float = valid_mask.to(cond_nll.dtype)
        cond_pred_loss = (cond_nll * valid_float).sum() / valid_float.sum().clamp_min(1.0)
        return target, gain, base_nll.mean() + cond_pred_loss


class PersistentState(nn.Module):
    def __init__(self, cfg: CASMConfig):
