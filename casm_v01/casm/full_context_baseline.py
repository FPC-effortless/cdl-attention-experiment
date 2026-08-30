from __future__ import annotations

from dataclasses import replace
from typing import Dict

import torch
import torch.nn as nn

from .data import VOCAB_SIZE
from .model import Block, CASMConfig, RMSNorm


def baseline_config() -> CASMConfig:
    """~parameter-matched full-context Transformer configuration.

    Memory/compression fields are inert; this model uses only embedding,
    causal Transformer blocks, final normalization, and a tied LM head.
    """
    return replace(
        CASMConfig(vocab_size=VOCAB_SIZE),
        d_model=160,
        n_layers=5,
        n_heads=5,
        n_kv_heads=1,
        d_ff=448,
        dropout=0.0,
        use_memory=False,
        use_compression_score=False,
        compression_loss_weight=0.0,
        compression_predictor_loss_weight=0.0,
        mtp_loss_weight=0.0,
        verifier_loss_weight=0.0,
    )


class FullContextTransformer(nn.Module):
    """Ordinary causal Transformer over the complete prompt context."""

    def __init__(self, cfg: CASMConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or baseline_config()
        self.embed = nn.Embedding(self.cfg.vocab_size, self.cfg.d_model)
        self.blocks = nn.ModuleList([Block(self.cfg) for _ in range(self.cfg.n_layers)])
        self.norm = RMSNorm(self.cfg.d_model)
        self.lm_head = nn.Linear(self.cfg.d_model, self.cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.shape[1] < 1:
            raise ValueError("input_ids must have shape [batch, time] with time >= 1")
        h = self.embed(input_ids)
        for block in self.blocks:
            h = block(h)
        return self.lm_head(self.norm(h))

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def checkpoint(self, *, kind: str) -> Dict[str, object]:
        from dataclasses import asdict

        return {
            "kind": kind,
            "config": asdict(self.cfg),
            "state_dict": self.state_dict(),
            "parameters": self.parameter_count(),
        }
