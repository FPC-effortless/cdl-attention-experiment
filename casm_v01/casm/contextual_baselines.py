from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .explicit_compute import NUM_REGISTERS, VALUE_MODULUS, ProgramBatch, StateCommandEncoder


class MatchedGRUProgramBaseline(nn.Module):
    def __init__(self, d_model: int = 114):
        super().__init__()
        self.encoder = StateCommandEncoder(d_model)
        self.initial = nn.Sequential(
            nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, d_model)
        )
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.head = nn.Linear(d_model, NUM_REGISTERS * VALUE_MODULUS)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def logits(self, batch: ProgramBatch) -> torch.Tensor:
        h0 = self.initial(self.encoder.encode_state(batch.initial))[None, :, :]
        commands = []
        for t in range(batch.depth):
            commands.append(
                self.encoder.encode_command(
                    batch.commands[:, t], batch.arg_a[:, t], batch.arg_b[:, t], batch.dst[:, t]
                )
            )
        hidden, _ = self.gru(torch.stack(commands, dim=1), h0)
        return self.head(hidden).view(
            batch.initial.shape[0], batch.depth, NUM_REGISTERS, VALUE_MODULUS
        )

    def training_loss(self, batch: ProgramBatch) -> torch.Tensor:
        logits = self.logits(batch)
        return F.cross_entropy(logits.reshape(-1, VALUE_MODULUS), batch.target_states.reshape(-1))

    @torch.no_grad()
    def rollout(self, batch: ProgramBatch) -> torch.Tensor:
        return self.logits(batch).argmax(dim=-1)


class MatchedTransformerProgramBaseline(nn.Module):
    def __init__(self, d_model: int = 92, nhead: int = 4, layers: int = 2, max_depth: int = 128):
        super().__init__()
        self.encoder = StateCommandEncoder(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2 * d_model,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.position = nn.Embedding(max_depth + 1, d_model)
        self.head = nn.Linear(d_model, NUM_REGISTERS * VALUE_MODULUS)
        self.max_depth = max_depth

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def logits(self, batch: ProgramBatch) -> torch.Tensor:
        if batch.depth > self.max_depth:
            raise ValueError(f"depth {batch.depth} exceeds max_depth {self.max_depth}")
        prefix = self.encoder.encode_state(batch.initial)
        commands = []
        for t in range(batch.depth):
            commands.append(
                self.encoder.encode_command(
                    batch.commands[:, t], batch.arg_a[:, t], batch.arg_b[:, t], batch.dst[:, t]
                )
            )
        tokens = torch.cat([prefix[:, None, :], torch.stack(commands, dim=1)], dim=1)
        positions = torch.arange(batch.depth + 1, device=tokens.device)[None, :]
        tokens = tokens + self.position(positions)
        causal = torch.full(
            (batch.depth + 1, batch.depth + 1), float("-inf"), device=tokens.device
        )
        causal = torch.triu(causal, diagonal=1)
        hidden = self.transformer(tokens, mask=causal)
        return self.head(hidden[:, 1:, :]).view(
            batch.initial.shape[0], batch.depth, NUM_REGISTERS, VALUE_MODULUS
        )

    def training_loss(self, batch: ProgramBatch) -> torch.Tensor:
        logits = self.logits(batch)
        return F.cross_entropy(logits.reshape(-1, VALUE_MODULUS), batch.target_states.reshape(-1))

    @torch.no_grad()
    def rollout(self, batch: ProgramBatch) -> torch.Tensor:
        return self.logits(batch).argmax(dim=-1)
