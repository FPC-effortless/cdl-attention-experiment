from __future__ import annotations

from copy import deepcopy
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .explicit_compute import NUM_OPERATORS, NUM_REGISTERS, VALUE_MODULUS, ProgramBatch


class SoftExplicitTransitionModel(nn.Module):
    """Differentiable explicit-state transition model with no teacher forcing.

    State is represented as a categorical distribution over values for each register. Training
    rolls the model's own predicted state forward. Intermediate target states are consulted only
    at supervision points selected by the loss regime.
    """

    def __init__(self, d_model: int = 96):
        super().__init__()
        self.d_model = d_model
        self.value = nn.Embedding(VALUE_MODULUS, d_model)
        self.register = nn.Embedding(NUM_REGISTERS, d_model)
        self.command = nn.Embedding(NUM_OPERATORS, d_model)
        self.state_proj = nn.Sequential(
            nn.Linear(NUM_REGISTERS * d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.command_proj = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.transition = nn.Sequential(
            nn.Linear(6 * d_model, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, VALUE_MODULUS),
        )

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def initial_probs(self, initial: torch.Tensor) -> torch.Tensor:
        return F.one_hot(initial, VALUE_MODULUS).to(dtype=self.value.weight.dtype)

    def expected_values(self, probs: torch.Tensor) -> torch.Tensor:
        return probs @ self.value.weight

    def encode_state(self, probs: torch.Tensor) -> torch.Tensor:
        batch = probs.shape[0]
        values = self.expected_values(probs)
        registers = torch.arange(NUM_REGISTERS, device=probs.device)[None, :].expand(batch, -1)
        encoded = values + self.register(registers)
        return self.state_proj(encoded.flatten(1))

    def encode_command(self, command, a, b, dst):
        return self.command_proj(
            torch.cat(
                [
                    self.command(command),
                    self.register(a),
                    self.register(b),
                    self.register(dst),
                ],
                dim=-1,
            )
        )

    @staticmethod
    def _gather_register(probs: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        batch = torch.arange(probs.shape[0], device=probs.device)
        return probs[batch, index]

    def step_logits(self, probs, command, a, b, dst):
        state_repr = self.encode_state(probs)
        command_repr = self.encode_command(command, a, b, dst)
        value_a = self._gather_register(probs, a) @ self.value.weight
        value_b = self._gather_register(probs, b) @ self.value.weight
        value_dst = self._gather_register(probs, dst) @ self.value.weight
        register_dst = self.register(dst)
        features = torch.cat(
            [state_repr, command_repr, value_a, value_b, value_dst, register_dst], dim=-1
        )
        return self.transition(features)

    @staticmethod
    def update_state(probs: torch.Tensor, dst: torch.Tensor, new_value_probs: torch.Tensor) -> torch.Tensor:
        mask = F.one_hot(dst, NUM_REGISTERS).to(dtype=probs.dtype)[:, :, None]
        return probs * (1.0 - mask) + new_value_probs[:, None, :] * mask

    def rollout_soft(self, batch: ProgramBatch) -> torch.Tensor:
        probs = self.initial_probs(batch.initial)
        states = []
        for t in range(batch.depth):
            logits = self.step_logits(
                probs,
                batch.commands[:, t],
                batch.arg_a[:, t],
                batch.arg_b[:, t],
                batch.dst[:, t],
            )
            probs = self.update_state(probs, batch.dst[:, t], F.softmax(logits, dim=-1))
            states.append(probs)
        return torch.stack(states, dim=1)

    @torch.no_grad()
    def rollout_hard(self, batch: ProgramBatch) -> torch.Tensor:
        probs = self.initial_probs(batch.initial)
        states = []
        for t in range(batch.depth):
            logits = self.step_logits(
                probs,
                batch.commands[:, t],
                batch.arg_a[:, t],
                batch.arg_b[:, t],
                batch.dst[:, t],
            )
            value = logits.argmax(dim=-1)
            hard = F.one_hot(value, VALUE_MODULUS).to(dtype=probs.dtype)
            probs = self.update_state(probs, batch.dst[:, t], hard)
            states.append(probs.argmax(dim=-1))
        return torch.stack(states, dim=1)

    @staticmethod
    def supervision_indices(depth: int, regime: str) -> list[int]:
        if regime == "process":
            return list(range(depth))
        if regime == "quarter":
            indices = list(range(3, depth, 4))
            if depth - 1 not in indices:
                indices.append(depth - 1)
            return sorted(set(indices))
        if regime == "final":
            return [depth - 1]
        raise ValueError(regime)

    def training_loss(self, batch: ProgramBatch, regime: str) -> Dict[str, torch.Tensor]:
        predicted = self.rollout_soft(batch)
        indices = self.supervision_indices(batch.depth, regime)
        losses = []
        for index in indices:
            probs = predicted[:, index].clamp_min(1e-9)
            target = batch.target_states[:, index]
            target_prob = probs.gather(2, target[:, :, None]).squeeze(-1)
            losses.append(-target_prob.log().mean())
        loss = torch.stack(losses).mean()
        entropy = -(predicted[:, -1].clamp_min(1e-9) * predicted[:, -1].clamp_min(1e-9).log()).sum(-1).mean()
        return {
            "loss": loss,
            "final_entropy": entropy.detach(),
            "supervised_steps": torch.tensor(float(len(indices)), device=loss.device),
        }


def cloned_regime_models(seed_model: SoftExplicitTransitionModel) -> dict[str, SoftExplicitTransitionModel]:
    state = deepcopy(seed_model.state_dict())
    models = {}
    for regime in ("process", "quarter", "final"):
        model = SoftExplicitTransitionModel(d_model=seed_model.d_model)
        model.load_state_dict(state)
        models[regime] = model
    return models
