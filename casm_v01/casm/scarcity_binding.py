from __future__ import annotations

from copy import deepcopy
import math

import torch
import torch.nn as nn

from .coordinated_binding import X10BindingModel
from .explicit_compute import ProgramBatch
from .resource_competitive_binding import binding_overlap

SPREAD_LAMBDA = 1.0
CAPACITY_LAMBDA = 1.0
NUM_SLOTS = 8

X12_MODES = (
    "canonical_functional",
    "relational_independent_overlap",
    "relational_coordinated_overlap",
    "relational_independent_scarcity",
    "relational_coordinated_scarcity",
)
LEARNED_X12_MODES = X12_MODES[1:]
SCARCITY_MODES = (
    "relational_independent_scarcity",
    "relational_coordinated_scarcity",
)
OVERLAP_MODES = (
    "relational_independent_overlap",
    "relational_coordinated_overlap",
)

_BASE_MODE = {
    "canonical_functional": "canonical_functional",
    "relational_independent_overlap": "relational_independent",
    "relational_coordinated_overlap": "relational_coordinated",
    "relational_independent_scarcity": "relational_independent",
    "relational_coordinated_scarcity": "relational_coordinated",
}


def normalized_row_spread(binding: torch.Tensor) -> torch.Tensor:
    if binding.ndim != 2 or binding.shape[1] != NUM_SLOTS:
        raise ValueError(f"expected [n,{NUM_SLOTS}] binding, got {tuple(binding.shape)}")
    safe = binding.clamp_min(1e-12)
    entropy = -(safe * safe.log()).sum(dim=1)
    return entropy.mean() / math.log(NUM_SLOTS)


def slot_capacity_overflow(binding: torch.Tensor) -> torch.Tensor:
    if binding.ndim != 2 or binding.shape[1] != NUM_SLOTS:
        raise ValueError(f"expected [n,{NUM_SLOTS}] binding, got {tuple(binding.shape)}")
    n = binding.shape[0]
    if n < 1:
        raise ValueError("at least one active variable is required")
    occupancy = binding.sum(dim=0)
    overflow = torch.relu(occupancy - 1.0)
    return overflow.square().sum() / float(n)


class X12BindingModel(nn.Module):
    """X10 relational binding with overlap or decomposed soft scarcity objective."""

    def __init__(self, *, mode: str, d_model: int = 96):
        super().__init__()
        if mode not in X12_MODES:
            raise ValueError(mode)
        self.mode = mode
        self.core = X10BindingModel(mode=_BASE_MODE[mode], d_model=d_model)

    @property
    def executor(self):
        return self.core.executor

    @property
    def binding_generator(self):
        return self.core.binding_generator

    def parameter_count(self) -> int:
        return self.core.parameter_count()

    def trainable_parameter_count(self) -> int:
        return self.core.trainable_parameter_count()

    def soft_binding(self, num_registers: int) -> torch.Tensor:
        return self.core.soft_binding(num_registers)

    def independent_argmax_binding(self, num_registers: int):
        return self.core.independent_argmax_binding(num_registers)

    def binding_stats(self, num_registers: int) -> dict[str, object]:
        stats = dict(self.core.binding_stats(num_registers))
        binding = self.soft_binding(num_registers)
        stats.update(
            mean_pairwise_overlap=float(binding_overlap(binding).detach()),
            normalized_row_spread=float(normalized_row_spread(binding).detach()),
            slot_capacity_overflow=float(slot_capacity_overflow(binding).detach()),
        )
        return stats

    def rollout_soft(self, batch: ProgramBatch) -> torch.Tensor:
        return self.core.rollout_soft(batch)

    def rollout_hard(self, batch: ProgramBatch, *, discrete_binding: bool) -> torch.Tensor:
        return self.core.rollout_hard(batch, discrete_binding=discrete_binding)

    def answer_loss(self, batch: ProgramBatch) -> torch.Tensor:
        return self.core.fixed_answer_loss(batch)

    def loss_components(self, batch: ProgramBatch) -> dict[str, torch.Tensor]:
        answer = self.answer_loss(batch)
        zero = answer.new_zeros(())
        overlap = zero
        spread = zero
        capacity = zero
        if self.mode in OVERLAP_MODES:
            overlap = binding_overlap(self.soft_binding(batch.initial.shape[1]))
        elif self.mode in SCARCITY_MODES:
            binding = self.soft_binding(batch.initial.shape[1])
            spread = normalized_row_spread(binding)
            capacity = slot_capacity_overflow(binding)
        total = answer + overlap + SPREAD_LAMBDA * spread + CAPACITY_LAMBDA * capacity
        return {
            "answer_loss": answer,
            "overlap_penalty": overlap,
            "spread_penalty": spread,
            "capacity_penalty": capacity,
            "total_loss": total,
        }


def cloned_x12_models(d_model: int = 96) -> dict[str, X12BindingModel]:
    models = {mode: X12BindingModel(mode=mode, d_model=d_model) for mode in X12_MODES}

    reference_executor = deepcopy(models["canonical_functional"].executor.state_dict())
    for model in models.values():
        model.executor.load_state_dict(reference_executor)

    reference_generator = deepcopy(models["relational_independent_overlap"].binding_generator.state_dict())
    for mode in LEARNED_X12_MODES:
        models[mode].binding_generator.load_state_dict(reference_generator)

    return models
