from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn

from .coordinated_binding import X10BindingModel
from .explicit_compute import ProgramBatch
from .scarcity_binding import normalized_row_spread, slot_capacity_overflow

BARRIER_EPSILON = 1e-3
SPREAD_LAMBDA = 1.0
BARRIER_LAMBDA = 1.0

X13_MODES = (
    "canonical_functional",
    "relational_independent_quadratic",
    "relational_coordinated_quadratic",
    "relational_independent_barrier",
    "relational_coordinated_barrier",
)
LEARNED_X13_MODES = X13_MODES[1:]
QUADRATIC_MODES = (
    "relational_independent_quadratic",
    "relational_coordinated_quadratic",
)
BARRIER_MODES = (
    "relational_independent_barrier",
    "relational_coordinated_barrier",
)

_BASE_MODE = {
    "canonical_functional": "canonical_functional",
    "relational_independent_quadratic": "relational_independent",
    "relational_coordinated_quadratic": "relational_coordinated",
    "relational_independent_barrier": "relational_independent",
    "relational_coordinated_barrier": "relational_coordinated",
}


def collision_barrier(binding: torch.Tensor, *, epsilon: float = BARRIER_EPSILON) -> torch.Tensor:
    if binding.ndim != 2 or binding.shape[1] != 8:
        raise ValueError(f"expected [n,8] binding, got {tuple(binding.shape)}")
    if binding.shape[0] < 2:
        return binding.new_zeros(())
    if not (0.0 < epsilon < 1.0):
        raise ValueError("epsilon must be in (0,1)")
    rows = []
    scale = 1.0 - epsilon
    for i in range(binding.shape[0]):
        for j in range(i + 1, binding.shape[0]):
            overlap = (binding[i] * binding[j]).sum()
            rows.append(-torch.log(1.0 - scale * overlap))
    return torch.stack(rows).mean()


class X13BindingModel(nn.Module):
    """X10 relational binding with X12 quadratic or X13 barrier resource objective."""

    def __init__(self, *, mode: str, d_model: int = 96):
        super().__init__()
        if mode not in X13_MODES:
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
            normalized_row_spread=float(normalized_row_spread(binding).detach()),
            slot_capacity_overflow=float(slot_capacity_overflow(binding).detach()),
            collision_barrier=float(collision_barrier(binding).detach()),
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
        spread = zero
        capacity = zero
        barrier = zero
        if self.mode in QUADRATIC_MODES:
            binding = self.soft_binding(batch.initial.shape[1])
            spread = normalized_row_spread(binding)
            capacity = slot_capacity_overflow(binding)
        elif self.mode in BARRIER_MODES:
            binding = self.soft_binding(batch.initial.shape[1])
            spread = normalized_row_spread(binding)
            barrier = collision_barrier(binding)
        total = answer + SPREAD_LAMBDA * spread + capacity + BARRIER_LAMBDA * barrier
        return {
            "answer_loss": answer,
            "spread_penalty": spread,
            "capacity_penalty": capacity,
            "barrier_penalty": barrier,
            "total_loss": total,
        }


def cloned_x13_models(d_model: int = 96) -> dict[str, X13BindingModel]:
    models = {mode: X13BindingModel(mode=mode, d_model=d_model) for mode in X13_MODES}

    reference_executor = deepcopy(models["canonical_functional"].executor.state_dict())
    for model in models.values():
        model.executor.load_state_dict(reference_executor)

    reference_generator = deepcopy(models["relational_independent_quadratic"].binding_generator.state_dict())
    for mode in LEARNED_X13_MODES:
        models[mode].binding_generator.load_state_dict(reference_generator)

    return models
