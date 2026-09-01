from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn

from .coordinated_binding import X10BindingModel
from .explicit_compute import ProgramBatch

RESOURCE_COMPETITION_LAMBDA = 1.0
X11_MODES = (
    "canonical_functional",
    "relational_independent_no_competition",
    "relational_coordinated_no_competition",
    "relational_independent_competitive",
    "relational_coordinated_competitive",
)
LEARNED_X11_MODES = X11_MODES[1:]
COMPETITIVE_MODES = (
    "relational_independent_competitive",
    "relational_coordinated_competitive",
)

_BASE_MODE = {
    "canonical_functional": "canonical_functional",
    "relational_independent_no_competition": "relational_independent",
    "relational_coordinated_no_competition": "relational_coordinated",
    "relational_independent_competitive": "relational_independent",
    "relational_coordinated_competitive": "relational_coordinated",
}


def binding_overlap(binding: torch.Tensor) -> torch.Tensor:
    """Mean pairwise overlap between row-stochastic variable-to-slot bindings."""
    if binding.ndim != 2:
        raise ValueError(f"expected rank-2 binding, got shape={tuple(binding.shape)}")
    n = binding.shape[0]
    if n < 2:
        raise ValueError("resource competition requires at least two active variables")
    gram = binding @ binding.transpose(0, 1)
    upper = torch.triu(gram, diagonal=1)
    return upper.sum() * (2.0 / float(n * (n - 1)))


class X11BindingModel(nn.Module):
    """X10 relational model plus an optional soft resource-competition loss."""

    def __init__(self, *, mode: str, d_model: int = 96):
        super().__init__()
        if mode not in X11_MODES:
            raise ValueError(mode)
        self.mode = mode
        self.competitive = mode in COMPETITIVE_MODES
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
        stats["mean_pairwise_overlap"] = float(binding_overlap(binding).detach())
        return stats

    def rollout_soft(self, batch: ProgramBatch) -> torch.Tensor:
        return self.core.rollout_soft(batch)

    def rollout_hard(self, batch: ProgramBatch, *, discrete_binding: bool) -> torch.Tensor:
        return self.core.rollout_hard(batch, discrete_binding=discrete_binding)

    def answer_loss(self, batch: ProgramBatch) -> torch.Tensor:
        return self.core.fixed_answer_loss(batch)

    def loss_components(self, batch: ProgramBatch) -> dict[str, torch.Tensor]:
        answer = self.answer_loss(batch)
        if self.mode == "canonical_functional":
            overlap = answer.new_zeros(())
        else:
            overlap = binding_overlap(self.soft_binding(batch.initial.shape[1]))
        weighted = overlap * (RESOURCE_COMPETITION_LAMBDA if self.competitive else 0.0)
        total = answer + weighted
        return {
            "answer_loss": answer,
            "overlap_penalty": overlap,
            "weighted_overlap": weighted,
            "total_loss": total,
        }


def cloned_x11_models(d_model: int = 96) -> dict[str, X11BindingModel]:
    """Build paired X11 regimes with identical executor and relational initialization."""
    models = {mode: X11BindingModel(mode=mode, d_model=d_model) for mode in X11_MODES}

    reference_executor = deepcopy(models["canonical_functional"].executor.state_dict())
    for model in models.values():
        model.executor.load_state_dict(reference_executor)

    reference_generator = deepcopy(
        models["relational_independent_no_competition"].binding_generator.state_dict()
    )
    for mode in LEARNED_X11_MODES:
        models[mode].binding_generator.load_state_dict(reference_generator)

    return models
