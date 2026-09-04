from __future__ import annotations

import math
from copy import deepcopy

import torch
import torch.nn as nn

from .collision_barrier_binding import collision_barrier
from .coordinated_binding import X10BindingModel, slot_descriptor
from .explicit_compute import ProgramBatch
from .primal_dual_capacity_binding import primal_dual_binding
from .scarcity_binding import normalized_row_spread
from .variable_cardinality_binding import (
    FIXED_ANSWER_REGISTER,
    NUM_CANDIDATE_SLOTS,
    variable_descriptor,
)
from .variable_contextual_data import MAX_CARDINALITY, MIN_CARDINALITY

X18_MODES = (
    "canonical_functional",
    "relative_descriptor",
    "global_descriptor",
)
LEARNED_X18_MODES = X18_MODES[1:]


def global_variable_descriptor(
    external_indices: torch.Tensor,
    num_registers: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Frozen X18 global external coordinate, independent of active cardinality."""
    if not MIN_CARDINALITY <= num_registers <= MAX_CARDINALITY:
        raise ValueError(num_registers)
    if external_indices.ndim != 1:
        raise ValueError("external_indices must be rank-1")
    indices_long = external_indices.to(dtype=torch.long)
    if ((indices_long < 0) | (indices_long >= num_registers)).any():
        raise ValueError("external index is outside active cardinality")

    e = external_indices.to(dtype=dtype)
    workspace = float(NUM_CANDIDATE_SLOTS)
    global_position = e / float(NUM_CANDIDATE_SLOTS - 1)
    normalized_workspace = torch.ones_like(e)
    phase1 = math.pi * e / workspace
    phase2 = 2.0 * math.pi * e / workspace
    bits = []
    for bit in range(3):
        one = ((indices_long >> bit) & 1).to(dtype=dtype)
        bits.append(one * 2.0 - 1.0)
    return torch.stack(
        [
            global_position,
            normalized_workspace,
            torch.sin(phase1),
            torch.cos(phase1),
            torch.sin(phase2),
            torch.cos(phase2),
            bits[0],
            bits[1],
            bits[2],
        ],
        dim=-1,
    )


class X18BindingModel(nn.Module):
    """Validated local executor plus relative/global descriptor and X16 dual pricing."""

    def __init__(self, *, mode: str, d_model: int = 96):
        super().__init__()
        if mode not in X18_MODES:
            raise ValueError(mode)
        self.mode = mode
        base_mode = "canonical_functional" if mode == "canonical_functional" else "relational_independent"
        self.core = X10BindingModel(mode=base_mode, d_model=d_model)

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

    def external_descriptor(self, external_indices: torch.Tensor, num_registers: int, *, dtype: torch.dtype) -> torch.Tensor:
        if self.mode == "relative_descriptor":
            return variable_descriptor(external_indices, num_registers, dtype=dtype)
        if self.mode == "global_descriptor":
            return global_variable_descriptor(external_indices, num_registers, dtype=dtype)
        raise RuntimeError("canonical functional mode has no learned external descriptor")

    def base_logits(self, num_registers: int) -> torch.Tensor:
        if self.mode == "canonical_functional":
            raise RuntimeError("canonical functional mode has no learned base logits")
        if not MIN_CARDINALITY <= num_registers <= MAX_CARDINALITY:
            raise ValueError(num_registers)
        assert self.binding_generator is not None
        parameter = next(self.binding_generator.parameters())
        external_indices = torch.arange(num_registers, device=parameter.device)
        slot_indices = torch.arange(NUM_CANDIDATE_SLOTS, device=parameter.device)
        external = self.external_descriptor(external_indices, num_registers, dtype=parameter.dtype)
        slots = slot_descriptor(slot_indices, dtype=parameter.dtype)
        return self.binding_generator.logits_from_descriptors(external, slots)

    def soft_binding_and_prices(self, num_registers: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "canonical_functional":
            matrix = self.executor.binding_matrix(num_registers)
            return matrix, matrix.new_zeros(NUM_CANDIDATE_SLOTS)
        return primal_dual_binding(self.base_logits(num_registers), enabled=True)

    def soft_binding(self, num_registers: int) -> torch.Tensor:
        return self.soft_binding_and_prices(num_registers)[0]

    @torch.no_grad()
    def independent_argmax_binding(self, num_registers: int) -> tuple[torch.Tensor, list[int]]:
        matrix = self.soft_binding(num_registers).detach()
        assignment = matrix.argmax(dim=1)
        rows = torch.arange(num_registers, device=matrix.device)
        projected = torch.zeros_like(matrix)
        projected[rows, assignment] = 1.0
        return projected, assignment.cpu().tolist()

    def binding_matrix(self, num_registers: int, *, discrete: bool) -> torch.Tensor:
        if self.mode == "canonical_functional":
            return self.executor.binding_matrix(num_registers)
        if not discrete:
            return self.soft_binding(num_registers)
        projected, _ = self.independent_argmax_binding(num_registers)
        return projected

    def binding_stats(self, num_registers: int) -> dict[str, object]:
        matrix, prices = self.soft_binding_and_prices(num_registers)
        safe = matrix.clamp_min(1e-12)
        occupancy = matrix.sum(dim=0)
        overload = torch.relu(occupancy - 1.0)
        _, assignment = self.independent_argmax_binding(num_registers)
        unique = len(set(assignment))
        return {
            "num_registers": num_registers,
            "matrix": matrix.detach().cpu().tolist(),
            "row_max_mean": float(matrix.max(dim=1).values.mean().detach()),
            "row_entropy_mean": float((-(safe * safe.log()).sum(dim=1).mean()).detach()),
            "independent_argmax_assignment": assignment,
            "independent_argmax_unique_slot_count": unique,
            "independent_argmax_collision_count": num_registers - unique,
            "max_row_sum_error": float((matrix.sum(dim=1) - 1.0).abs().max().detach()),
            "max_column_occupancy": float(occupancy.max().detach()),
            "min_column_occupancy": float(occupancy.min().detach()),
            "total_capacity_overload": float(overload.sum().detach()),
            "max_capacity_overload": float(overload.max().detach()),
            "dual_prices": prices.detach().cpu().tolist(),
            "max_dual_price": float(prices.max().detach()),
            "mean_dual_price": float(prices.mean().detach()),
            "total_binding_mass": float(matrix.sum().detach()),
            "normalized_row_spread": float(normalized_row_spread(matrix).detach()),
            "collision_barrier": float(collision_barrier(matrix).detach()),
        }

    def rollout_soft(self, batch: ProgramBatch) -> torch.Tensor:
        binding = self.binding_matrix(batch.initial.shape[1], discrete=False)
        return self.executor.rollout_soft_with_binding(batch, binding)

    def answer_loss(self, batch: ProgramBatch) -> torch.Tensor:
        decoded = self.rollout_soft(batch)[:, -1, FIXED_ANSWER_REGISTER].clamp_min(1e-12)
        target = batch.target_states[:, -1, FIXED_ANSWER_REGISTER]
        probability = decoded.gather(1, target[:, None]).squeeze(1)
        return -probability.log().mean()

    def loss_components(self, batch: ProgramBatch) -> dict[str, torch.Tensor]:
        answer = self.answer_loss(batch)
        zero = answer.new_zeros(())
        spread = zero
        barrier = zero
        if self.mode != "canonical_functional":
            binding = self.soft_binding(batch.initial.shape[1])
            spread = normalized_row_spread(binding)
            barrier = collision_barrier(binding)
        total = answer + spread + barrier
        return {
            "answer_loss": answer,
            "spread_penalty": spread,
            "barrier_penalty": barrier,
            "total_loss": total,
        }

    @torch.no_grad()
    def rollout_hard(self, batch: ProgramBatch, *, discrete_binding: bool) -> torch.Tensor:
        binding = self.binding_matrix(batch.initial.shape[1], discrete=discrete_binding)
        return self.executor.rollout_hard_with_binding(batch, binding)


def cloned_x18_models(d_model: int = 96) -> dict[str, X18BindingModel]:
    models = {mode: X18BindingModel(mode=mode, d_model=d_model) for mode in X18_MODES}

    executor_state = deepcopy(models["canonical_functional"].executor.state_dict())
    for model in models.values():
        model.executor.load_state_dict(executor_state)

    relative = models["relative_descriptor"].binding_generator
    global_model = models["global_descriptor"].binding_generator
    assert relative is not None and global_model is not None
    generator_state = deepcopy(relative.state_dict())
    global_model.load_state_dict(generator_state)

    return models
