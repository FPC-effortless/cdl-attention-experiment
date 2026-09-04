from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .collision_barrier_binding import collision_barrier
from .coordinated_binding import X10BindingModel, slot_descriptor
from .explicit_compute import ProgramBatch
from .scarcity_binding import normalized_row_spread
from .variable_cardinality_binding import (
    FIXED_ANSWER_REGISTER,
    NUM_CANDIDATE_SLOTS,
    variable_descriptor,
)
from .variable_contextual_data import MAX_CARDINALITY, MIN_CARDINALITY

DUAL_ROUNDS = 8
DUAL_ETA = 1.0

X16_MODES = (
    "canonical_functional",
    "dual_neutral",
    "dual_priced",
)
LEARNED_X16_MODES = X16_MODES[1:]


def projected_dual_update(
    prices: torch.Tensor,
    occupancy: torch.Tensor,
    *,
    eta: float = DUAL_ETA,
) -> torch.Tensor:
    if prices.shape != (NUM_CANDIDATE_SLOTS,) or occupancy.shape != (NUM_CANDIDATE_SLOTS,):
        raise ValueError("prices and occupancy must both be length-8 vectors")
    if eta != DUAL_ETA:
        raise ValueError(f"X16 freezes eta={DUAL_ETA}")
    return F.relu(prices + eta * (occupancy - 1.0))


def primal_dual_binding(
    base_logits: torch.Tensor,
    *,
    enabled: bool,
    rounds: int = DUAL_ROUNDS,
    eta: float = DUAL_ETA,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return final binding and persistent projected dual prices.

    Each candidate slot has unit soft capacity. Prices are generated from the current
    aggregate occupancy and are never learned parameters. This is not a matching or
    injective projection; exact symmetric rows remain symmetric.
    """
    if base_logits.ndim != 2 or base_logits.shape[1] != NUM_CANDIDATE_SLOTS:
        raise ValueError(f"expected [n,{NUM_CANDIDATE_SLOTS}] logits, got {tuple(base_logits.shape)}")
    if rounds != DUAL_ROUNDS:
        raise ValueError(f"X16 freezes exactly {DUAL_ROUNDS} dual rounds")
    if eta != DUAL_ETA:
        raise ValueError(f"X16 freezes eta={DUAL_ETA}")

    prices = base_logits.new_zeros(NUM_CANDIDATE_SLOTS)
    if enabled:
        for _ in range(rounds):
            probs = F.softmax(base_logits - prices.unsqueeze(0), dim=-1)
            occupancy = probs.sum(dim=0)
            prices = projected_dual_update(prices, occupancy, eta=eta)
    final_probs = F.softmax(base_logits - prices.unsqueeze(0), dim=-1)
    return final_probs, prices


class X16BindingModel(nn.Module):
    """Validated local executor plus neutral or persistent-dual relational binding."""

    def __init__(self, *, mode: str, d_model: int = 96):
        super().__init__()
        if mode not in X16_MODES:
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

    def base_logits(self, num_registers: int) -> torch.Tensor:
        if self.mode == "canonical_functional":
            raise RuntimeError("canonical functional mode has no learned base logits")
        if not MIN_CARDINALITY <= num_registers <= MAX_CARDINALITY:
            raise ValueError(num_registers)
        assert self.binding_generator is not None
        parameter = next(self.binding_generator.parameters())
        external_indices = torch.arange(num_registers, device=parameter.device)
        slot_indices = torch.arange(NUM_CANDIDATE_SLOTS, device=parameter.device)
        external = variable_descriptor(external_indices, num_registers, dtype=parameter.dtype)
        slots = slot_descriptor(slot_indices, dtype=parameter.dtype)
        return self.binding_generator.logits_from_descriptors(external, slots)

    def soft_binding_and_prices(self, num_registers: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "canonical_functional":
            matrix = self.executor.binding_matrix(num_registers)
            return matrix, matrix.new_zeros(NUM_CANDIDATE_SLOTS)
        logits = self.base_logits(num_registers)
        return primal_dual_binding(logits, enabled=(self.mode == "dual_priced"))

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
        overload = F.relu(occupancy - 1.0)
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


def cloned_x16_models(d_model: int = 96) -> dict[str, X16BindingModel]:
    models = {mode: X16BindingModel(mode=mode, d_model=d_model) for mode in X16_MODES}

    executor_state = deepcopy(models["canonical_functional"].executor.state_dict())
    for model in models.values():
        model.executor.load_state_dict(executor_state)

    neutral_generator = models["dual_neutral"].binding_generator
    priced_generator = models["dual_priced"].binding_generator
    assert neutral_generator is not None and priced_generator is not None
    generator_state = deepcopy(neutral_generator.state_dict())
    priced_generator.load_state_dict(generator_state)

    return models
