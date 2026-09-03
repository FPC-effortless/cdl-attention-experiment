from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cardinality_valid_executor import LocalEquivariantTransitionModel
from .collision_barrier_binding import BARRIER_EPSILON, collision_barrier
from .coordinated_binding import RelationalBindingGenerator, slot_descriptor
from .explicit_compute import ProgramBatch
from .scarcity_binding import normalized_row_spread, slot_capacity_overflow
from .variable_cardinality_binding import (
    FIXED_ANSWER_REGISTER,
    NUM_CANDIDATE_SLOTS,
    variable_descriptor,
)
from .variable_contextual_data import MAX_CARDINALITY, MIN_CARDINALITY

REFINEMENT_ROUNDS = 8
SPREAD_LAMBDA = 1.0
BARRIER_LAMBDA = 1.0

X14_MODES = (
    "canonical_functional",
    "x13_one_shot_barrier",
    "iterative_no_occupancy",
    "iterative_occupancy",
)
LEARNED_X14_MODES = X14_MODES[1:]
ITERATIVE_MODES = ("iterative_no_occupancy", "iterative_occupancy")


class OccupancyRefinementBindingGenerator(nn.Module):
    """Permutation-equivariant soft allocator with generated occupancy feedback."""

    def __init__(
        self,
        d_model: int = 96,
        *,
        use_occupancy: bool,
        refinement_rounds: int = REFINEMENT_ROUNDS,
    ):
        super().__init__()
        if refinement_rounds != REFINEMENT_ROUNDS:
            raise ValueError(f"X14 freezes exactly {REFINEMENT_ROUNDS} refinement rounds")
        self.d_model = int(d_model)
        self.use_occupancy = bool(use_occupancy)
        self.refinement_rounds = int(refinement_rounds)
        self.base = RelationalBindingGenerator(d_model=d_model, coordinated=False)
        self.refiner = nn.Sequential(
            nn.Linear(4, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        last = self.refiner[-1]
        assert isinstance(last, nn.Linear)
        nn.init.normal_(last.weight, mean=0.0, std=0.01)
        nn.init.zeros_(last.bias)

    @staticmethod
    def generated_other_occupancy(probabilities: torch.Tensor) -> torch.Tensor:
        if probabilities.ndim != 2 or probabilities.shape[1] != NUM_CANDIDATE_SLOTS:
            raise ValueError(
                f"expected [n,{NUM_CANDIDATE_SLOTS}] probabilities, got {tuple(probabilities.shape)}"
            )
        return probabilities.sum(dim=0, keepdim=True) - probabilities

    def logits_from_descriptors(
        self,
        external_descriptors: torch.Tensor,
        slot_descriptors: torch.Tensor,
    ) -> torch.Tensor:
        base_logits = self.base.logits_from_descriptors(external_descriptors, slot_descriptors)
        logits = base_logits
        for _ in range(self.refinement_rounds):
            probabilities = F.softmax(logits, dim=-1)
            occupancy = self.generated_other_occupancy(probabilities)
            if not self.use_occupancy:
                occupancy = torch.zeros_like(occupancy)
            features = torch.stack(
                [base_logits, logits, probabilities, occupancy],
                dim=-1,
            )
            delta = self.refiner(features).squeeze(-1)
            logits = logits + delta
        return logits

    def probabilities_from_descriptors(
        self,
        external_descriptors: torch.Tensor,
        slot_descriptors: torch.Tensor,
    ) -> torch.Tensor:
        return F.softmax(
            self.logits_from_descriptors(external_descriptors, slot_descriptors),
            dim=-1,
        )

    def forward(self, num_registers: int) -> torch.Tensor:
        if not MIN_CARDINALITY <= num_registers <= MAX_CARDINALITY:
            raise ValueError(num_registers)
        parameter = next(self.parameters())
        external_indices = torch.arange(num_registers, device=parameter.device)
        slot_indices = torch.arange(NUM_CANDIDATE_SLOTS, device=parameter.device)
        external = variable_descriptor(
            external_indices,
            num_registers,
            dtype=parameter.dtype,
        )
        slots = slot_descriptor(slot_indices, dtype=parameter.dtype)
        return self.probabilities_from_descriptors(external, slots)


class X14BindingModel(nn.Module):
    """Validated local executor plus one X14 allocation regime."""

    def __init__(self, *, mode: str, d_model: int = 96):
        super().__init__()
        if mode not in X14_MODES:
            raise ValueError(mode)
        self.mode = mode
        self.d_model = int(d_model)
        self.executor = LocalEquivariantTransitionModel(d_model=d_model)
        if mode == "x13_one_shot_barrier":
            self.binding_generator: nn.Module | None = RelationalBindingGenerator(
                d_model=d_model,
                coordinated=False,
            )
        elif mode in ITERATIVE_MODES:
            self.binding_generator = OccupancyRefinementBindingGenerator(
                d_model=d_model,
                use_occupancy=(mode == "iterative_occupancy"),
            )
        else:
            self.binding_generator = None

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def soft_binding(self, num_registers: int) -> torch.Tensor:
        if self.mode == "canonical_functional":
            return self.executor.binding_matrix(num_registers)
        assert self.binding_generator is not None
        return self.binding_generator(num_registers)

    @torch.no_grad()
    def independent_argmax_binding(self, num_registers: int) -> tuple[torch.Tensor, list[int]]:
        matrix = self.soft_binding(num_registers).detach()
        assignment = matrix.argmax(dim=1)
        rows = torch.arange(num_registers, device=matrix.device)
        projected = torch.zeros_like(matrix)
        projected[rows, assignment] = 1.0
        return projected, assignment.cpu().tolist()

    def binding_matrix(self, num_registers: int, *, discrete: bool) -> torch.Tensor:
        if not discrete:
            return self.soft_binding(num_registers)
        if self.mode == "canonical_functional":
            return self.executor.binding_matrix(num_registers)
        projected, _ = self.independent_argmax_binding(num_registers)
        return projected

    def binding_stats(self, num_registers: int) -> dict[str, object]:
        matrix = self.soft_binding(num_registers)
        safe = matrix.clamp_min(1e-12)
        occupancy = matrix.sum(dim=0)
        _, assignment = self.independent_argmax_binding(num_registers)
        unique = len(set(assignment))
        stats: dict[str, object] = {
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
            "total_binding_mass": float(matrix.sum().detach()),
            "normalized_row_spread": float(normalized_row_spread(matrix).detach()),
            "slot_capacity_overflow": float(slot_capacity_overflow(matrix).detach()),
            "collision_barrier": float(collision_barrier(matrix).detach()),
        }
        if self.mode in ITERATIVE_MODES:
            assert isinstance(self.binding_generator, OccupancyRefinementBindingGenerator)
            stats["refinement_rounds"] = self.binding_generator.refinement_rounds
            stats["occupancy_channel_enabled"] = self.binding_generator.use_occupancy
        return stats

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
        total = answer + SPREAD_LAMBDA * spread + BARRIER_LAMBDA * barrier
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


def cloned_x14_models(d_model: int = 96) -> dict[str, X14BindingModel]:
    models = {mode: X14BindingModel(mode=mode, d_model=d_model) for mode in X14_MODES}

    reference_executor = deepcopy(models["canonical_functional"].executor.state_dict())
    for model in models.values():
        model.executor.load_state_dict(reference_executor)

    one_shot = models["x13_one_shot_barrier"].binding_generator
    assert isinstance(one_shot, RelationalBindingGenerator)
    reference_base = deepcopy(one_shot.state_dict())
    for mode in ITERATIVE_MODES:
        generator = models[mode].binding_generator
        assert isinstance(generator, OccupancyRefinementBindingGenerator)
        generator.base.load_state_dict(reference_base)

    no_occupancy = models["iterative_no_occupancy"].binding_generator
    occupancy = models["iterative_occupancy"].binding_generator
    assert isinstance(no_occupancy, OccupancyRefinementBindingGenerator)
    assert isinstance(occupancy, OccupancyRefinementBindingGenerator)
    occupancy.load_state_dict(deepcopy(no_occupancy.state_dict()))

    return models
