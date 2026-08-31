from __future__ import annotations

import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cardinality_valid_executor import LocalEquivariantTransitionModel
from .explicit_compute import VALUE_MODULUS, ProgramBatch
from .variable_cardinality_binding import (
    DESCRIPTOR_DIM,
    FIXED_ANSWER_REGISTER,
    NUM_CANDIDATE_SLOTS,
    variable_descriptor,
)
from .variable_contextual_data import MAX_CARDINALITY, MIN_CARDINALITY

SLOT_DESCRIPTOR_DIM = 8
X10_MODES = (
    "canonical_functional",
    "x9_direct_independent",
    "relational_independent",
    "relational_coordinated",
)
LEARNED_MODES = X10_MODES[1:]


def slot_descriptor(
    slot_indices: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Deterministic candidate-slot descriptor derived only from slot index."""
    s = slot_indices.to(dtype=dtype)
    denom = float(NUM_CANDIDATE_SLOTS)
    normalized_position = s / float(NUM_CANDIDATE_SLOTS - 1)
    phase1 = math.pi * s / denom
    phase2 = 2.0 * math.pi * s / denom
    indices_long = slot_indices.to(dtype=torch.long)
    bits = []
    for bit in range(3):
        one = ((indices_long >> bit) & 1).to(dtype=dtype)
        bits.append(one * 2.0 - 1.0)
    return torch.stack(
        [
            normalized_position,
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


class DirectIndependentBindingGenerator(nn.Module):
    """X9-style independent descriptor -> fixed-column binding logits."""

    def __init__(self, d_model: int = 96):
        super().__init__()
        self.d_model = int(d_model)
        self.network = nn.Sequential(
            nn.Linear(DESCRIPTOR_DIM, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, NUM_CANDIDATE_SLOTS),
        )
        last = self.network[-1]
        assert isinstance(last, nn.Linear)
        nn.init.normal_(last.weight, mean=0.0, std=0.01)
        nn.init.zeros_(last.bias)

    def logits_from_descriptors(self, descriptors: torch.Tensor) -> torch.Tensor:
        return self.network(descriptors)

    def forward(self, num_registers: int) -> torch.Tensor:
        if not MIN_CARDINALITY <= num_registers <= MAX_CARDINALITY:
            raise ValueError(num_registers)
        parameter = next(self.parameters())
        indices = torch.arange(num_registers, device=parameter.device)
        descriptors = variable_descriptor(indices, num_registers, dtype=parameter.dtype)
        return F.softmax(self.logits_from_descriptors(descriptors), dim=-1)


class RelationalBindingGenerator(nn.Module):
    """Shared external-slot scorer with optional cross-variable self-attention.

    The independent and coordinated variants have identical parameters. In independent mode each
    external token is passed through the attention block as a length-one sequence. In coordinated
    mode all active external tokens are passed together as one sequence. No positional embedding is
    used, so coordinated processing is permutation-equivariant to external-row presentation order.
    """

    def __init__(self, d_model: int = 96, *, coordinated: bool):
        super().__init__()
        if d_model % 4:
            raise ValueError("d_model must be divisible by 4")
        self.d_model = int(d_model)
        self.coordinated = bool(coordinated)
        self.external_encoder = nn.Sequential(
            nn.Linear(DESCRIPTOR_DIM, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.slot_encoder = nn.Sequential(
            nn.Linear(SLOT_DESCRIPTOR_DIM, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.pair_scorer = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )
        last = self.pair_scorer[-1]
        assert isinstance(last, nn.Linear)
        nn.init.normal_(last.weight, mean=0.0, std=0.01)
        nn.init.zeros_(last.bias)

    def logits_from_descriptors(
        self,
        external_descriptors: torch.Tensor,
        slot_descriptors: torch.Tensor,
    ) -> torch.Tensor:
        external = self.external_encoder(external_descriptors)
        slots = self.slot_encoder(slot_descriptors)

        if self.coordinated:
            sequence = external.unsqueeze(0)
            attended, _ = self.attention(sequence, sequence, sequence, need_weights=False)
            contextual = self.norm(sequence + attended).squeeze(0)
        else:
            sequence = external.unsqueeze(1)
            attended, _ = self.attention(sequence, sequence, sequence, need_weights=False)
            contextual = self.norm(sequence + attended).squeeze(1)

        n_external = contextual.shape[0]
        n_slots = slots.shape[0]
        left = contextual[:, None, :].expand(n_external, n_slots, self.d_model)
        right = slots[None, :, :].expand(n_external, n_slots, self.d_model)
        return self.pair_scorer(torch.cat([left, right], dim=-1)).squeeze(-1)

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


class X10BindingModel(nn.Module):
    """Validated local executor plus one X10 binding regime."""

    def __init__(self, *, mode: str, d_model: int = 96):
        super().__init__()
        if mode not in X10_MODES:
            raise ValueError(mode)
        self.mode = mode
        self.d_model = int(d_model)
        self.executor = LocalEquivariantTransitionModel(d_model=d_model)
        if mode == "x9_direct_independent":
            self.binding_generator: nn.Module | None = DirectIndependentBindingGenerator(d_model)
        elif mode in {"relational_independent", "relational_coordinated"}:
            self.binding_generator = RelationalBindingGenerator(
                d_model,
                coordinated=(mode == "relational_coordinated"),
            )
        else:
            self.binding_generator = None

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

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
            "total_binding_mass": float(matrix.sum().detach()),
        }

    def rollout_soft(self, batch: ProgramBatch) -> torch.Tensor:
        binding = self.binding_matrix(batch.initial.shape[1], discrete=False)
        return self.executor.rollout_soft_with_binding(batch, binding)

    def fixed_answer_loss(self, batch: ProgramBatch) -> torch.Tensor:
        decoded = self.rollout_soft(batch)[:, -1, FIXED_ANSWER_REGISTER].clamp_min(1e-12)
        target = batch.target_states[:, -1, FIXED_ANSWER_REGISTER]
        probability = decoded.gather(1, target[:, None]).squeeze(1)
        return -probability.log().mean()

    @torch.no_grad()
    def rollout_hard(self, batch: ProgramBatch, *, discrete_binding: bool) -> torch.Tensor:
        binding = self.binding_matrix(batch.initial.shape[1], discrete=discrete_binding)
        return self.executor.rollout_hard_with_binding(batch, binding)


def cloned_x10_models(d_model: int = 96) -> dict[str, X10BindingModel]:
    """Build X10 regimes with identical executor initialization and paired relational generators."""
    executor_seed = LocalEquivariantTransitionModel(d_model=d_model)
    executor_state = deepcopy(executor_seed.state_dict())

    models = {mode: X10BindingModel(mode=mode, d_model=d_model) for mode in X10_MODES}
    for model in models.values():
        model.executor.load_state_dict(executor_state)

    relational_state = deepcopy(models["relational_independent"].binding_generator.state_dict())
    models["relational_coordinated"].binding_generator.load_state_dict(relational_state)
    return models
