from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cardinality_valid_executor import LocalEquivariantTransitionModel
from .collision_barrier_binding import collision_barrier
from .coordinated_binding import SLOT_DESCRIPTOR_DIM, slot_descriptor
from .explicit_compute import ProgramBatch
from .global_coordinate_binding import global_variable_descriptor
from .scarcity_binding import normalized_row_spread
from .variable_cardinality_binding import FIXED_ANSWER_REGISTER, NUM_CANDIDATE_SLOTS
from .variable_contextual_data import MAX_CARDINALITY, MIN_CARDINALITY

ROLE_DIM = 32
ROLE_CONTEXT_DIM = 9
ROLE_HIDDEN_DIM = 64
ROLE_DIAGNOSTIC_COUNT = 8

X19_MODES = (
    "canonical_functional",
    "static_global_roles",
    "recursive_roles",
)
LEARNED_X19_MODES = X19_MODES[1:]


class SharedRoleCell(nn.Module):
    """Parameter-shared role cell used by both static and recursive regimes."""

    def __init__(self):
        super().__init__()
        self.context_proj = nn.Linear(ROLE_CONTEXT_DIM, ROLE_DIM, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(ROLE_DIM, ROLE_HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(ROLE_HIDDEN_DIM, ROLE_DIM),
        )
        self.norm = nn.LayerNorm(ROLE_DIM)

    def forward(self, role: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if role.shape[-1] != ROLE_DIM:
            raise ValueError("role dimension mismatch")
        if context.shape[-1] != ROLE_CONTEXT_DIM:
            raise ValueError("role-context dimension mismatch")
        u = role + self.context_proj(context)
        return self.norm(u + self.mlp(u))


class RoleToSlotScorer(nn.Module):
    """Shared storage bridge from role identity to deterministic physical slot descriptors."""

    def __init__(self):
        super().__init__()
        self.slot_encoder = nn.Sequential(
            nn.Linear(SLOT_DESCRIPTOR_DIM, ROLE_DIM),
            nn.SiLU(),
            nn.Linear(ROLE_DIM, ROLE_DIM),
        )
        self.pair_scorer = nn.Sequential(
            nn.Linear(2 * ROLE_DIM, ROLE_HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(ROLE_HIDDEN_DIM, 1),
        )
        last = self.pair_scorer[-1]
        assert isinstance(last, nn.Linear)
        nn.init.normal_(last.weight, mean=0.0, std=0.01)
        nn.init.zeros_(last.bias)

    def logits_from_roles(self, roles: torch.Tensor, slot_descriptors: torch.Tensor) -> torch.Tensor:
        if roles.ndim != 2 or roles.shape[1] != ROLE_DIM:
            raise ValueError("roles must be [n, role_dim]")
        if slot_descriptors.ndim != 2 or slot_descriptors.shape[1] != SLOT_DESCRIPTOR_DIM:
            raise ValueError("slot descriptors must be [slots, slot_descriptor_dim]")
        slots = self.slot_encoder(slot_descriptors)
        left = roles[:, None, :].expand(roles.shape[0], slots.shape[0], ROLE_DIM)
        right = slots[None, :, :].expand(roles.shape[0], slots.shape[0], ROLE_DIM)
        return self.pair_scorer(torch.cat([left, right], dim=-1)).squeeze(-1)

    def probabilities_from_roles(self, roles: torch.Tensor, slot_descriptors: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.logits_from_roles(roles, slot_descriptors), dim=-1)


class X19RoleModel(nn.Module):
    """Recursive/static computational-role construction plus a shared physical storage bridge."""

    def __init__(self, *, mode: str, d_model: int = 96):
        super().__init__()
        if mode not in X19_MODES:
            raise ValueError(mode)
        self.mode = mode
        self.executor = LocalEquivariantTransitionModel(d_model=d_model)
        if mode == "canonical_functional":
            self.role_seed = None
            self.role_cell = None
            self.storage_bridge = None
        else:
            self.role_seed = nn.Parameter(torch.randn(ROLE_DIM) * 0.02)
            self.role_cell = SharedRoleCell()
            self.storage_bridge = RoleToSlotScorer()

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def normalized_seed(self) -> torch.Tensor:
        if self.role_seed is None:
            raise RuntimeError("canonical mode has no learned role seed")
        return F.normalize(self.role_seed, dim=0, eps=1e-8)

    def static_context(self, role_indices: torch.Tensor) -> torch.Tensor:
        if self.mode == "canonical_functional":
            raise RuntimeError("canonical mode has no role context")
        # num_registers is used only to validate each requested index; choosing 8 makes the
        # frozen X18 global coordinate independent of active task cardinality.
        return global_variable_descriptor(role_indices, ROLE_DIAGNOSTIC_COUNT, dtype=self.role_seed.dtype)

    def roles(self, num_roles: int) -> torch.Tensor:
        if self.mode == "canonical_functional":
            raise RuntimeError("canonical mode has no learned role vectors")
        if not 1 <= num_roles <= ROLE_DIAGNOSTIC_COUNT:
            raise ValueError(num_roles)
        assert self.role_cell is not None and self.role_seed is not None
        seed = self.normalized_seed()
        if self.mode == "static_global_roles":
            indices = torch.arange(num_roles, device=seed.device)
            context = self.static_context(indices)
            expanded = seed[None, :].expand(num_roles, ROLE_DIM)
            return self.role_cell(expanded, context)
        if self.mode == "recursive_roles":
            out = [seed]
            step_context = seed.new_zeros(ROLE_CONTEXT_DIM)
            for _ in range(1, num_roles):
                out.append(self.role_cell(out[-1], step_context))
            return torch.stack(out, dim=0)
        raise RuntimeError(self.mode)

    def role_to_slot_logits(self, num_registers: int) -> torch.Tensor:
        if self.mode == "canonical_functional":
            raise RuntimeError("canonical mode has no learned storage bridge")
        if not MIN_CARDINALITY <= num_registers <= MAX_CARDINALITY:
            raise ValueError(num_registers)
        assert self.storage_bridge is not None and self.role_seed is not None
        roles = self.roles(num_registers)
        slot_indices = torch.arange(NUM_CANDIDATE_SLOTS, device=self.role_seed.device)
        slots = slot_descriptor(slot_indices, dtype=self.role_seed.dtype)
        return self.storage_bridge.logits_from_roles(roles, slots)

    def soft_binding(self, num_registers: int) -> torch.Tensor:
        if self.mode == "canonical_functional":
            return self.executor.binding_matrix(num_registers)
        return F.softmax(self.role_to_slot_logits(num_registers), dim=-1)

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
        return self.independent_argmax_binding(num_registers)[0]

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
            "normalized_row_spread": float(normalized_row_spread(matrix).detach()),
            "collision_barrier": float(collision_barrier(matrix).detach()),
        }

    @torch.no_grad()
    def role_diagnostics(self, count: int = ROLE_DIAGNOSTIC_COUNT) -> dict[str, object]:
        if self.mode == "canonical_functional":
            raise RuntimeError("canonical mode has no learned roles")
        roles = self.roles(count).detach()
        norms = roles.norm(dim=1)
        normalized = F.normalize(roles, dim=1, eps=1e-8)
        cosine = normalized @ normalized.T
        distances = 1.0 - cosine
        mask = ~torch.eye(count, dtype=torch.bool, device=roles.device)
        min_pairwise = float(distances[mask].min()) if count > 1 else 0.0
        consecutive = [float(cosine[i, i + 1]) for i in range(count - 1)]
        assert self.storage_bridge is not None and self.role_seed is not None
        slot_indices = torch.arange(NUM_CANDIDATE_SLOTS, device=self.role_seed.device)
        slots = slot_descriptor(slot_indices, dtype=self.role_seed.dtype)
        probs = self.storage_bridge.probabilities_from_roles(roles, slots)
        assignment = probs.argmax(dim=1).cpu().tolist()
        return {
            "count": count,
            "role_norms": norms.cpu().tolist(),
            "pairwise_cosine_similarity": cosine.cpu().tolist(),
            "minimum_pairwise_cosine_distance": min_pairwise,
            "consecutive_cosine_similarity": consecutive,
            "hard_storage_preferences": assignment,
            "unique_hard_storage_preferences": len(set(assignment)),
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


def cloned_x19_models(d_model: int = 96) -> dict[str, X19RoleModel]:
    models = {mode: X19RoleModel(mode=mode, d_model=d_model) for mode in X19_MODES}

    executor_state = deepcopy(models["canonical_functional"].executor.state_dict())
    for model in models.values():
        model.executor.load_state_dict(executor_state)

    static_state = deepcopy(models["static_global_roles"].state_dict())
    recursive = models["recursive_roles"]
    # Copy all shared learned state, then restore the already-common executor state explicitly.
    recursive.load_state_dict(static_state)
    recursive.executor.load_state_dict(executor_state)

    return models
