from __future__ import annotations

import itertools
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .explicit_compute import NUM_OPERATORS, NUM_REGISTERS, VALUE_MODULUS, ProgramBatch

NUM_EXTERNAL_REGISTERS = NUM_REGISTERS
NUM_CANDIDATE_SLOTS = 8
EMPTY_VALUE = VALUE_MODULUS
NUM_INTERNAL_VALUES = VALUE_MODULUS + 1
FIXED_ANSWER_REGISTER = 0
BINDING_MODES = ("canonical_sparse", "learned_sparse", "diffuse_surplus")


class SurplusSlotTransitionModel(nn.Module):
    """Explicit transition model with 8 candidate slots for 4 external registers.

    The learned 4x8 binding is a convex mixture over all 8P4 injective assignments. Rows therefore
    sum to one, column occupancies are at most one, and surplus mass is represented by an explicit
    EMPTY categorical symbol so every internal slot remains normalized.
    """

    def __init__(
        self,
        d_model: int = 96,
        *,
        binding_mode: str = "learned_sparse",
        binding_temperature: float = 1.0,
    ):
        super().__init__()
        if binding_mode not in BINDING_MODES:
            raise ValueError(binding_mode)
        if binding_temperature <= 0:
            raise ValueError("binding_temperature must be positive")

        self.d_model = int(d_model)
        self.binding_mode = binding_mode
        self.binding_temperature = float(binding_temperature)

        self.value = nn.Embedding(NUM_INTERNAL_VALUES, d_model)
        self.slot = nn.Embedding(NUM_CANDIDATE_SLOTS, d_model)
        self.command = nn.Embedding(NUM_OPERATORS, d_model)
        self.state_proj = nn.Sequential(
            nn.Linear(NUM_CANDIDATE_SLOTS * d_model, d_model),
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

        self.binding_logits = nn.Parameter(
            torch.empty(NUM_EXTERNAL_REGISTERS, NUM_CANDIDATE_SLOTS)
        )
        nn.init.normal_(self.binding_logits, mean=0.0, std=0.01)

        assignments = list(
            itertools.permutations(range(NUM_CANDIDATE_SLOTS), NUM_EXTERNAL_REGISTERS)
        )
        assignment_indices = torch.tensor(assignments, dtype=torch.long)
        assignment_matrices = torch.zeros(
            len(assignments),
            NUM_EXTERNAL_REGISTERS,
            NUM_CANDIDATE_SLOTS,
            dtype=torch.float32,
        )
        rows = torch.arange(NUM_EXTERNAL_REGISTERS)
        for i, assignment in enumerate(assignments):
            assignment_matrices[i, rows, torch.tensor(assignment)] = 1.0
        self.register_buffer("_assignment_indices", assignment_indices, persistent=False)
        self.register_buffer("_assignment_matrices", assignment_matrices, persistent=False)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def _canonical(device, dtype) -> torch.Tensor:
        matrix = torch.zeros(
            NUM_EXTERNAL_REGISTERS, NUM_CANDIDATE_SLOTS, device=device, dtype=dtype
        )
        rows = torch.arange(NUM_EXTERNAL_REGISTERS, device=device)
        matrix[rows, rows] = 1.0
        return matrix

    @staticmethod
    def _diffuse(device, dtype) -> torch.Tensor:
        return torch.full(
            (NUM_EXTERNAL_REGISTERS, NUM_CANDIDATE_SLOTS),
            1.0 / NUM_CANDIDATE_SLOTS,
            device=device,
            dtype=dtype,
        )

    def exact_injective_binding(self) -> torch.Tensor:
        rows = torch.arange(NUM_EXTERNAL_REGISTERS, device=self.binding_logits.device)[None, :]
        indices = self._assignment_indices.to(device=self.binding_logits.device)
        selected = self.binding_logits[rows, indices]
        scores = selected.sum(dim=1) / self.binding_temperature
        weights = F.softmax(scores, dim=0)
        matrices = self._assignment_matrices.to(
            device=self.binding_logits.device, dtype=self.binding_logits.dtype
        )
        return torch.einsum("p,pej->ej", weights, matrices)

    def soft_binding(self) -> torch.Tensor:
        if self.binding_mode == "canonical_sparse":
            return self._canonical(self.binding_logits.device, self.binding_logits.dtype)
        if self.binding_mode == "diffuse_surplus":
            return self._diffuse(self.binding_logits.device, self.binding_logits.dtype)
        return self.exact_injective_binding()

    @torch.no_grad()
    def best_injective_assignment(self) -> tuple[torch.Tensor, list[int], float]:
        matrix = self.soft_binding().detach()
        rows = torch.arange(NUM_EXTERNAL_REGISTERS, device=matrix.device)
        best_assignment = None
        best_score = float("-inf")
        for assignment in itertools.permutations(
            range(NUM_CANDIDATE_SLOTS), NUM_EXTERNAL_REGISTERS
        ):
            cols = torch.tensor(assignment, device=matrix.device)
            score = float(matrix[rows, cols].sum().item())
            if score > best_score:
                best_score = score
                best_assignment = list(assignment)
        assert best_assignment is not None
        projected = torch.zeros_like(matrix)
        projected[rows, torch.tensor(best_assignment, device=matrix.device)] = 1.0
        return projected, best_assignment, best_score

    def binding_matrix(self, *, discrete: bool = False) -> torch.Tensor:
        if not discrete:
            return self.soft_binding()
        if self.binding_mode == "canonical_sparse":
            return self._canonical(self.binding_logits.device, self.binding_logits.dtype)
        if self.binding_mode == "diffuse_surplus":
            return self._diffuse(self.binding_logits.device, self.binding_logits.dtype)
        projected, _, _ = self.best_injective_assignment()
        return projected

    def binding_stats(self) -> dict[str, object]:
        matrix = self.soft_binding()
        safe = matrix.clamp_min(1e-12)
        column_occupancy = matrix.sum(dim=0)
        _, assignment, score = self.best_injective_assignment()
        return {
            "matrix": matrix.detach().cpu().tolist(),
            "row_max_mean": float(matrix.max(dim=1).values.mean().detach()),
            "row_entropy_mean": float((-(safe * safe.log()).sum(dim=1).mean()).detach()),
            "best_assignment_score": float(score),
            "projected_assignment": assignment,
            "selected_slot_count": len(set(assignment)),
            "max_row_sum_error": float((matrix.sum(dim=1) - 1.0).abs().max().detach()),
            "max_column_occupancy": float(column_occupancy.max().detach()),
            "min_column_occupancy": float(column_occupancy.min().detach()),
            "total_assignment_mass": float(matrix.sum().detach()),
        }

    def initial_internal_probs(self, initial: torch.Tensor, binding: torch.Tensor) -> torch.Tensor:
        batch = initial.shape[0]
        external_world = F.one_hot(initial, VALUE_MODULUS).to(dtype=self.value.weight.dtype)
        external = torch.zeros(
            batch,
            NUM_EXTERNAL_REGISTERS,
            NUM_INTERNAL_VALUES,
            device=initial.device,
            dtype=self.value.weight.dtype,
        )
        external[:, :, :VALUE_MODULUS] = external_world
        occupied = torch.einsum("es,bev->bsv", binding, external)
        occupancy = binding.sum(dim=0).clamp(0.0, 1.0)
        empty = torch.zeros(
            NUM_INTERNAL_VALUES, device=initial.device, dtype=self.value.weight.dtype
        )
        empty[EMPTY_VALUE] = 1.0
        return occupied + (1.0 - occupancy)[None, :, None] * empty[None, None, :]

    @staticmethod
    def decode_external_probs(probs: torch.Tensor, binding: torch.Tensor) -> torch.Tensor:
        return torch.einsum("es,bsv->bev", binding, probs)

    def expected_values(self, probs: torch.Tensor) -> torch.Tensor:
        return probs @ self.value.weight

    def encode_state(self, probs: torch.Tensor) -> torch.Tensor:
        batch = probs.shape[0]
        values = self.expected_values(probs)
        slots = torch.arange(NUM_CANDIDATE_SLOTS, device=probs.device)[None, :].expand(batch, -1)
        encoded = values + self.slot(slots)
        return self.state_proj(encoded.flatten(1))

    def bound_slot_repr(self, external_index: torch.Tensor, binding: torch.Tensor) -> torch.Tensor:
        weights = binding[external_index]
        return weights @ self.slot.weight

    def encode_command(
        self,
        command: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        dst: torch.Tensor,
        binding: torch.Tensor,
    ) -> torch.Tensor:
        return self.command_proj(
            torch.cat(
                [
                    self.command(command),
                    self.bound_slot_repr(a, binding),
                    self.bound_slot_repr(b, binding),
                    self.bound_slot_repr(dst, binding),
                ],
                dim=-1,
            )
        )

    @staticmethod
    def gather_external_register(
        probs: torch.Tensor, external_index: torch.Tensor, binding: torch.Tensor
    ) -> torch.Tensor:
        weights = binding[external_index]
        return torch.einsum("bs,bsv->bv", weights, probs)

    def step_logits(
        self,
        probs: torch.Tensor,
        command: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        dst: torch.Tensor,
        binding: torch.Tensor,
    ) -> torch.Tensor:
        state_repr = self.encode_state(probs)
        command_repr = self.encode_command(command, a, b, dst, binding)
        value_a = self.gather_external_register(probs, a, binding) @ self.value.weight
        value_b = self.gather_external_register(probs, b, binding) @ self.value.weight
        value_dst = self.gather_external_register(probs, dst, binding) @ self.value.weight
        destination_repr = self.bound_slot_repr(dst, binding)
        return self.transition(
            torch.cat(
                [state_repr, command_repr, value_a, value_b, value_dst, destination_repr], dim=-1
            )
        )

    @staticmethod
    def world_value_to_internal(value_probs: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros(
            value_probs.shape[0], 1, device=value_probs.device, dtype=value_probs.dtype
        )
        return torch.cat([value_probs, zeros], dim=-1)

    @staticmethod
    def update_internal_state(
        probs: torch.Tensor,
        dst: torch.Tensor,
        new_value_probs: torch.Tensor,
        binding: torch.Tensor,
    ) -> torch.Tensor:
        mask = binding[dst][:, :, None]
        return probs * (1.0 - mask) + new_value_probs[:, None, :] * mask

    def rollout_soft(self, batch: ProgramBatch) -> torch.Tensor:
        binding = self.binding_matrix(discrete=False)
        probs = self.initial_internal_probs(batch.initial, binding)
        decoded_states = []
        for t in range(batch.depth):
            logits = self.step_logits(
                probs,
                batch.commands[:, t],
                batch.arg_a[:, t],
                batch.arg_b[:, t],
                batch.dst[:, t],
                binding,
            )
            new_value = self.world_value_to_internal(F.softmax(logits, dim=-1))
            probs = self.update_internal_state(probs, batch.dst[:, t], new_value, binding)
            decoded_states.append(self.decode_external_probs(probs, binding))
        return torch.stack(decoded_states, dim=1)

    def fixed_answer_loss(self, batch: ProgramBatch) -> torch.Tensor:
        decoded = self.rollout_soft(batch)[:, -1, FIXED_ANSWER_REGISTER].clamp_min(1e-12)
        target = batch.target_states[:, -1, FIXED_ANSWER_REGISTER]
        target_prob = decoded.gather(1, target[:, None]).squeeze(1)
        return -target_prob.log().mean()

    @torch.no_grad()
    def rollout_hard(self, batch: ProgramBatch, *, discrete_binding: bool = True) -> torch.Tensor:
        binding = self.binding_matrix(discrete=discrete_binding)
        probs = self.initial_internal_probs(batch.initial, binding)
        states = []
        for t in range(batch.depth):
            logits = self.step_logits(
                probs,
                batch.commands[:, t],
                batch.arg_a[:, t],
                batch.arg_b[:, t],
                batch.dst[:, t],
                binding,
            )
            value = logits.argmax(dim=-1)
            world_hard = F.one_hot(value, VALUE_MODULUS).to(dtype=probs.dtype)
            new_value = self.world_value_to_internal(world_hard)
            probs = self.update_internal_state(probs, batch.dst[:, t], new_value, binding)
            external = self.decode_external_probs(probs, binding)
            states.append(external[:, :, :VALUE_MODULUS].argmax(dim=-1))
        return torch.stack(states, dim=1)


def cloned_surplus_models(
    seed_model: SurplusSlotTransitionModel,
) -> dict[str, SurplusSlotTransitionModel]:
    state = deepcopy(seed_model.state_dict())
    models: dict[str, SurplusSlotTransitionModel] = {}
    for mode in BINDING_MODES:
        model = SurplusSlotTransitionModel(
            d_model=seed_model.d_model,
            binding_mode=mode,
            binding_temperature=seed_model.binding_temperature,
        )
        model.load_state_dict(state)
        models[mode] = model
    return models
