from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .explicit_compute import NUM_OPERATORS, VALUE_MODULUS, ProgramBatch
from .variable_cardinality_binding import (
    EMPTY_VALUE,
    FIXED_ANSWER_REGISTER,
    NUM_CANDIDATE_SLOTS,
    NUM_INTERNAL_VALUES,
    VariableCardinalityTransitionModel,
)
from .variable_contextual_data import MAX_CARDINALITY, MIN_CARDINALITY


class LocalEquivariantTransitionModel(nn.Module):
    """Canonical-binding executor with no dependence on absolute internal slot identity.

    The learned transition consumes only command identity and values gathered through the supplied
    binding at a, b and dst. Internal slots remain the explicit storage substrate, but no slot
    embedding, flattened workspace, cardinality feature or external-variable ID enters the learned
    transition.
    """

    def __init__(self, d_model: int = 96):
        super().__init__()
        self.d_model = int(d_model)
        self.value = nn.Embedding(NUM_INTERNAL_VALUES, d_model)
        self.command = nn.Embedding(NUM_OPERATORS, d_model)
        self.transition = nn.Sequential(
            nn.Linear(4 * d_model, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, VALUE_MODULUS),
        )

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def canonical_binding(num_registers: int, *, device, dtype) -> torch.Tensor:
        if not MIN_CARDINALITY <= num_registers <= MAX_CARDINALITY:
            raise ValueError(num_registers)
        matrix = torch.zeros(num_registers, NUM_CANDIDATE_SLOTS, device=device, dtype=dtype)
        rows = torch.arange(num_registers, device=device)
        matrix[rows, rows] = 1.0
        return matrix

    def binding_matrix(self, num_registers: int) -> torch.Tensor:
        return self.canonical_binding(
            num_registers, device=self.value.weight.device, dtype=self.value.weight.dtype
        )

    @staticmethod
    def initial_internal_probs(initial: torch.Tensor, binding: torch.Tensor) -> torch.Tensor:
        batch, num_registers = initial.shape
        if binding.shape != (num_registers, NUM_CANDIDATE_SLOTS):
            raise ValueError((binding.shape, initial.shape))
        dtype = binding.dtype
        external_world = F.one_hot(initial, VALUE_MODULUS).to(dtype=dtype)
        external = torch.zeros(
            batch,
            num_registers,
            NUM_INTERNAL_VALUES,
            device=initial.device,
            dtype=dtype,
        )
        external[:, :, :VALUE_MODULUS] = external_world
        occupancy = binding.sum(dim=0)
        denom = torch.maximum(torch.ones_like(occupancy), occupancy)
        transport = binding / denom[None, :]
        occupied = torch.einsum("es,bev->bsv", transport, external)
        empty_mass = 1.0 - occupancy / denom
        empty = torch.zeros(NUM_INTERNAL_VALUES, device=initial.device, dtype=dtype)
        empty[EMPTY_VALUE] = 1.0
        return occupied + empty_mass[None, :, None] * empty[None, None, :]

    @staticmethod
    def decode_external_probs(probs: torch.Tensor, binding: torch.Tensor) -> torch.Tensor:
        return torch.einsum("es,bsv->bev", binding, probs)

    @staticmethod
    def gather_external_register(
        probs: torch.Tensor, external_index: torch.Tensor, binding: torch.Tensor
    ) -> torch.Tensor:
        return torch.einsum("bs,bsv->bv", binding[external_index], probs)

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

    def step_logits(
        self,
        probs: torch.Tensor,
        command: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        dst: torch.Tensor,
        binding: torch.Tensor,
    ) -> torch.Tensor:
        value_a = self.gather_external_register(probs, a, binding) @ self.value.weight
        value_b = self.gather_external_register(probs, b, binding) @ self.value.weight
        value_dst = self.gather_external_register(probs, dst, binding) @ self.value.weight
        return self.transition(
            torch.cat([self.command(command), value_a, value_b, value_dst], dim=-1)
        )

    def rollout_soft_with_binding(self, batch: ProgramBatch, binding: torch.Tensor) -> torch.Tensor:
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

    def rollout_soft(self, batch: ProgramBatch) -> torch.Tensor:
        return self.rollout_soft_with_binding(
            batch, self.binding_matrix(batch.initial.shape[1])
        )

    def fixed_answer_loss(self, batch: ProgramBatch) -> torch.Tensor:
        decoded = self.rollout_soft(batch)[:, -1, FIXED_ANSWER_REGISTER].clamp_min(1e-12)
        target = batch.target_states[:, -1, FIXED_ANSWER_REGISTER]
        probability = decoded.gather(1, target[:, None]).squeeze(1)
        return -probability.log().mean()

    @torch.no_grad()
    def rollout_hard_with_binding(self, batch: ProgramBatch, binding: torch.Tensor) -> torch.Tensor:
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
            world = F.one_hot(value, VALUE_MODULUS).to(dtype=probs.dtype)
            probs = self.update_internal_state(
                probs,
                batch.dst[:, t],
                self.world_value_to_internal(world),
                binding,
            )
            decoded = self.decode_external_probs(probs, binding)
            states.append(decoded[:, :, :VALUE_MODULUS].argmax(dim=-1))
        return torch.stack(states, dim=1)

    @torch.no_grad()
    def rollout_hard(self, batch: ProgramBatch) -> torch.Tensor:
        return self.rollout_hard_with_binding(
            batch, self.binding_matrix(batch.initial.shape[1])
        )


def x9_absolute_slot_control(d_model: int = 96) -> VariableCardinalityTransitionModel:
    return VariableCardinalityTransitionModel(
        d_model=d_model,
        binding_mode="canonical_functional",
        binding_temperature=1.0,
    )
