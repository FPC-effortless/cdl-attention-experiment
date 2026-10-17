from __future__ import annotations

import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .explicit_compute import NUM_OPERATORS, VALUE_MODULUS, ProgramBatch
from .variable_contextual_data import MAX_CARDINALITY, MIN_CARDINALITY

NUM_CANDIDATE_SLOTS = 8
EMPTY_VALUE = VALUE_MODULUS
NUM_INTERNAL_VALUES = VALUE_MODULUS + 1
FIXED_ANSWER_REGISTER = 0
DESCRIPTOR_DIM = 9
BINDING_MODES = ("canonical_functional", "shared_generator_dense", "diffuse_dense")


def variable_descriptor(
    external_indices: torch.Tensor,
    num_registers: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Deterministic, non-learned descriptor derived only from (external index, cardinality)."""
    if not MIN_CARDINALITY <= num_registers <= MAX_CARDINALITY:
        raise ValueError(num_registers)
    e = external_indices.to(dtype=dtype)
    n = torch.tensor(float(num_registers), device=e.device, dtype=dtype)
    normalized_position = e / float(num_registers - 1)
    normalized_cardinality = torch.full_like(e, float(num_registers) / float(MAX_CARDINALITY))
    phase1 = math.pi * e / n
    phase2 = 2.0 * math.pi * e / n
    bits = []
    indices_long = external_indices.to(dtype=torch.long)
    for bit in range(3):
        one = ((indices_long >> bit) & 1).to(dtype=dtype)
        bits.append(one * 2.0 - 1.0)
    return torch.stack(
        [
            normalized_position,
            normalized_cardinality,
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


class VariableCardinalityTransitionModel(nn.Module):
    """Explicit state transition with a shared descriptor-to-binding generator.

    No trainable parameter is indexed directly by external variable identity. The same binding
    generator maps deterministic descriptors for every active external variable and cardinality.
    """

    def __init__(
        self,
        d_model: int = 96,
        *,
        binding_mode: str = "shared_generator_dense",
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
        self.binding_generator = nn.Sequential(
            nn.Linear(DESCRIPTOR_DIM, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, NUM_CANDIDATE_SLOTS),
        )
        last = self.binding_generator[-1]
        assert isinstance(last, nn.Linear)
        nn.init.normal_(last.weight, mean=0.0, std=0.01)
        nn.init.zeros_(last.bias)

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

    @staticmethod
    def diffuse_binding(num_registers: int, *, device, dtype) -> torch.Tensor:
        return torch.full(
            (num_registers, NUM_CANDIDATE_SLOTS),
            1.0 / NUM_CANDIDATE_SLOTS,
            device=device,
            dtype=dtype,
        )

    def generated_binding(self, num_registers: int) -> torch.Tensor:
        indices = torch.arange(num_registers, device=self.value.weight.device)
        descriptors = variable_descriptor(
            indices,
            num_registers,
            dtype=self.value.weight.dtype,
        )
        logits = self.binding_generator(descriptors)
        return F.softmax(logits / self.binding_temperature, dim=-1)

    def soft_binding(self, num_registers: int) -> torch.Tensor:
        if self.binding_mode == "canonical_functional":
            return self.canonical_binding(
                num_registers, device=self.value.weight.device, dtype=self.value.weight.dtype
            )
        if self.binding_mode == "diffuse_dense":
            return self.diffuse_binding(
                num_registers, device=self.value.weight.device, dtype=self.value.weight.dtype
            )
        return self.generated_binding(num_registers)

    @torch.no_grad()
    def independent_argmax_binding(self, num_registers: int) -> tuple[torch.Tensor, list[int]]:
        matrix = self.soft_binding(num_registers).detach()
        assignment = matrix.argmax(dim=1)
        rows = torch.arange(num_registers, device=matrix.device)
        projected = torch.zeros_like(matrix)
        projected[rows, assignment] = 1.0
        return projected, assignment.cpu().tolist()

    def binding_matrix(self, num_registers: int, *, discrete: bool = False) -> torch.Tensor:
        if not discrete:
            return self.soft_binding(num_registers)
        if self.binding_mode == "canonical_functional":
            return self.canonical_binding(
                num_registers, device=self.value.weight.device, dtype=self.value.weight.dtype
            )
        if self.binding_mode == "shared_generator_dense":
            projected, _ = self.independent_argmax_binding(num_registers)
            return projected
        return self.diffuse_binding(
            num_registers, device=self.value.weight.device, dtype=self.value.weight.dtype
        )

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

    def initial_internal_probs(self, initial: torch.Tensor, binding: torch.Tensor) -> torch.Tensor:
        batch, num_registers = initial.shape
        if binding.shape != (num_registers, NUM_CANDIDATE_SLOTS):
            raise ValueError((binding.shape, initial.shape))
        external_world = F.one_hot(initial, VALUE_MODULUS).to(dtype=self.value.weight.dtype)
        external = torch.zeros(
            batch,
            num_registers,
            NUM_INTERNAL_VALUES,
            device=initial.device,
            dtype=self.value.weight.dtype,
        )
        external[:, :, :VALUE_MODULUS] = external_world
        occupancy = binding.sum(dim=0)
        denom = torch.maximum(torch.ones_like(occupancy), occupancy)
        transport = binding / denom[None, :]
        occupied = torch.einsum("es,bev->bsv", transport, external)
        empty_mass = 1.0 - occupancy / denom
        empty = torch.zeros(
            NUM_INTERNAL_VALUES, device=initial.device, dtype=self.value.weight.dtype
        )
        empty[EMPTY_VALUE] = 1.0
        return occupied + empty_mass[None, :, None] * empty[None, None, :]

    @staticmethod
    def decode_external_probs(probs: torch.Tensor, binding: torch.Tensor) -> torch.Tensor:
        return torch.einsum("es,bsv->bev", binding, probs)

    def expected_values(self, probs: torch.Tensor) -> torch.Tensor:
        return probs @ self.value.weight

    def encode_state(self, probs: torch.Tensor) -> torch.Tensor:
        batch = probs.shape[0]
        values = self.expected_values(probs)
        slots = torch.arange(NUM_CANDIDATE_SLOTS, device=probs.device)[None, :].expand(batch, -1)
        return self.state_proj((values + self.slot(slots)).flatten(1))

    def bound_slot_repr(self, external_index: torch.Tensor, binding: torch.Tensor) -> torch.Tensor:
        return binding[external_index] @ self.slot.weight

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
        return torch.einsum("bs,bsv->bv", binding[external_index], probs)

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
        num_registers = batch.initial.shape[1]
        binding = self.binding_matrix(num_registers, discrete=False)
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
        probability = decoded.gather(1, target[:, None]).squeeze(1)
        return -probability.log().mean()

    @torch.no_grad()
    def rollout_hard(self, batch: ProgramBatch, *, discrete_binding: bool = True) -> torch.Tensor:
        num_registers = batch.initial.shape[1]
        binding = self.binding_matrix(num_registers, discrete=discrete_binding)
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
            hard = F.one_hot(value, VALUE_MODULUS).to(dtype=probs.dtype)
            probs = self.update_internal_state(
                probs,
                batch.dst[:, t],
                self.world_value_to_internal(hard),
                binding,
            )
            decoded = self.decode_external_probs(probs, binding)
            states.append(decoded[:, :, :VALUE_MODULUS].argmax(dim=-1))
        return torch.stack(states, dim=1)


def cloned_cardinality_models(
    seed_model: VariableCardinalityTransitionModel,
) -> dict[str, VariableCardinalityTransitionModel]:
    state = deepcopy(seed_model.state_dict())
    out = {}
    for mode in BINDING_MODES:
        model = VariableCardinalityTransitionModel(
            d_model=seed_model.d_model,
            binding_mode=mode,
            binding_temperature=seed_model.binding_temperature,
        )
        model.load_state_dict(state)
        out[mode] = model
    return out
