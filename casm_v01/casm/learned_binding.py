from __future__ import annotations

import itertools
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .explicit_compute import NUM_OPERATORS, NUM_REGISTERS, VALUE_MODULUS, ProgramBatch

BINDING_MODES = ("canonical_binding", "learned_binding", "diffuse_binding")
FIXED_ANSWER_REGISTER = 0


class BoundExplicitTransitionModel(nn.Module):
    """Explicit transition model with a probability-preserving register↔slot binding.

    External register identities never index internal slots directly. Initial-state placement,
    source/destination lookup, destination updates, slot-position features and output decoding all
    pass through the same binding matrix.

    The learned binding is parameterized as a convex mixture of all 4! permutation matrices. This
    gives an exactly doubly-stochastic matrix (up to floating-point summation error) for every
    value of the learned 4x4 score tensor, rather than relying on a finite approximate Sinkhorn
    iteration count.
    """

    def __init__(
        self,
        d_model: int = 96,
        *,
        binding_mode: str = "learned_binding",
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

        self.value = nn.Embedding(VALUE_MODULUS, d_model)
        self.slot = nn.Embedding(NUM_REGISTERS, d_model)
        self.command = nn.Embedding(NUM_OPERATORS, d_model)
        self.state_proj = nn.Sequential(
            nn.Linear(NUM_REGISTERS * d_model, d_model),
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

        # The same 16 learned scores used in X6 v0. Near-zero initialization makes the softmax
        # over the 24 permutation scores close to uniform, which yields a near-1/4 binding.
        self.binding_logits = nn.Parameter(torch.empty(NUM_REGISTERS, NUM_REGISTERS))
        nn.init.normal_(self.binding_logits, mean=0.0, std=0.01)

        permutations = list(itertools.permutations(range(NUM_REGISTERS)))
        permutation_indices = torch.tensor(permutations, dtype=torch.long)
        permutation_matrices = torch.zeros(
            len(permutations), NUM_REGISTERS, NUM_REGISTERS, dtype=torch.float32
        )
        rows = torch.arange(NUM_REGISTERS)
        for i, permutation in enumerate(permutations):
            permutation_matrices[i, rows, torch.tensor(permutation)] = 1.0
        self.register_buffer("_permutation_indices", permutation_indices, persistent=False)
        self.register_buffer("_permutation_matrices", permutation_matrices, persistent=False)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def exact_birkhoff_binding(self) -> torch.Tensor:
        rows = torch.arange(NUM_REGISTERS, device=self.binding_logits.device)[None, :]
        selected = self.binding_logits[
            rows, self._permutation_indices.to(device=self.binding_logits.device)
        ]
        scores = selected.sum(dim=1) / self.binding_temperature
        weights = F.softmax(scores, dim=0)
        matrices = self._permutation_matrices.to(
            device=self.binding_logits.device, dtype=self.binding_logits.dtype
        )
        return torch.einsum("p,pij->ij", weights, matrices)

    @staticmethod
    def _identity(device, dtype) -> torch.Tensor:
        return torch.eye(NUM_REGISTERS, device=device, dtype=dtype)

    @staticmethod
    def _diffuse(device, dtype) -> torch.Tensor:
        return torch.full(
            (NUM_REGISTERS, NUM_REGISTERS),
            1.0 / NUM_REGISTERS,
            device=device,
            dtype=dtype,
        )

    def soft_binding(self) -> torch.Tensor:
        if self.binding_mode == "canonical_binding":
            return self._identity(self.binding_logits.device, self.binding_logits.dtype)
        if self.binding_mode == "diffuse_binding":
            return self._diffuse(self.binding_logits.device, self.binding_logits.dtype)
        return self.exact_birkhoff_binding()

    @torch.no_grad()
    def best_permutation(self) -> tuple[torch.Tensor, list[int], float]:
        matrix = self.soft_binding().detach()
        rows = torch.arange(NUM_REGISTERS, device=matrix.device)
        best_perm = None
        best_score = float("-inf")
        for permutation in itertools.permutations(range(NUM_REGISTERS)):
            cols = torch.tensor(permutation, device=matrix.device)
            score = float(matrix[rows, cols].sum().item())
            if score > best_score:
                best_score = score
                best_perm = list(permutation)
        assert best_perm is not None
        projected = torch.zeros_like(matrix)
        projected[rows, torch.tensor(best_perm, device=matrix.device)] = 1.0
        return projected, best_perm, best_score

    def binding_matrix(self, *, discrete: bool = False) -> torch.Tensor:
        if not discrete:
            return self.soft_binding()
        if self.binding_mode == "canonical_binding":
            return self._identity(self.binding_logits.device, self.binding_logits.dtype)
        if self.binding_mode == "diffuse_binding":
            # The diffuse control deliberately has no one-to-one correspondence to project.
            return self._diffuse(self.binding_logits.device, self.binding_logits.dtype)
        projected, _, _ = self.best_permutation()
        return projected

    def binding_stats(self) -> dict[str, object]:
        matrix = self.soft_binding()
        safe = matrix.clamp_min(1e-9)
        _, permutation, score = self.best_permutation()
        row_error = (matrix.sum(dim=1) - 1.0).abs().max()
        column_error = (matrix.sum(dim=0) - 1.0).abs().max()
        return {
            "matrix": matrix.detach().cpu().tolist(),
            "row_max_mean": float(matrix.max(dim=1).values.mean().detach()),
            "column_max_mean": float(matrix.max(dim=0).values.mean().detach()),
            "row_entropy_mean": float((-(safe * safe.log()).sum(dim=1).mean()).detach()),
            "best_permutation_score": float(score),
            "projected_permutation": permutation,
            "max_row_sum_error": float(row_error.detach()),
            "max_column_sum_error": float(column_error.detach()),
        }

    def initial_internal_probs(self, initial: torch.Tensor, binding: torch.Tensor) -> torch.Tensor:
        external = F.one_hot(initial, VALUE_MODULUS).to(dtype=self.value.weight.dtype)
        return torch.einsum("es,bev->bsv", binding, external)

    def decode_external_probs(self, probs: torch.Tensor, binding: torch.Tensor) -> torch.Tensor:
        return torch.einsum("es,bsv->bev", binding, probs)

    def expected_values(self, probs: torch.Tensor) -> torch.Tensor:
        return probs @ self.value.weight

    def encode_state(self, probs: torch.Tensor) -> torch.Tensor:
        batch = probs.shape[0]
        values = self.expected_values(probs)
        slots = torch.arange(NUM_REGISTERS, device=probs.device)[None, :].expand(batch, -1)
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
        probs: torch.Tensor,
        external_index: torch.Tensor,
        binding: torch.Tensor,
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
        features = torch.cat(
            [state_repr, command_repr, value_a, value_b, value_dst, destination_repr], dim=-1
        )
        return self.transition(features)

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
            probs = self.update_internal_state(
                probs,
                batch.dst[:, t],
                F.softmax(logits, dim=-1),
                binding,
            )
            decoded_states.append(self.decode_external_probs(probs, binding))
        return torch.stack(decoded_states, dim=1)

    def fixed_answer_loss(self, batch: ProgramBatch) -> torch.Tensor:
        decoded = self.rollout_soft(batch)[:, -1, FIXED_ANSWER_REGISTER].clamp_min(1e-9)
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
            hard_value = F.one_hot(value, VALUE_MODULUS).to(dtype=probs.dtype)
            probs = self.update_internal_state(
                probs,
                batch.dst[:, t],
                hard_value,
                binding,
            )
            external = self.decode_external_probs(probs, binding)
            states.append(external.argmax(dim=-1))
        return torch.stack(states, dim=1)


def cloned_binding_models(
    seed_model: BoundExplicitTransitionModel,
) -> dict[str, BoundExplicitTransitionModel]:
    state = deepcopy(seed_model.state_dict())
    models = {}
    for mode in BINDING_MODES:
        model = BoundExplicitTransitionModel(
            d_model=seed_model.d_model,
            binding_mode=mode,
            binding_temperature=seed_model.binding_temperature,
        )
        model.load_state_dict(state)
        models[mode] = model
    return models
