from __future__ import annotations

import torch
import torch.nn.functional as F

from .explicit_compute import VALUE_MODULUS, ProgramBatch
from .noncontractive_role_dynamics import (
    ROLE_DIM,
    X19DRoleKeyedModel,
    decode_memory,
    write_memory,
)

TRAIN_BETA = 16.0
COUNTERFACTUAL_BETA = 64.0


def address_probabilities_at_beta(
    query_roles: torch.Tensor,
    key_roles: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    if query_roles.ndim != 2 or key_roles.ndim != 2:
        raise ValueError("roles must be rank-2")
    if query_roles.shape[1] != ROLE_DIM or key_roles.shape[1] != ROLE_DIM:
        raise ValueError("role dimension mismatch")
    if beta <= 0.0:
        raise ValueError(beta)
    queries = F.normalize(query_roles, dim=-1, eps=1e-8)
    keys = F.normalize(key_roles, dim=-1, eps=1e-8)
    logits = float(beta) * (queries @ keys.T)
    if not torch.isfinite(logits).all():
        raise RuntimeError("non-finite address logits")
    probs = F.softmax(logits, dim=-1)
    if not torch.isfinite(probs).all():
        raise RuntimeError("non-finite address probabilities")
    return probs


class X19VAddressView:
    """Evaluation-only address-temperature view over one already-trained X19D model."""

    def __init__(self, model: X19DRoleKeyedModel, *, beta: float):
        self.model = model
        self.beta = float(beta)
        if self.beta not in (TRAIN_BETA, COUNTERFACTUAL_BETA):
            raise ValueError(self.beta)

    def roles(self, count: int) -> torch.Tensor:
        return self.model.roles(count)

    def address_matrix(self, num_registers: int, *, discrete: bool) -> torch.Tensor:
        roles = self.roles(num_registers)
        probs = address_probabilities_at_beta(roles, roles, beta=self.beta)
        if not discrete:
            return probs
        index = probs.argmax(dim=-1)
        return F.one_hot(index, num_registers).to(dtype=probs.dtype)

    def address_stats(self, num_registers: int) -> dict[str, object]:
        roles = self.roles(num_registers)
        normalized = F.normalize(roles, dim=-1, eps=1e-8)
        cosine = normalized @ normalized.T
        probs = address_probabilities_at_beta(roles, roles, beta=self.beta)
        hard = probs.argmax(dim=-1)
        expected = torch.arange(num_registers, device=hard.device)
        maxv = probs.max(dim=-1).values
        tie_counts = torch.isclose(probs, maxv[:, None], atol=1e-8, rtol=0.0).sum(dim=-1)
        self_prob = probs.diagonal()
        masked = probs.masked_fill(
            torch.eye(num_registers, dtype=torch.bool, device=probs.device), -1.0
        )
        competitor = masked.max(dim=-1).values
        offdiag = cosine.masked_fill(
            torch.eye(num_registers, dtype=torch.bool, device=cosine.device), -1.0
        )
        return {
            "beta": self.beta,
            "num_registers": num_registers,
            "hard_address": hard.detach().cpu().tolist(),
            "hard_self_address_count": int((hard == expected).sum()),
            "hard_all_self": bool(torch.equal(hard, expected)),
            "hard_tied_query_count": int((tie_counts > 1).sum()),
            "mean_soft_self_address_probability": float(self_prob.mean().detach()),
            "minimum_soft_self_address_probability": float(self_prob.min().detach()),
            "maximum_competing_address_probability": float(competitor.max().detach()),
            "maximum_offdiagonal_cosine_similarity": float(offdiag.max().detach()),
        }

    @torch.no_grad()
    def rollout_hard(self, batch: ProgramBatch, *, discrete_binding: bool) -> torch.Tensor:
        n = batch.initial.shape[1]
        address = self.address_matrix(n, discrete=discrete_binding)
        memory = self.model.initial_memory(batch.initial, dtype=address.dtype)
        states = []
        for t in range(batch.depth):
            logits = self.model.step_logits(
                memory,
                address,
                batch.commands[:, t],
                batch.arg_a[:, t],
                batch.arg_b[:, t],
                batch.dst[:, t],
            )
            value = logits.argmax(dim=-1)
            world = F.one_hot(value, VALUE_MODULUS).to(dtype=memory.dtype)
            new_value = torch.cat(
                [world, torch.zeros(world.shape[0], 1, device=world.device, dtype=world.dtype)],
                dim=-1,
            )
            memory = write_memory(memory, batch.dst[:, t], new_value, address)
            decoded = decode_memory(memory, address)[:, :, :VALUE_MODULUS]
            states.append(decoded.argmax(dim=-1))
        return torch.stack(states, dim=1)
