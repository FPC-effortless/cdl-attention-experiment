from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cardinality_valid_executor import LocalEquivariantTransitionModel
from .explicit_compute import VALUE_MODULUS, ProgramBatch
from .variable_cardinality_binding import FIXED_ANSWER_REGISTER, NUM_INTERNAL_VALUES
from .variable_contextual_data import MAX_CARDINALITY, MIN_CARDINALITY

ROLE_DIM = 32
ALPHA = 0.1
ADDRESS_BETA = 16.0
DIAGNOSTIC_ROLE_COUNT = 32
RAW_MATRIX_INIT_STD = 0.5

X19D_MODES = (
    "canonical_keyed",
    "frozen_random_orthogonal",
    "unconstrained_recursive",
    "orthogonal_recursive",
)
RECURRENT_X19D_MODES = X19D_MODES[1:]
LEARNED_X19D_MODES = X19D_MODES[2:]
ORTHOGONAL_X19D_MODES = ("frozen_random_orthogonal", "orthogonal_recursive")


def _finite(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"non-finite {name}")
    return tensor


def cayley_orthogonal(raw_matrix: torch.Tensor) -> torch.Tensor:
    """Cayley transform of the skew part of one shared raw recurrence matrix."""
    if raw_matrix.shape != (ROLE_DIM, ROLE_DIM):
        raise ValueError(raw_matrix.shape)
    skew = raw_matrix - raw_matrix.T
    eye = torch.eye(ROLE_DIM, device=raw_matrix.device, dtype=raw_matrix.dtype)
    q = torch.linalg.solve(eye - ALPHA * skew, eye + ALPHA * skew)
    return _finite("Cayley matrix", q)


def unconstrained_matrix(raw_matrix: torch.Tensor) -> torch.Tensor:
    if raw_matrix.shape != (ROLE_DIM, ROLE_DIM):
        raise ValueError(raw_matrix.shape)
    eye = torch.eye(ROLE_DIM, device=raw_matrix.device, dtype=raw_matrix.dtype)
    return _finite("unconstrained recurrence matrix", eye + ALPHA * raw_matrix)


def address_probabilities(query_roles: torch.Tensor, key_roles: torch.Tensor) -> torch.Tensor:
    """Role-only cosine addressing. Record order enters only through key presentation order."""
    if query_roles.ndim != 2 or key_roles.ndim != 2:
        raise ValueError("roles must be rank-2")
    if query_roles.shape[1] != ROLE_DIM or key_roles.shape[1] != ROLE_DIM:
        raise ValueError("role dimension mismatch")
    queries = F.normalize(query_roles, dim=-1, eps=1e-8)
    keys = F.normalize(key_roles, dim=-1, eps=1e-8)
    logits = ADDRESS_BETA * (queries @ keys.T)
    _finite("address logits", logits)
    probs = F.softmax(logits, dim=-1)
    _finite("address probabilities", probs)
    return probs


def hard_address_probabilities(query_roles: torch.Tensor, key_roles: torch.Tensor) -> torch.Tensor:
    soft = address_probabilities(query_roles, key_roles)
    index = soft.argmax(dim=-1)
    return F.one_hot(index, key_roles.shape[0]).to(dtype=soft.dtype)


def read_memory(memory: torch.Tensor, external_index: torch.Tensor, address: torch.Tensor) -> torch.Tensor:
    """Read only through an address row; external_index never indexes memory directly."""
    if memory.ndim != 3 or address.ndim != 2:
        raise ValueError("invalid memory/address rank")
    weights = address[external_index]
    return torch.einsum("bn,bnv->bv", weights, memory)


def write_memory(
    memory: torch.Tensor,
    external_index: torch.Tensor,
    new_value: torch.Tensor,
    address: torch.Tensor,
) -> torch.Tensor:
    """Write only through role-address weights; no direct record selection bypass."""
    weights = address[external_index][:, :, None]
    return memory * (1.0 - weights) + new_value[:, None, :] * weights


def decode_memory(memory: torch.Tensor, address: torch.Tensor) -> torch.Tensor:
    """Decode every external role through the same role-key address matrix."""
    return torch.einsum("en,bnv->bev", address, memory)


class X19DRoleKeyedModel(nn.Module):
    """Recursive role generator plus transient role-keyed executable memory."""

    def __init__(
        self,
        *,
        mode: str,
        d_model: int = 96,
        constructor_seed: torch.Tensor | None = None,
        raw_matrix: torch.Tensor | None = None,
    ):
        super().__init__()
        if mode not in X19D_MODES:
            raise ValueError(mode)
        self.mode = mode
        self.executor = LocalEquivariantTransitionModel(d_model=d_model)

        if mode == "canonical_keyed":
            self.constructor_seed = None
            self.raw_matrix = None
            return

        if constructor_seed is None:
            constructor_seed = torch.randn(ROLE_DIM)
        if raw_matrix is None:
            raw_matrix = torch.randn(ROLE_DIM, ROLE_DIM) * RAW_MATRIX_INIT_STD
        if constructor_seed.shape != (ROLE_DIM,) or raw_matrix.shape != (ROLE_DIM, ROLE_DIM):
            raise ValueError("constructor tensor shape mismatch")

        trainable = mode != "frozen_random_orthogonal"
        self.constructor_seed = nn.Parameter(constructor_seed.detach().clone(), requires_grad=trainable)
        self.raw_matrix = nn.Parameter(raw_matrix.detach().clone(), requires_grad=trainable)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def normalized_seed(self) -> torch.Tensor:
        if self.constructor_seed is None:
            raise RuntimeError("canonical mode has no recurrent seed")
        return F.normalize(self.constructor_seed, dim=0, eps=1e-8)

    def transition_matrix(self) -> torch.Tensor:
        if self.raw_matrix is None:
            raise RuntimeError("canonical mode has no recurrent transition")
        if self.mode == "unconstrained_recursive":
            return unconstrained_matrix(self.raw_matrix)
        if self.mode in ORTHOGONAL_X19D_MODES:
            return cayley_orthogonal(self.raw_matrix)
        raise RuntimeError(self.mode)

    def _roles_from_seed(self, count: int, seed: torch.Tensor) -> torch.Tensor:
        if not 1 <= count <= DIAGNOSTIC_ROLE_COUNT:
            raise ValueError(count)
        transition = self.transition_matrix()
        roles = [F.normalize(seed, dim=0, eps=1e-8)]
        for _ in range(1, count):
            nxt = transition @ roles[-1]
            if self.mode == "unconstrained_recursive":
                nxt = F.normalize(nxt, dim=0, eps=1e-8)
            roles.append(_finite("role", nxt))
        return torch.stack(roles, dim=0)

    def roles(self, count: int) -> torch.Tensor:
        if self.mode == "canonical_keyed":
            if not 1 <= count <= DIAGNOSTIC_ROLE_COUNT:
                raise ValueError(count)
            eye = torch.eye(ROLE_DIM, device=self.executor.value.weight.device, dtype=self.executor.value.weight.dtype)
            return eye[:count]
        return self._roles_from_seed(count, self.normalized_seed())

    def address_matrix(self, num_registers: int, *, discrete: bool) -> torch.Tensor:
        if not MIN_CARDINALITY <= num_registers <= MAX_CARDINALITY:
            raise ValueError(num_registers)
        roles = self.roles(num_registers)
        if discrete:
            return hard_address_probabilities(roles, roles)
        return address_probabilities(roles, roles)

    def address_stats(self, num_registers: int) -> dict[str, object]:
        roles = self.roles(num_registers)
        normalized = F.normalize(roles, dim=-1, eps=1e-8)
        cosine = normalized @ normalized.T
        probs = address_probabilities(roles, roles)
        hard = probs.argmax(dim=-1)
        expected = torch.arange(num_registers, device=hard.device)
        maxv = probs.max(dim=-1).values
        tie_counts = (torch.isclose(probs, maxv[:, None], atol=1e-8, rtol=0.0)).sum(dim=-1)
        self_prob = probs.diagonal()
        masked = probs.masked_fill(torch.eye(num_registers, dtype=torch.bool, device=probs.device), -1.0)
        competitor = masked.max(dim=-1).values
        offdiag = cosine.masked_fill(torch.eye(num_registers, dtype=torch.bool, device=cosine.device), -1.0)
        return {
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

    @staticmethod
    def initial_memory(initial: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        world = F.one_hot(initial, VALUE_MODULUS).to(dtype=dtype)
        zeros = torch.zeros(*world.shape[:-1], NUM_INTERNAL_VALUES - VALUE_MODULUS, device=initial.device, dtype=dtype)
        memory = torch.cat([world, zeros], dim=-1)
        return _finite("initial memory", memory)

    def step_logits(
        self,
        memory: torch.Tensor,
        address: torch.Tensor,
        command: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        dst: torch.Tensor,
    ) -> torch.Tensor:
        value_a = read_memory(memory, a, address) @ self.executor.value.weight
        value_b = read_memory(memory, b, address) @ self.executor.value.weight
        value_dst = read_memory(memory, dst, address) @ self.executor.value.weight
        logits = self.executor.transition(
            torch.cat([self.executor.command(command), value_a, value_b, value_dst], dim=-1)
        )
        return _finite("executor logits", logits)

    def rollout_soft(self, batch: ProgramBatch) -> torch.Tensor:
        n = batch.initial.shape[1]
        address = self.address_matrix(n, discrete=False)
        memory = self.initial_memory(batch.initial, dtype=address.dtype)
        decoded = []
        for t in range(batch.depth):
            logits = self.step_logits(
                memory,
                address,
                batch.commands[:, t],
                batch.arg_a[:, t],
                batch.arg_b[:, t],
                batch.dst[:, t],
            )
            world = F.softmax(logits, dim=-1)
            new_value = torch.cat([world, torch.zeros(world.shape[0], 1, device=world.device, dtype=world.dtype)], dim=-1)
            memory = write_memory(memory, batch.dst[:, t], new_value, address)
            _finite("soft memory", memory)
            decoded.append(decode_memory(memory, address))
        return torch.stack(decoded, dim=1)

    def fixed_answer_loss(self, batch: ProgramBatch) -> torch.Tensor:
        decoded = self.rollout_soft(batch)[:, -1, FIXED_ANSWER_REGISTER, :VALUE_MODULUS].clamp_min(1e-12)
        target = batch.target_states[:, -1, FIXED_ANSWER_REGISTER]
        probability = decoded.gather(1, target[:, None]).squeeze(1)
        loss = -probability.log().mean()
        return _finite("answer loss", loss)

    def loss_components(self, batch: ProgramBatch) -> dict[str, torch.Tensor]:
        answer = self.fixed_answer_loss(batch)
        return {"answer_loss": answer, "total_loss": answer}

    @torch.no_grad()
    def rollout_hard(self, batch: ProgramBatch, *, discrete_binding: bool) -> torch.Tensor:
        n = batch.initial.shape[1]
        address = self.address_matrix(n, discrete=discrete_binding)
        memory = self.initial_memory(batch.initial, dtype=address.dtype)
        states = []
        for t in range(batch.depth):
            logits = self.step_logits(
                memory,
                address,
                batch.commands[:, t],
                batch.arg_a[:, t],
                batch.arg_b[:, t],
                batch.dst[:, t],
            )
            value = logits.argmax(dim=-1)
            world = F.one_hot(value, VALUE_MODULUS).to(dtype=memory.dtype)
            new_value = torch.cat([world, torch.zeros(world.shape[0], 1, device=world.device, dtype=world.dtype)], dim=-1)
            memory = write_memory(memory, batch.dst[:, t], new_value, address)
            decoded = decode_memory(memory, address)[:, :, :VALUE_MODULUS]
            states.append(decoded.argmax(dim=-1))
        return torch.stack(states, dim=1)

    @torch.no_grad()
    def constructor_diagnostics(self, *, perturb_seed: int, count: int = DIAGNOSTIC_ROLE_COUNT) -> dict[str, object]:
        if self.mode == "canonical_keyed":
            raise RuntimeError("canonical mode has no recurrent diagnostics")
        roles = self.roles(count).detach()
        normalized = F.normalize(roles, dim=-1, eps=1e-8)
        cosine = normalized @ normalized.T
        prefixes = {}
        for prefix in (4, 6, 8, 16, 32):
            c = cosine[:prefix, :prefix]
            mask = ~torch.eye(prefix, dtype=torch.bool, device=c.device)
            off = c[mask]
            prefixes[str(prefix)] = {
                "minimum_pairwise_cosine_distance": float((1.0 - off).min()),
                "maximum_offdiagonal_cosine_similarity": float(off.max()),
            }

        first_repeat = None
        for i in range(1, count):
            if float(cosine[i, :i].max()) > 0.99:
                first_repeat = i
                break

        probs = address_probabilities(roles, roles)
        diag = probs.diagonal()
        competitor = probs.masked_fill(torch.eye(count, dtype=torch.bool, device=probs.device), -1.0).max(dim=-1).values
        hard = probs.argmax(dim=-1)
        maxv = probs.max(dim=-1).values
        ties = torch.isclose(probs, maxv[:, None], atol=1e-8, rtol=0.0).sum(dim=-1)

        base_seed = self.normalized_seed().detach()
        perturbations = []
        for k in range(8):
            generator = torch.Generator(device=base_seed.device)
            generator.manual_seed(int(perturb_seed + k * 1009))
            noise = torch.randn(ROLE_DIM, generator=generator, device=base_seed.device, dtype=base_seed.dtype)
            noise = 1e-3 * F.normalize(noise, dim=0, eps=1e-8)
            perturbed_seed = F.normalize(base_seed + noise, dim=0, eps=1e-8)
            perturbed = self._roles_from_seed(count, perturbed_seed).detach()
            denom = (perturbed[0] - roles[0]).norm().clamp_min(1e-12)
            gains = ((perturbed - roles).norm(dim=-1) / denom).cpu().tolist()
            perturbations.append({
                "index": k,
                "gain": gains,
                "maximum_gain": max(gains),
                "final_gain": gains[-1],
            })

        return {
            "count": count,
            "role_norms": roles.norm(dim=-1).cpu().tolist(),
            "pairwise_cosine_similarity": cosine.cpu().tolist(),
            "prefix_diagnostics": prefixes,
            "consecutive_cosine_similarity": [float(cosine[i, i + 1]) for i in range(count - 1)],
            "first_index_similarity_to_earlier_above_0_99": first_repeat,
            "soft_self_address_probability": diag.cpu().tolist(),
            "nearest_competing_address_probability": competitor.cpu().tolist(),
            "hard_address": hard.cpu().tolist(),
            "hard_tied_query_count": int((ties > 1).sum()),
            "perturbation_stability": perturbations,
        }


def cloned_x19d_models(d_model: int = 96) -> dict[str, X19DRoleKeyedModel]:
    """Build exact paired constructor treatments from one common stochastic initialization."""
    constructor_seed = torch.randn(ROLE_DIM)
    raw_matrix = torch.randn(ROLE_DIM, ROLE_DIM) * RAW_MATRIX_INIT_STD

    models = {
        mode: X19DRoleKeyedModel(
            mode=mode,
            d_model=d_model,
            constructor_seed=constructor_seed if mode != "canonical_keyed" else None,
            raw_matrix=raw_matrix if mode != "canonical_keyed" else None,
        )
        for mode in X19D_MODES
    }

    executor_state = deepcopy(models["canonical_keyed"].executor.state_dict())
    for model in models.values():
        model.executor.load_state_dict(executor_state)

    return models
