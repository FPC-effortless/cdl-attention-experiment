from __future__ import annotations

import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cardinality_valid_executor import LocalEquivariantTransitionModel
from .explicit_compute import VALUE_MODULUS
from .state_instantiation_data import (
    NUM_CANDIDATES,
    OUTPUT_CANDIDATE,
    StateInstantiationBatch,
)
from .variable_cardinality_binding import EMPTY_VALUE, NUM_INTERNAL_VALUES

CONSTRUCTOR_DIM = 64
TOPOLOGICAL_PASSES = 1
STORAGE_LAMBDA = 0.05
LEARNED_MODES = ("learned_instantiation", "structure_blind_gate")
X20_MODES = ("canonical_live_mask", "all_records", *LEARNED_MODES)


def candidate_code(*, device=None, dtype=torch.float32) -> torch.Tensor:
    rows = []
    for i in range(NUM_CANDIDATES):
        rows.append(
            [
                i / 7.0,
                1.0,
                math.sin(math.pi * i / 8.0),
                math.cos(math.pi * i / 8.0),
                math.sin(2.0 * math.pi * i / 8.0),
                math.cos(2.0 * math.pi * i / 8.0),
                1.0 if (i & 1) else -1.0,
                1.0 if (i & 2) else -1.0,
                1.0 if (i & 4) else -1.0,
            ]
        )
    return torch.tensor(rows, device=device, dtype=dtype)


class ProgramStateConstructor(nn.Module):
    """Shared constructor over explicit temporal candidate-version graph structure."""

    def __init__(self, *, structure_blind: bool):
        super().__init__()
        self.structure_blind = bool(structure_blind)
        self.code_proj = nn.Linear(10, CONSTRUCTOR_DIM)
        self.command = nn.Embedding(16, CONSTRUCTOR_DIM)
        self.message = nn.Sequential(
            nn.Linear(2 * CONSTRUCTOR_DIM, 2 * CONSTRUCTOR_DIM),
            nn.SiLU(),
            nn.Linear(2 * CONSTRUCTOR_DIM, CONSTRUCTOR_DIM),
        )
        self.update = nn.GRUCell(CONSTRUCTOR_DIM, CONSTRUCTOR_DIM)
        self.gate = nn.Sequential(
            nn.Linear(CONSTRUCTOR_DIM, CONSTRUCTOR_DIM),
            nn.SiLU(),
            nn.Linear(CONSTRUCTOR_DIM, 1),
        )

    @staticmethod
    def version_graph_indices(program) -> dict[str, torch.Tensor]:
        """Build temporal version-node indices from syntax only.

        Local node IDs 0..7 are initial candidate versions. Operation t creates node
        8+t as the new version of its destination. Source-node IDs are captured before
        the destructive write, so overwritten versions remain distinct graph nodes.
        """
        batch_size = int(program.initial.shape[0])
        depth = int(program.commands.shape[1])
        device = program.initial.device
        current = torch.arange(NUM_CANDIDATES, device=device)[None, :].expand(batch_size, -1).clone()
        owners = torch.empty(batch_size, NUM_CANDIDATES + depth, device=device, dtype=torch.long)
        owners[:, :NUM_CANDIDATES] = torch.arange(NUM_CANDIDATES, device=device)[None, :]
        src_a, src_b = [], []
        batch = torch.arange(batch_size, device=device)
        for t in range(depth):
            a = program.arg_a[:, t]
            b = program.arg_b[:, t]
            d = program.dst[:, t]
            src_a.append(current[batch, a])
            src_b.append(current[batch, b])
            new_id = NUM_CANDIDATES + t
            owners[:, new_id] = d
            next_current = current.clone()
            next_current[batch, d] = new_id
            current = next_current
        return {
            "src_a": torch.stack(src_a, dim=1),
            "src_b": torch.stack(src_b, dim=1),
            "new_dst": torch.arange(
                NUM_CANDIDATES, NUM_CANDIDATES + depth, device=device, dtype=torch.long
            )[None, :].expand(batch_size, -1),
            "owners": owners,
            "final_output": current[:, OUTPUT_CANDIDATE],
        }

    @staticmethod
    def _gather_nodes(h: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        batch = torch.arange(h.shape[0], device=h.device)
        return h[batch, index]

    @staticmethod
    def _scatter_nodes(h: torch.Tensor, index: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        out = h.clone()
        batch = torch.arange(h.shape[0], device=h.device)
        out[batch, index] = value
        return out

    def _candidate_base(self, *, batch_size: int, device) -> torch.Tensor:
        codes = candidate_code(device=device, dtype=self.code_proj.weight.dtype)
        output_flag = torch.zeros(NUM_CANDIDATES, 1, device=device, dtype=codes.dtype)
        output_flag[OUTPUT_CANDIDATE, 0] = 1.0
        base = self.code_proj(torch.cat([codes, output_flag], dim=-1))
        return base[None, :, :].expand(batch_size, -1, -1)

    def _blind_forward(self, program) -> torch.Tensor:
        batch_size = int(program.initial.shape[0])
        base = self._candidate_base(batch_size=batch_size, device=program.initial.device)
        # Frozen structure-blind ablation: supplied candidate/output identity plus aggregate
        # command-family statistics, but no operand/destination/version connectivity.
        cmd_mean = self.command(program.commands).mean(dim=1)
        source = base.mean(dim=1)
        msg = self.message(torch.cat([source, cmd_mean], dim=-1))
        old = base.reshape(batch_size * NUM_CANDIDATES, CONSTRUCTOR_DIM)
        expanded = msg[:, None, :].expand(-1, NUM_CANDIDATES, -1).reshape_as(old)
        h = self.update(expanded, old).reshape(batch_size, NUM_CANDIDATES, CONSTRUCTOR_DIM)
        return torch.sigmoid(self.gate(h).squeeze(-1))

    def _graph_forward(self, program) -> torch.Tensor:
        graph = self.version_graph_indices(program)
        batch_size = int(program.initial.shape[0])
        depth = int(program.commands.shape[1])
        owners = graph["owners"]
        node_count = int(owners.shape[1])
        codes = candidate_code(device=program.initial.device, dtype=self.code_proj.weight.dtype)
        node_codes = codes[owners]
        root_flag = torch.zeros(batch_size, node_count, 1, device=program.initial.device, dtype=node_codes.dtype)
        batch = torch.arange(batch_size, device=program.initial.device)
        root_flag[batch, graph["final_output"], 0] = 1.0
        h = self.code_proj(torch.cat([node_codes, root_flag], dim=-1))

        # One exact reverse-topological graph pass. Operation-state messages are computed
        # from the post-write destination version plus command identity and sent only to
        # the source versions that existed before that destructive write.
        assert TOPOLOGICAL_PASSES == 1
        for t in range(depth - 1, -1, -1):
            post_id = graph["new_dst"][:, t]
            post = self._gather_nodes(h, post_id)
            cmd = self.command(program.commands[:, t])
            op_state = self.message(torch.cat([post, cmd], dim=-1))
            for source_id in (graph["src_a"][:, t], graph["src_b"][:, t]):
                old = self._gather_nodes(h, source_id)
                new = self.update(op_state, old)
                h = self._scatter_nodes(h, source_id, new)

        # Aggregate all temporal versions belonging to each candidate. This is the only
        # collapse from version-level graph state to the candidate-level existence gate.
        owner_onehot = F.one_hot(owners, NUM_CANDIDATES).to(dtype=h.dtype)
        summed = torch.einsum("bnc,bnd->bcd", owner_onehot, h)
        counts = owner_onehot.sum(dim=1).clamp_min(1.0)
        candidate_h = summed / counts[:, :, None]
        return torch.sigmoid(self.gate(candidate_h).squeeze(-1))

    def forward(self, program) -> torch.Tensor:
        if self.structure_blind:
            return self._blind_forward(program)
        return self._graph_forward(program)


class GatedStateExecutor(nn.Module):
    """Local-equivariant transition kernel over eight candidate records with soft existence gates."""

    def __init__(self, d_model: int = 96):
        super().__init__()
        self.kernel = LocalEquivariantTransitionModel(d_model=d_model)

    @staticmethod
    def _record(probs: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        batch = torch.arange(probs.shape[0], device=probs.device)
        return probs[batch, index]

    def initial_probs(self, initial: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        dtype = self.kernel.value.weight.dtype
        values = F.one_hot(initial, VALUE_MODULUS).to(dtype=dtype)
        probs = torch.zeros(
            initial.shape[0], NUM_CANDIDATES, NUM_INTERNAL_VALUES,
            device=initial.device, dtype=dtype,
        )
        probs[:, :, :VALUE_MODULUS] = gates[:, :, None] * values
        probs[:, :, EMPTY_VALUE] = 1.0 - gates
        return probs

    def step_logits(self, probs, command, a, b, dst):
        va = self._record(probs, a) @ self.kernel.value.weight
        vb = self._record(probs, b) @ self.kernel.value.weight
        vd = self._record(probs, dst) @ self.kernel.value.weight
        return self.kernel.transition(
            torch.cat([self.kernel.command(command), va, vb, vd], dim=-1)
        )

    @staticmethod
    def _internal_value(world_probs: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros(world_probs.shape[0], 1, device=world_probs.device, dtype=world_probs.dtype)
        return torch.cat([world_probs, zeros], dim=-1)

    def rollout_soft(self, program, gates: torch.Tensor) -> torch.Tensor:
        probs = self.initial_probs(program.initial, gates)
        states = []
        batch = torch.arange(program.initial.shape[0], device=program.initial.device)
        for t in range(program.depth):
            dst = program.dst[:, t]
            logits = self.step_logits(
                probs, program.commands[:, t], program.arg_a[:, t], program.arg_b[:, t], dst
            )
            new_value = self._internal_value(F.softmax(logits, dim=-1))
            gdst = gates[batch, dst]
            old = probs[batch, dst]
            updated = old * (1.0 - gdst[:, None]) + new_value * gdst[:, None]
            probs = probs.clone()
            probs[batch, dst] = updated
            states.append(probs)
        return torch.stack(states, dim=1)

    @torch.no_grad()
    def rollout_hard(self, program, gates: torch.Tensor) -> torch.Tensor:
        hard = (gates >= 0.5).to(dtype=self.kernel.value.weight.dtype)
        probs = self.initial_probs(program.initial, hard)
        states = []
        batch = torch.arange(program.initial.shape[0], device=program.initial.device)
        for t in range(program.depth):
            dst = program.dst[:, t]
            logits = self.step_logits(
                probs, program.commands[:, t], program.arg_a[:, t], program.arg_b[:, t], dst
            )
            value = logits.argmax(dim=-1)
            world = F.one_hot(value, VALUE_MODULUS).to(dtype=probs.dtype)
            new_value = self._internal_value(world)
            gdst = hard[batch, dst]
            old = probs[batch, dst]
            updated = old * (1.0 - gdst[:, None]) + new_value * gdst[:, None]
            probs = probs.clone()
            probs[batch, dst] = updated
            states.append(probs[:, :, :VALUE_MODULUS].argmax(dim=-1))
        return torch.stack(states, dim=1)


class StateInstantiationModel(nn.Module):
    def __init__(self, *, mode: str, d_model: int = 96):
        super().__init__()
        if mode not in X20_MODES:
            raise ValueError(mode)
        self.mode = mode
        self.executor = GatedStateExecutor(d_model=d_model)
        if mode in LEARNED_MODES:
            self.constructor = ProgramStateConstructor(structure_blind=(mode == "structure_blind_gate"))
        else:
            self.constructor = None

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def gates(self, batch: StateInstantiationBatch) -> torch.Tensor:
        if self.mode == "canonical_live_mask":
            return batch.live_mask.to(dtype=self.executor.kernel.value.weight.dtype)
        if self.mode == "all_records":
            return torch.ones(
                batch.batch_size, NUM_CANDIDATES,
                device=batch.program.initial.device,
                dtype=self.executor.kernel.value.weight.dtype,
            )
        assert self.constructor is not None
        return self.constructor(batch.program)

    def loss_components(self, batch: StateInstantiationBatch) -> dict[str, torch.Tensor]:
        gates = self.gates(batch)
        states = self.executor.rollout_soft(batch.program, gates)
        target = batch.program.target_states[:, -1, OUTPUT_CANDIDATE]
        output = states[:, -1, OUTPUT_CANDIDATE, :VALUE_MODULUS].clamp_min(1e-12)
        p = output.gather(1, target[:, None]).squeeze(1)
        answer = -p.log().mean()
        storage = gates.mean() if self.mode in LEARNED_MODES else torch.zeros_like(answer)
        total = answer + STORAGE_LAMBDA * storage if self.mode in LEARNED_MODES else answer
        return {"answer_loss": answer, "storage_penalty": storage, "total_loss": total}


def cloned_x20_models(d_model: int = 96) -> dict[str, StateInstantiationModel]:
    learned_seed = StateInstantiationModel(mode="learned_instantiation", d_model=d_model)
    learned_state = deepcopy(learned_seed.state_dict())
    learned = StateInstantiationModel(mode="learned_instantiation", d_model=d_model)
    blind = StateInstantiationModel(mode="structure_blind_gate", d_model=d_model)
    learned.load_state_dict(learned_state)
    blind.load_state_dict(learned_state)

    canonical = StateInstantiationModel(mode="canonical_live_mask", d_model=d_model)
    all_records = StateInstantiationModel(mode="all_records", d_model=d_model)
    executor_state = deepcopy(learned.executor.state_dict())
    canonical.executor.load_state_dict(executor_state)
    all_records.executor.load_state_dict(executor_state)
    return {
        "canonical_live_mask": canonical,
        "all_records": all_records,
        "learned_instantiation": learned,
        "structure_blind_gate": blind,
    }
