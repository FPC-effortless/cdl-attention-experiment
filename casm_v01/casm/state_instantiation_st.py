from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn

from .explicit_compute import VALUE_MODULUS
from .state_instantiation_data import NUM_CANDIDATES, StateInstantiationBatch
from .state_instantiation_model import (
    GatedStateExecutor,
    ProgramStateConstructor,
    STORAGE_LAMBDA,
)

X20_FROZEN_RESULT = "3225172c78ca44ad57a26d64b13ae24f122b96bb"
SOFT_MODE = "soft_x20_replication"
ST_GRAPH_MODE = "straight_through_instantiation"
ST_BLIND_MODE = "straight_through_structure_blind"
X20R_LEARNED_MODES = (SOFT_MODE, ST_GRAPH_MODE, ST_BLIND_MODE)
X20R_MODES = ("canonical_live_mask", "all_records", *X20R_LEARNED_MODES)


class _StraightThroughBinary(torch.autograd.Function):
    """Exact binary forward with identity gradient to the soft gate input."""

    @staticmethod
    def forward(ctx, g_soft: torch.Tensor) -> torch.Tensor:
        del ctx
        return (g_soft >= 0.5).to(dtype=g_soft.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        del ctx
        return (grad_output,)


def straight_through_binary(g_soft: torch.Tensor) -> torch.Tensor:
    """Binary forward value with identity-through-soft backward path."""
    return _StraightThroughBinary.apply(g_soft)


class X20RStateInstantiationModel(nn.Module):
    """X20 state constructor with either soft or straight-through training existence."""

    def __init__(self, *, mode: str, d_model: int = 96):
        super().__init__()
        if mode not in X20R_MODES:
            raise ValueError(mode)
        self.mode = mode
        self.executor = GatedStateExecutor(d_model=d_model)
        if mode in X20R_LEARNED_MODES:
            self.constructor = ProgramStateConstructor(structure_blind=(mode == ST_BLIND_MODE))
        else:
            self.constructor = None

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def soft_gates(self, batch: StateInstantiationBatch) -> torch.Tensor:
        if self.mode == "canonical_live_mask":
            return batch.live_mask.to(dtype=self.executor.kernel.value.weight.dtype)
        if self.mode == "all_records":
            return torch.ones(
                batch.batch_size,
                NUM_CANDIDATES,
                device=batch.program.initial.device,
                dtype=self.executor.kernel.value.weight.dtype,
            )
        assert self.constructor is not None
        return self.constructor(batch.program)

    def training_gates(self, batch: StateInstantiationBatch) -> torch.Tensor:
        g_soft = self.soft_gates(batch)
        if self.mode in (ST_GRAPH_MODE, ST_BLIND_MODE):
            return straight_through_binary(g_soft)
        return g_soft

    def loss_components(self, batch: StateInstantiationBatch) -> dict[str, torch.Tensor]:
        gates = self.training_gates(batch)
        states = self.executor.rollout_soft(batch.program, gates)
        target = batch.program.target_states[:, -1, 0]
        output = states[:, -1, 0, :VALUE_MODULUS].clamp_min(1e-12)
        p = output.gather(1, target[:, None]).squeeze(1)
        answer = -p.log().mean()
        storage = gates.mean() if self.mode in X20R_LEARNED_MODES else torch.zeros_like(answer)
        total = answer + STORAGE_LAMBDA * storage if self.mode in X20R_LEARNED_MODES else answer
        return {"answer_loss": answer, "storage_penalty": storage, "total_loss": total}


def cloned_x20r_models(d_model: int = 96) -> dict[str, X20RStateInstantiationModel]:
    seed_model = X20RStateInstantiationModel(mode=SOFT_MODE, d_model=d_model)
    learned_state = deepcopy(seed_model.state_dict())

    soft = X20RStateInstantiationModel(mode=SOFT_MODE, d_model=d_model)
    st_graph = X20RStateInstantiationModel(mode=ST_GRAPH_MODE, d_model=d_model)
    st_blind = X20RStateInstantiationModel(mode=ST_BLIND_MODE, d_model=d_model)
    for model in (soft, st_graph, st_blind):
        model.load_state_dict(learned_state)

    canonical = X20RStateInstantiationModel(mode="canonical_live_mask", d_model=d_model)
    all_records = X20RStateInstantiationModel(mode="all_records", d_model=d_model)
    executor_state = deepcopy(soft.executor.state_dict())
    canonical.executor.load_state_dict(executor_state)
    all_records.executor.load_state_dict(executor_state)

    return {
        "canonical_live_mask": canonical,
        "all_records": all_records,
        SOFT_MODE: soft,
        ST_GRAPH_MODE: st_graph,
        ST_BLIND_MODE: st_blind,
    }
