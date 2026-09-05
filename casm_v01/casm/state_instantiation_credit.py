from __future__ import annotations

from copy import deepcopy

import torch

from .explicit_compute import VALUE_MODULUS
from .state_instantiation_data import StateInstantiationBatch
from .state_instantiation_st import (
    SOFT_MODE,
    ST_BLIND_MODE,
    ST_GRAPH_MODE,
    X20RStateInstantiationModel,
    straight_through_binary,
)

X20S_FROZEN_RESULT = "9d4ebc5805f11e7e6e208de006878508111e201e"
X20T_PREREGISTRATION = "9365e9e1ce3242df6305abe9fd816e66298caa64"

CANONICAL_MODE = "canonical_live_mask"
HARD_ONLY_MODE = "hard_only_st"
SOFT_X20_MODE = "soft_x20_replication"
SOFT_CREDIT_MODE = "soft_credit_hard_storage"
DUAL_CREDIT_MODE = "dual_credit_hard_storage"
SOFT_CREDIT_BLIND_MODE = "soft_credit_hard_storage_structure_blind"

X20T_GRAPH_MODES = (HARD_ONLY_MODE, SOFT_X20_MODE, SOFT_CREDIT_MODE, DUAL_CREDIT_MODE)
X20T_CREDIT_GRAPH_MODES = (SOFT_CREDIT_MODE, DUAL_CREDIT_MODE)
X20T_LEARNED_MODES = (*X20T_GRAPH_MODES, SOFT_CREDIT_BLIND_MODE)
X20T_MODES = (CANONICAL_MODE, *X20T_LEARNED_MODES)

STORAGE_LAMBDA = 0.05
PREREGISTERED_STEPS = 12_000


def _new_model(mode: str, *, d_model: int) -> X20RStateInstantiationModel:
    if mode == CANONICAL_MODE:
        return X20RStateInstantiationModel(mode=CANONICAL_MODE, d_model=d_model)
    if mode == SOFT_X20_MODE:
        return X20RStateInstantiationModel(mode=SOFT_MODE, d_model=d_model)
    if mode == SOFT_CREDIT_BLIND_MODE:
        return X20RStateInstantiationModel(mode=ST_BLIND_MODE, d_model=d_model)
    if mode in (HARD_ONLY_MODE, SOFT_CREDIT_MODE, DUAL_CREDIT_MODE):
        return X20RStateInstantiationModel(mode=ST_GRAPH_MODE, d_model=d_model)
    raise ValueError(mode)


def cloned_x20t_models(d_model: int = 96) -> dict[str, X20RStateInstantiationModel]:
    """Create parameter-matched learned regimes from one shared graph initialization."""
    seed_graph = _new_model(HARD_ONLY_MODE, d_model=d_model)
    learned_state = deepcopy(seed_graph.state_dict())

    models: dict[str, X20RStateInstantiationModel] = {}
    for mode in X20T_LEARNED_MODES:
        model = _new_model(mode, d_model=d_model)
        model.load_state_dict(learned_state)
        models[mode] = model

    canonical = _new_model(CANONICAL_MODE, d_model=d_model)
    canonical.executor.load_state_dict(deepcopy(seed_graph.executor.state_dict()))
    models[CANONICAL_MODE] = canonical
    return {mode: models[mode] for mode in X20T_MODES}


def _answer_loss(
    model: X20RStateInstantiationModel,
    batch: StateInstantiationBatch,
    gates: torch.Tensor,
) -> torch.Tensor:
    states = model.executor.rollout_soft(batch.program, gates)
    target = batch.program.target_states[:, -1, 0]
    output = states[:, -1, 0, :VALUE_MODULUS].clamp_min(1e-12)
    p = output.gather(1, target[:, None]).squeeze(1)
    return -p.log().mean()


def credit_loss_components(
    model: X20RStateInstantiationModel,
    batch: StateInstantiationBatch,
    *,
    mode: str,
) -> dict[str, torch.Tensor]:
    if mode not in X20T_MODES:
        raise ValueError(mode)

    g_soft = model.soft_gates(batch)
    g_st = straight_through_binary(g_soft)
    a_soft = _answer_loss(model, batch, g_soft)
    a_hard = _answer_loss(model, batch, g_st)

    if mode == CANONICAL_MODE:
        total = a_hard
        storage = torch.zeros_like(a_hard)
        task = a_hard
    elif mode == HARD_ONLY_MODE:
        storage = g_st.mean()
        task = a_hard
        total = task + STORAGE_LAMBDA * storage
    elif mode == SOFT_X20_MODE:
        storage = g_soft.mean()
        task = a_soft
        total = task + STORAGE_LAMBDA * storage
    elif mode in (SOFT_CREDIT_MODE, SOFT_CREDIT_BLIND_MODE):
        storage = g_st.mean()
        task = a_soft
        total = task + STORAGE_LAMBDA * storage
    elif mode == DUAL_CREDIT_MODE:
        storage = g_st.mean()
        task = 0.5 * a_hard + 0.5 * a_soft
        total = task + STORAGE_LAMBDA * storage
    else:
        raise AssertionError(mode)

    return {
        "hard_answer_loss": a_hard,
        "soft_answer_loss": a_soft,
        "task_loss": task,
        "storage_penalty": storage,
        "storage_cost": STORAGE_LAMBDA * storage if mode != CANONICAL_MODE else torch.zeros_like(storage),
        "total_loss": total,
    }
