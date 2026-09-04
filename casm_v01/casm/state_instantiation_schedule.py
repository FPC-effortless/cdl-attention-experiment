from __future__ import annotations

from copy import deepcopy

import torch

from .explicit_compute import VALUE_MODULUS
from .state_instantiation_data import StateInstantiationBatch
from .state_instantiation_st import (
    ST_BLIND_MODE,
    ST_GRAPH_MODE,
    X20RStateInstantiationModel,
)

X20R_FROZEN_RESULT = "a4b8ee98dd300dc51e4398c84020a2e90c2cccc6"
X20S_PREREGISTRATION = "cc9168011e8a2f578bc7dce9879dc42a6161e201"

CANONICAL_MODE = "canonical_live_mask"
IMMEDIATE_MODE = "immediate_st"
NO_STORAGE_MODE = "no_storage_st"
DELAYED_ABRUPT_MODE = "delayed_abrupt_st"
DELAYED_RAMP_MODE = "delayed_ramp_st"
DELAYED_RAMP_BLIND_MODE = "delayed_ramp_structure_blind"

X20S_GRAPH_MODES = (
    IMMEDIATE_MODE,
    NO_STORAGE_MODE,
    DELAYED_ABRUPT_MODE,
    DELAYED_RAMP_MODE,
)
X20S_LEARNED_MODES = (*X20S_GRAPH_MODES, DELAYED_RAMP_BLIND_MODE)
X20S_MODES = (CANONICAL_MODE, *X20S_LEARNED_MODES)

STORAGE_LAMBDA_FINAL = 0.05
WARMUP_STEPS = 1_000
RAMP_END_STEP = 2_000
PREREGISTERED_STEPS = 12_000


def storage_lambda(mode: str, step: int) -> float:
    """Deterministic preregistered X20S storage coefficient."""
    if mode not in X20S_MODES:
        raise ValueError(mode)
    if not 1 <= step <= PREREGISTERED_STEPS:
        raise ValueError(step)
    if mode == CANONICAL_MODE:
        return 0.0
    if mode == IMMEDIATE_MODE:
        return STORAGE_LAMBDA_FINAL
    if mode == NO_STORAGE_MODE:
        return 0.0
    if mode == DELAYED_ABRUPT_MODE:
        return 0.0 if step <= WARMUP_STEPS else STORAGE_LAMBDA_FINAL
    if mode in (DELAYED_RAMP_MODE, DELAYED_RAMP_BLIND_MODE):
        if step <= WARMUP_STEPS:
            return 0.0
        if step <= RAMP_END_STEP:
            return STORAGE_LAMBDA_FINAL * float(step - WARMUP_STEPS) / float(RAMP_END_STEP - WARMUP_STEPS)
        return STORAGE_LAMBDA_FINAL
    raise AssertionError(mode)


def _new_model_for_mode(mode: str, *, d_model: int) -> X20RStateInstantiationModel:
    if mode == CANONICAL_MODE:
        return X20RStateInstantiationModel(mode=CANONICAL_MODE, d_model=d_model)
    if mode == DELAYED_RAMP_BLIND_MODE:
        return X20RStateInstantiationModel(mode=ST_BLIND_MODE, d_model=d_model)
    if mode in X20S_GRAPH_MODES:
        return X20RStateInstantiationModel(mode=ST_GRAPH_MODE, d_model=d_model)
    raise ValueError(mode)


def cloned_x20s_models(d_model: int = 96) -> dict[str, X20RStateInstantiationModel]:
    """Return parameter-matched X20S regimes from one shared learned initialization."""
    seed_graph = _new_model_for_mode(IMMEDIATE_MODE, d_model=d_model)
    learned_state = deepcopy(seed_graph.state_dict())

    models: dict[str, X20RStateInstantiationModel] = {}
    for mode in X20S_LEARNED_MODES:
        model = _new_model_for_mode(mode, d_model=d_model)
        model.load_state_dict(learned_state)
        models[mode] = model

    canonical = _new_model_for_mode(CANONICAL_MODE, d_model=d_model)
    canonical.executor.load_state_dict(deepcopy(seed_graph.executor.state_dict()))
    models[CANONICAL_MODE] = canonical
    return {mode: models[mode] for mode in X20S_MODES}


def scheduled_loss_components(
    model: X20RStateInstantiationModel,
    batch: StateInstantiationBatch,
    *,
    mode: str,
    step: int,
) -> dict[str, torch.Tensor]:
    if mode not in X20S_MODES:
        raise ValueError(mode)
    gates = model.training_gates(batch)
    states = model.executor.rollout_soft(batch.program, gates)
    target = batch.program.target_states[:, -1, 0]
    output = states[:, -1, 0, :VALUE_MODULUS].clamp_min(1e-12)
    p = output.gather(1, target[:, None]).squeeze(1)
    answer = -p.log().mean()
    hard_record_fraction = gates.mean() if mode in X20S_LEARNED_MODES else torch.zeros_like(answer)
    coefficient = storage_lambda(mode, step)
    total = answer + coefficient * hard_record_fraction
    return {
        "answer_loss": answer,
        "hard_record_fraction": hard_record_fraction,
        "storage_lambda": torch.as_tensor(coefficient, dtype=answer.dtype, device=answer.device),
        "storage_cost": coefficient * hard_record_fraction,
        "total_loss": total,
    }
