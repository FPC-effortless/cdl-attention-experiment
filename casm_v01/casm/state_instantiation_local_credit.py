from __future__ import annotations

from copy import deepcopy

import torch

from .explicit_compute import ProgramBatch, VALUE_MODULUS
from .state_instantiation_credit import (
    CANONICAL_MODE,
    DUAL_CREDIT_MODE,
    STORAGE_LAMBDA,
    credit_loss_components,
)
from .state_instantiation_data import NUM_CANDIDATES, StateInstantiationBatch
from .state_instantiation_st import ST_BLIND_MODE, ST_GRAPH_MODE, X20RStateInstantiationModel, straight_through_binary

X20T_FROZEN_RESULT = "a122c447efc31054b61134b3271cbc282f167ddb"
X20U_PREREGISTRATION = "524b372f89e2b3fd554131782c87278c58bf0552"

CANONICAL_MODE = "canonical_live_mask"
DUAL_REPLICATION_MODE = "dual_credit_replication"
LOCAL_CREDIT_MODE = "local_counterfactual_credit"
LOCAL_CREDIT_BLIND_MODE = "local_counterfactual_credit_structure_blind"

X20U_LEARNED_MODES = (DUAL_REPLICATION_MODE, LOCAL_CREDIT_MODE, LOCAL_CREDIT_BLIND_MODE)
X20U_GRAPH_MODES = (DUAL_REPLICATION_MODE, LOCAL_CREDIT_MODE)
X20U_MODES = (CANONICAL_MODE, *X20U_LEARNED_MODES)

GLOBAL_TASK_WEIGHT = 0.5
LOCAL_TASK_WEIGHT = 0.5
PREREGISTERED_STEPS = 12_000


def _new_model(mode: str, *, d_model: int) -> X20RStateInstantiationModel:
    if mode == CANONICAL_MODE:
        return X20RStateInstantiationModel(mode=CANONICAL_MODE, d_model=d_model)
    if mode in (DUAL_REPLICATION_MODE, LOCAL_CREDIT_MODE):
        return X20RStateInstantiationModel(mode=ST_GRAPH_MODE, d_model=d_model)
    if mode == LOCAL_CREDIT_BLIND_MODE:
        return X20RStateInstantiationModel(mode=ST_BLIND_MODE, d_model=d_model)
    raise ValueError(mode)


def cloned_x20u_models(d_model: int = 96) -> dict[str, X20RStateInstantiationModel]:
    seed_graph = _new_model(DUAL_REPLICATION_MODE, d_model=d_model)
    learned_state = deepcopy(seed_graph.state_dict())

    models: dict[str, X20RStateInstantiationModel] = {}
    for mode in X20U_LEARNED_MODES:
        model = _new_model(mode, d_model=d_model)
        model.load_state_dict(learned_state)
        models[mode] = model

    canonical = _new_model(CANONICAL_MODE, d_model=d_model)
    canonical.executor.load_state_dict(deepcopy(seed_graph.executor.state_dict()))
    models[CANONICAL_MODE] = canonical
    return {mode: models[mode] for mode in X20U_MODES}


def _repeat_program(program: ProgramBatch, factor: int) -> ProgramBatch:
    if factor < 1:
        raise ValueError(factor)
    return ProgramBatch(**{name: value.repeat_interleave(factor, dim=0) for name, value in program.__dict__.items()})


def _answer_nll_per_example(
    model: X20RStateInstantiationModel,
    program: ProgramBatch,
    gates: torch.Tensor,
) -> torch.Tensor:
    states = model.executor.rollout_soft(program, gates)
    target = program.target_states[:, -1, 0]
    output = states[:, -1, 0, :VALUE_MODULUS].clamp_min(1e-12)
    p = output.gather(1, target[:, None]).squeeze(1)
    return -p.log()


def _answer_loss(
    model: X20RStateInstantiationModel,
    batch: StateInstantiationBatch,
    gates: torch.Tensor,
) -> torch.Tensor:
    return _answer_nll_per_example(model, batch.program, gates).mean()


def forced_on_off_gates(g_soft: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return [B,C,C] tensors with candidate i forced on/off in slice [:,i,:]."""
    if g_soft.ndim != 2 or g_soft.shape[1] != NUM_CANDIDATES:
        raise ValueError(tuple(g_soft.shape))
    batch_size, candidates = g_soft.shape
    on = g_soft[:, None, :].expand(batch_size, candidates, candidates).clone()
    off = on.clone()
    idx = torch.arange(candidates, device=g_soft.device)
    on[:, idx, idx] = 1.0
    off[:, idx, idx] = 0.0
    return on, off


def counterfactual_answer_losses(
    model: X20RStateInstantiationModel,
    batch: StateInstantiationBatch,
    g_soft: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return detached per-example/per-candidate NLLs for forced-on and forced-off gates."""
    on, off = forced_on_off_gates(g_soft.detach())
    batch_size, candidates, _ = on.shape
    paired = torch.stack((on, off), dim=2).reshape(batch_size, 2 * candidates, candidates)
    flat_gates = paired.reshape(batch_size * 2 * candidates, candidates)
    expanded_program = _repeat_program(batch.program, 2 * candidates)
    with torch.no_grad():
        losses = _answer_nll_per_example(model, expanded_program, flat_gates).reshape(batch_size, 2 * candidates)
    a_on = losses[:, 0::2]
    a_off = losses[:, 1::2]
    if not (torch.isfinite(a_on).all() and torch.isfinite(a_off).all()):
        raise RuntimeError("non-finite local counterfactual answer loss")
    return a_on, a_off


def local_counterfactual_risk(
    g_soft: torch.Tensor,
    a_on: torch.Tensor,
    a_off: torch.Tensor,
) -> torch.Tensor:
    if not (g_soft.shape == a_on.shape == a_off.shape):
        raise ValueError((tuple(g_soft.shape), tuple(a_on.shape), tuple(a_off.shape)))
    return (g_soft * a_on.detach() + (1.0 - g_soft) * a_off.detach()).mean()


def local_credit_loss_components(
    model: X20RStateInstantiationModel,
    batch: StateInstantiationBatch,
    *,
    mode: str,
) -> dict[str, torch.Tensor]:
    if mode not in X20U_MODES:
        raise ValueError(mode)

    if mode == DUAL_REPLICATION_MODE:
        old = credit_loss_components(model, batch, mode=DUAL_CREDIT_MODE)
        zero = torch.zeros_like(old["task_loss"])
        return {
            "hard_answer_loss": old["hard_answer_loss"],
            "soft_answer_loss": old["soft_answer_loss"],
            "global_task_loss": old["task_loss"],
            "local_counterfactual_risk": zero,
            "mean_abs_local_advantage": zero,
            "task_loss": old["task_loss"],
            "storage_penalty": old["storage_penalty"],
            "storage_cost": old["storage_cost"],
            "total_loss": old["total_loss"],
        }

    g_soft = model.soft_gates(batch)
    g_st = straight_through_binary(g_soft)
    a_hard = _answer_loss(model, batch, g_st)
    a_soft = _answer_loss(model, batch, g_soft)

    if mode == CANONICAL_MODE:
        zero = torch.zeros_like(a_hard)
        return {
            "hard_answer_loss": a_hard,
            "soft_answer_loss": a_soft,
            "global_task_loss": a_hard,
            "local_counterfactual_risk": zero,
            "mean_abs_local_advantage": zero,
            "task_loss": a_hard,
            "storage_penalty": zero,
            "storage_cost": zero,
            "total_loss": a_hard,
        }

    if mode not in (LOCAL_CREDIT_MODE, LOCAL_CREDIT_BLIND_MODE):
        raise AssertionError(mode)

    a_on, a_off = counterfactual_answer_losses(model, batch, g_soft)
    local = local_counterfactual_risk(g_soft, a_on, a_off)
    global_task = 0.5 * a_hard + 0.5 * a_soft
    task = GLOBAL_TASK_WEIGHT * global_task + LOCAL_TASK_WEIGHT * local
    storage = g_st.mean()
    total = task + STORAGE_LAMBDA * storage
    advantage = (a_off - a_on).abs().mean()

    return {
        "hard_answer_loss": a_hard,
        "soft_answer_loss": a_soft,
        "global_task_loss": global_task,
        "local_counterfactual_risk": local,
        "mean_abs_local_advantage": advantage,
        "task_loss": task,
        "storage_penalty": storage,
        "storage_cost": STORAGE_LAMBDA * storage,
        "total_loss": total,
    }
