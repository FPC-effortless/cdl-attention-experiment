from __future__ import annotations

from copy import deepcopy

import torch

from .reuse_merge import ReuseSignature, compatible, propose_reuse_groups
from .state_instantiation_data import NUM_CANDIDATES, StateInstantiationBatch
from .state_instantiation_local_credit import (
    CANONICAL_MODE,
    LOCAL_CREDIT_BLIND_MODE,
    LOCAL_CREDIT_MODE,
    _answer_loss,
    _new_model,
    cloned_x20u_models,
    forced_on_off_gates,
    local_counterfactual_risk,
    straight_through_binary,
)
from .state_instantiation_st import X20RStateInstantiationModel

NO_REUSE_MODE = "no_reuse_control"
REUSE_GRAPH_MODE = "reuse_merge_graph"
REUSE_BLIND_MODE = "reuse_merge_structure_blind"
X20V_MODES = (NO_REUSE_MODE, REUSE_GRAPH_MODE, REUSE_BLIND_MODE)
X20V_LEARNED_MODES = (REUSE_GRAPH_MODE, REUSE_BLIND_MODE)
HARD_GATE_THRESHOLD = 0.5


def _signature_groups(batch: StateInstantiationBatch, *, structure_blind: bool) -> tuple[tuple[int, ...], ...]:
    p = batch.program
    # Supplied observations only: initial value is the identity cue; version is the
    # number of writes to the candidate; dependency signature is the final direct
    # source-version pair. No hidden target/live/cardinality information is used.
    write_counts = torch.zeros_like(p.initial)
    for t in range(p.depth):
        d = p.dst[:, t]
        write_counts.scatter_add_(1, d[:, None], torch.ones_like(d[:, None]))
    graph = _new_model(REUSE_GRAPH_MODE, d_model=96).constructor.version_graph_indices(p)
    final_ids = graph["final_output"]
    owners = graph["owners"]
    groups_per_batch = []
    for b in range(p.initial.shape[0]):
        sigs = []
        for c in range(NUM_CANDIDATES):
            node = int(final_ids[b]) if c == int(p.dst[b, -1]) and p.depth else c
            # Candidate-local final version ID and its owner-derived direct dependency
            # signature. Initial value is included as the supplied identity cue.
            dep = []
            if not structure_blind:
                for t in range(p.depth):
                    if int(p.dst[b, t]) == c:
                        dep.extend((int(graph["src_a"][b, t]), int(graph["src_b"][b, t])))
            version = int(write_counts[b, c]) if not structure_blind else 0
            entity = int(p.initial[b, c])
            sigs.append(ReuseSignature(entity, version, tuple(dep)))
        groups_per_batch.append(propose_reuse_groups(sigs))
    return tuple(groups_per_batch)


def merge_gates(gates: torch.Tensor, batch: StateInstantiationBatch, *, structure_blind: bool) -> tuple[torch.Tensor, dict[str, float]]:
    groups = _signature_groups(batch, structure_blind=structure_blind)
    out = gates.clone()
    duplicate_rates = []
    precisions = []
    recalls = []
    count_errors = []
    live = batch.live_mask.bool()
    for b, gs in enumerate(groups):
        active_before = gates[b] >= HARD_GATE_THRESHOLD
        merged = torch.zeros_like(gates[b])
        for group in gs:
            idx = torch.tensor(group, device=gates.device)
            score = gates[b, idx].max()
            rep = group[0]
            merged[rep] = score
        out[b] = merged
        active_after = merged >= HARD_GATE_THRESHOLD
        dup = max(0, int(active_before.sum()) - int(active_after.sum()))
        duplicate_rates.append(dup / max(1, int(active_before.sum())))
        tp = int((active_after & live[b]).sum())
        fp = int((active_after & ~live[b]).sum())
        fn = int((~active_after & live[b]).sum())
        precisions.append(tp / max(1, tp + fp))
        recalls.append(tp / max(1, tp + fn))
        count_errors.append(abs(int(active_after.sum()) - int(live[b].sum())))
    stats = {
        "duplicate_active_identity_rate": sum(duplicate_rates) / len(duplicate_rates),
        "merge_precision": sum(precisions) / len(precisions),
        "merge_recall": sum(recalls) / len(recalls),
        "mean_hard_record_count_error": sum(count_errors) / len(count_errors),
    }
    return out, stats


class X20VStateInstantiationModel(X20RStateInstantiationModel):
    def __init__(self, *, mode: str, d_model: int = 96):
        if mode == NO_REUSE_MODE:
            super().__init__(mode=LOCAL_CREDIT_MODE, d_model=d_model)
        elif mode == REUSE_GRAPH_MODE:
            super().__init__(mode=LOCAL_CREDIT_MODE, d_model=d_model)
        elif mode == REUSE_BLIND_MODE:
            super().__init__(mode=LOCAL_CREDIT_BLIND_MODE, d_model=d_model)
        else:
            raise ValueError(mode)
        self.x20v_mode = mode

    def soft_gates(self, batch: StateInstantiationBatch) -> torch.Tensor:
        base = super().soft_gates(batch)
        if self.x20v_mode == NO_REUSE_MODE:
            return base
        merged, _ = merge_gates(base, batch, structure_blind=self.x20v_mode == REUSE_BLIND_MODE)
        return merged


def cloned_x20v_models(d_model: int = 96) -> dict[str, X20VStateInstantiationModel]:
    base = cloned_x20u_models(d_model=d_model)
    out = {}
    for mode in X20V_MODES:
        model = X20VStateInstantiationModel(mode=mode, d_model=d_model)
        source = base[LOCAL_CREDIT_MODE]
        model.load_state_dict(source.state_dict(), strict=True)
        out[mode] = model
    return out


def reuse_merge_loss_components(model: X20VStateInstantiationModel, batch: StateInstantiationBatch, *, mode: str) -> dict[str, torch.Tensor]:
    if mode not in X20V_MODES:
        raise ValueError(mode)
    g_soft = model.soft_gates(batch)
    g_st = straight_through_binary(g_soft)
    a_hard = _answer_loss(model, batch, g_st)
    a_soft = _answer_loss(model, batch, g_soft)
    if mode == NO_REUSE_MODE:
        local = torch.zeros_like(a_hard)
        task = 0.5 * a_hard + 0.5 * a_soft
    else:
        a_on, a_off = forced_on_off_gates(g_soft)
        local = local_counterfactual_risk(g_soft, a_on, a_off)
        task = 0.5 * (0.5 * a_hard + 0.5 * a_soft) + 0.5 * local
    storage = g_soft.mean()
    total = task + 0.05 * storage
    return {
        "hard_answer_loss": a_hard,
        "soft_answer_loss": a_soft,
        "local_counterfactual_risk": local,
        "task_loss": task,
        "storage_penalty": storage,
        "storage_cost": 0.05 * storage,
        "total_loss": total,
    }
