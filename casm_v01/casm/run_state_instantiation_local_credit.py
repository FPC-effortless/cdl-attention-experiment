from __future__ import annotations

import argparse
import json
import math
import platform
import random
import subprocess
from pathlib import Path

import torch

from .run_stable_cardinality_executor import set_optimizer_lr
from .run_state_instantiation_credit import _selection_metrics, cosine_lr, evaluate_mode
from .state_instantiation_credit import STORAGE_LAMBDA
from .state_instantiation_data import (
    NUM_CANDIDATES,
    TRAIN_LIVE_CARDINALITIES,
    UNSEEN_LIVE_CARDINALITIES,
    make_state_instantiation_batch,
    training_live_cardinality_for_step,
)
from .state_instantiation_local_credit import (
    CANONICAL_MODE,
    DUAL_REPLICATION_MODE,
    GLOBAL_TASK_WEIGHT,
    LOCAL_CREDIT_BLIND_MODE,
    LOCAL_CREDIT_MODE,
    LOCAL_TASK_WEIGHT,
    PREREGISTERED_STEPS,
    X20U_GRAPH_MODES,
    X20U_LEARNED_MODES,
    X20U_MODES,
    cloned_x20u_models,
    local_credit_loss_components,
)

HARNESS_VERSION = "casm-x20u-local-counterfactual-credit-v0-2026-09-05"
LR_MAX = 2e-3
LR_MIN = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def train_step(model, optimizer, batch, *, mode: str, lr: float):
    set_optimizer_lr(optimizer, lr)
    optimizer.zero_grad(set_to_none=True)
    parts = local_credit_loss_components(model, batch, mode=mode)
    for key, value in parts.items():
        scalar = float(value.detach())
        if not math.isfinite(scalar) or scalar < 0.0:
            raise RuntimeError(f"invalid {key}: {scalar}")
    parts["total_loss"].backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
    if not math.isfinite(float(grad_norm)):
        raise RuntimeError("non-finite gradient norm")
    optimizer.step()
    return {
        **{k: float(v.detach()) for k, v in parts.items()},
        "grad_norm": float(grad_norm),
        "lr": float(lr),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=PREREGISTERED_STEPS)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--train-depth", type=int, default=12)
    p.add_argument("--seed", type=int, default=20261211)
    p.add_argument("--eval-seed", type=int, default=20261291)
    p.add_argument("--eval-n", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--output", type=Path, default=Path("casm-x20u-output/results.json"))
    args = p.parse_args()

    if args.steps != PREREGISTERED_STEPS or args.batch_size != 128 or args.train_depth != 12 or args.eval_n != 256:
        raise ValueError("CASM-X20U frozen run contract mismatch")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    models = cloned_x20u_models(d_model=96)
    optimizers = {
        mode: torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=WEIGHT_DECAY)
        for mode, model in models.items()
    }
    learned_counts = {m: models[m].parameter_count() for m in X20U_LEARNED_MODES}
    learned_trainable = {m: models[m].trainable_parameter_count() for m in X20U_LEARNED_MODES}
    assert len(set(learned_counts.values())) == 1
    assert len(set(learned_trainable.values())) == 1

    initial_gate_stats_train_only = {}
    for mode in X20U_LEARNED_MODES:
        initial_gate_stats_train_only[mode] = {}
        for nlive in TRAIN_LIVE_CARDINALITIES:
            b = make_state_instantiation_batch(
                32,
                args.train_depth,
                args.seed + nlive,
                live_cardinality=nlive,
                split="train",
            )
            initial_gate_stats_train_only[mode][str(nlive)] = _selection_metrics(models[mode].soft_gates(b), b.live_mask)

    cardinality_counts = {str(n): 0 for n in TRAIN_LIVE_CARDINALITIES}
    minimum_task_loss = {m: float("inf") for m in X20U_MODES}
    final_task_loss = {m: float("nan") for m in X20U_MODES}
    minimum_soft_answer_loss = {m: float("inf") for m in X20U_MODES}
    minimum_hard_answer_loss = {m: float("inf") for m in X20U_MODES}
    maximum_post4000_grad_norm = {m: 0.0 for m in X20U_MODES}
    maximum_local_advantage = {m: 0.0 for m in X20U_MODES}
    history = []

    for step in range(1, args.steps + 1):
        nlive = training_live_cardinality_for_step(step)
        cardinality_counts[str(nlive)] += 1
        batch = make_state_instantiation_batch(
            args.batch_size,
            args.train_depth,
            args.seed * 1_000_003 + step * 97,
            live_cardinality=nlive,
            split="train",
        )
        lr = cosine_lr(step)
        row = {"step": step, "live_cardinality": nlive}
        for mode in X20U_MODES:
            models[mode].train()
            row[mode] = train_step(models[mode], optimizers[mode], batch, mode=mode, lr=lr)
            minimum_task_loss[mode] = min(minimum_task_loss[mode], row[mode]["task_loss"])
            final_task_loss[mode] = row[mode]["task_loss"]
            minimum_soft_answer_loss[mode] = min(minimum_soft_answer_loss[mode], row[mode]["soft_answer_loss"])
            minimum_hard_answer_loss[mode] = min(minimum_hard_answer_loss[mode], row[mode]["hard_answer_loss"])
            maximum_local_advantage[mode] = max(maximum_local_advantage[mode], row[mode]["mean_abs_local_advantage"])
            if step > 4000:
                maximum_post4000_grad_norm[mode] = max(maximum_post4000_grad_norm[mode], row[mode]["grad_norm"])

        if step == 1 or step % 500 == 0 or step == args.steps:
            if step % 1000 == 0 or step in (1, args.steps):
                row["raw_gate_diagnostics"] = {
                    mode: _selection_metrics(models[mode].soft_gates(batch), batch.live_mask)
                    for mode in X20U_LEARNED_MODES
                }
            history.append(row)
            print(json.dumps(row), flush=True)

    for model in models.values():
        model.eval()

    suites = [
        ("iid_depth_12", "iid", 12),
        ("composition_depth_24", "composition", 24),
        ("stress_depth_48", "composition", 48),
        ("stress_depth_96", "composition", 96),
    ]
    evaluation = {}
    for nlive in (*TRAIN_LIVE_CARDINALITIES, *UNSEEN_LIVE_CARDINALITIES):
        evaluation[str(nlive)] = {}
        for si, (suite, split, depth) in enumerate(suites):
            evaluation[str(nlive)][suite] = {"split": split, "depth": depth}
            eval_seed = args.eval_seed + nlive * 1_000_003 + si * 100_003
            for mode in X20U_MODES:
                evaluation[str(nlive)][suite][mode] = evaluate_mode(
                    models[mode],
                    live_cardinality=nlive,
                    depth=depth,
                    split=split,
                    n=args.eval_n,
                    batch_size=args.eval_batch_size,
                    seed=eval_seed,
                )
            print(f"live_n={nlive} {suite}", json.dumps(evaluation[str(nlive)][suite], sort_keys=True), flush=True)

    report = {
        "harness_version": HARNESS_VERSION,
        "git_sha": _git_sha(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "question": "does per-record final-answer counterfactual credit robustly repair discrete state-existence selection?",
        "seed": args.seed,
        "eval_seed": args.eval_seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "train_depth": args.train_depth,
        "eval_n": args.eval_n,
        "candidate_count": NUM_CANDIDATES,
        "training_live_cardinalities": list(TRAIN_LIVE_CARDINALITIES),
        "unseen_live_cardinalities": list(UNSEEN_LIVE_CARDINALITIES),
        "training_cardinality_counts": cardinality_counts,
        "parameters": {m: models[m].parameter_count() for m in X20U_MODES},
        "trainable_parameters": {m: models[m].trainable_parameter_count() for m in X20U_MODES},
        "minimum_task_loss": minimum_task_loss,
        "final_task_loss": final_task_loss,
        "minimum_soft_answer_loss": minimum_soft_answer_loss,
        "minimum_hard_answer_loss": minimum_hard_answer_loss,
        "maximum_post4000_grad_norm": maximum_post4000_grad_norm,
        "maximum_local_advantage": maximum_local_advantage,
        "optimizer_contract": {
            "optimizer": "AdamW",
            "weight_decay": WEIGHT_DECAY,
            "global_gradient_clip_norm": GRAD_CLIP_NORM,
            "lr_max": LR_MAX,
            "lr_min": LR_MIN,
            "steps": PREREGISTERED_STEPS,
        },
        "objective_contract": {
            DUAL_REPLICATION_MODE: "0.5*A_hard + 0.5*A_soft + 0.05*S_hard",
            LOCAL_CREDIT_MODE: "0.5*(0.5*A_hard+0.5*A_soft) + 0.5*L_local + 0.05*S_hard",
            LOCAL_CREDIT_BLIND_MODE: "0.5*(0.5*A_hard+0.5*A_soft) + 0.5*L_local + 0.05*S_hard",
            "local_risk": "mean_i[g_soft_i*stopgrad(A_on_i)+(1-g_soft_i)*stopgrad(A_off_i)]",
            "counterfactual_on": "same g_soft except candidate i forced to 1",
            "counterfactual_off": "same g_soft except candidate i forced to 0",
            "global_task_weight": GLOBAL_TASK_WEIGHT,
            "local_task_weight": LOCAL_TASK_WEIGHT,
            "storage_lambda": STORAGE_LAMBDA,
            "same_final_answer_target": True,
            "counterfactual_path_adds_no_labels": True,
        },
        "constructor_contract": {
            "learned_live_mask_input": False,
            "learned_active_cardinality_input": False,
            "per_candidate_parameter_table": False,
            "hard_gate_threshold": 0.5,
            "straight_through_forward_binary": True,
            "graph_conditioned_modes": list(X20U_GRAPH_MODES),
            "structure_blind_ablation": LOCAL_CREDIT_BLIND_MODE,
        },
        "supervision_contract": {
            "final_answer_only": True,
            "live_mask_loss": False,
            "cardinality_loss": False,
            "intermediate_state_targets": False,
            "hidden_register_targets": False,
            "semantic_operator_labels": False,
            "causal_slice_labels": False,
            "counterfactual_pseudo_labels": False,
            "pretraining_unseen_live_cardinality": False,
        },
        "initial_gate_stats_train_cardinalities_only": initial_gate_stats_train_only,
        "training_history": history,
        "evaluation": evaluation,
    }

    for values in (
        minimum_task_loss,
        final_task_loss,
        minimum_soft_answer_loss,
        minimum_hard_answer_loss,
        maximum_post4000_grad_norm,
        maximum_local_advantage,
    ):
        if not all(math.isfinite(v) and v >= 0.0 for v in values.values()):
            raise RuntimeError("non-finite report scalar")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("RESULT_JSON")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
