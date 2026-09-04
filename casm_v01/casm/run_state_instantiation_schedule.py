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
from .state_instantiation_data import (
    NUM_CANDIDATES,
    OUTPUT_CANDIDATE,
    TRAIN_LIVE_CARDINALITIES,
    UNSEEN_LIVE_CARDINALITIES,
    make_state_instantiation_batch,
    training_live_cardinality_for_step,
)
from .state_instantiation_schedule import (
    CANONICAL_MODE,
    DELAYED_ABRUPT_MODE,
    DELAYED_RAMP_BLIND_MODE,
    DELAYED_RAMP_MODE,
    IMMEDIATE_MODE,
    NO_STORAGE_MODE,
    PREREGISTERED_STEPS,
    RAMP_END_STEP,
    STORAGE_LAMBDA_FINAL,
    WARMUP_STEPS,
    X20S_GRAPH_MODES,
    X20S_LEARNED_MODES,
    X20S_MODES,
    cloned_x20s_models,
    scheduled_loss_components,
    storage_lambda,
)

HARNESS_VERSION = "casm-x20s-storage-onset-v0-2026-09-04"
LR_MAX = 2e-3
LR_MIN = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0


def cosine_lr(step: int) -> float:
    if not 1 <= step <= PREREGISTERED_STEPS:
        raise ValueError(step)
    phase = math.pi * float(step - 1) / float(PREREGISTERED_STEPS - 1)
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1.0 + math.cos(phase))


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def train_step(model, optimizer, batch, *, mode: str, step: int, lr: float):
    set_optimizer_lr(optimizer, lr)
    optimizer.zero_grad(set_to_none=True)
    parts = scheduled_loss_components(model, batch, mode=mode, step=step)
    for key, value in parts.items():
        scalar = float(value.detach())
        if not math.isfinite(scalar) or scalar < 0.0:
            raise RuntimeError(f"invalid {key}: {scalar}")
    parts["total_loss"].backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
    optimizer.step()
    return {
        **{k: float(v.detach()) for k, v in parts.items()},
        "grad_norm": float(grad_norm),
        "lr": float(lr),
    }


def _selection_metrics(gates: torch.Tensor, live_mask: torch.Tensor) -> dict[str, float]:
    hard = gates >= 0.5
    live = live_mask.bool()
    tp = (hard & live).sum(dim=1).float()
    fp = (hard & ~live).sum(dim=1).float()
    fn = (~hard & live).sum(dim=1).float()
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
    count_error = (hard.sum(dim=1).float() - live.sum(dim=1).float()).abs()
    live_gate = gates[live]
    dead_gate = gates[~live]
    return {
        "hard_precision": float(precision.mean()),
        "hard_recall": float(recall.mean()),
        "hard_f1": float(f1.mean()),
        "mean_absolute_record_count_error": float(count_error.mean()),
        "mean_live_gate": float(live_gate.mean()) if live_gate.numel() else 1.0,
        "mean_distractor_gate": float(dead_gate.mean()) if dead_gate.numel() else 0.0,
        "mean_hard_record_count": float(hard.sum(dim=1).float().mean()),
    }


def _capability_metrics(pred: torch.Tensor, target: torch.Tensor, live_mask: torch.Tensor) -> dict[str, float]:
    live = live_mask[:, None, :].expand_as(target)
    correct = pred.eq(target)
    live_correct = correct | ~live
    step_exact = live_correct.all(dim=2).float().mean()
    live_acc = correct[live].float().mean()
    answer = pred[:, -1, OUTPUT_CANDIDATE].eq(target[:, -1, OUTPUT_CANDIDATE]).float().mean()
    final_live_exact = live_correct[:, -1].all(dim=1).float().mean()
    hidden_mask = live_mask.clone()
    hidden_mask[:, OUTPUT_CANDIDATE] = False
    hidden = hidden_mask[:, None, :].expand_as(target)
    hidden_acc = correct[hidden].float().mean() if hidden.any() else torch.tensor(1.0)
    return {
        "answer_final_accuracy": float(answer),
        "step_state_exact": float(step_exact),
        "live_register_accuracy": float(live_acc),
        "hidden_live_register_accuracy": float(hidden_acc),
        "final_live_state_exact": float(final_live_exact),
    }


@torch.no_grad()
def evaluate_mode(model, *, live_cardinality: int, depth: int, split: str, n: int, batch_size: int, seed: int):
    hard_metrics = []
    soft_metrics = []
    selection = []
    remaining = n
    batch_index = 0
    while remaining:
        size = min(batch_size, remaining)
        batch = make_state_instantiation_batch(
            size,
            depth,
            seed + batch_index * 1009,
            live_cardinality=live_cardinality,
            split=split,
        )
        gates = model.soft_gates(batch)
        if not torch.isfinite(gates).all():
            raise RuntimeError("non-finite gates")
        selection.append((_selection_metrics(gates, batch.live_mask), size))

        hard_pred = model.executor.rollout_hard(batch.program, gates)
        hard_metrics.append((_capability_metrics(hard_pred, batch.program.target_states, batch.live_mask), size))

        soft_probs = model.executor.rollout_soft(batch.program, gates)
        soft_pred = soft_probs[:, :, :, :16].argmax(dim=-1)
        soft_metrics.append((_capability_metrics(soft_pred, batch.program.target_states, batch.live_mask), size))
        remaining -= size
        batch_index += 1

    def merge(rows):
        total = sum(weight for _, weight in rows)
        keys = rows[0][0]
        return {k: sum(row[k] * w for row, w in rows) / total for k in keys}

    return {"hard": merge(hard_metrics), "soft": merge(soft_metrics), "selection": merge(selection)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=PREREGISTERED_STEPS)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--train-depth", type=int, default=12)
    p.add_argument("--seed", type=int, default=20261191)
    p.add_argument("--eval-seed", type=int, default=20261271)
    p.add_argument("--eval-n", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--output", type=Path, default=Path("casm-x20s-output/results.json"))
    args = p.parse_args()

    if args.steps != PREREGISTERED_STEPS or args.batch_size != 128 or args.train_depth != 12 or args.eval_n != 256:
        raise ValueError("CASM-X20S frozen run contract mismatch")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    models = cloned_x20s_models(d_model=96)
    optimizers = {
        mode: torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=WEIGHT_DECAY)
        for mode, model in models.items()
    }
    learned_counts = {m: models[m].parameter_count() for m in X20S_LEARNED_MODES}
    learned_trainable = {m: models[m].trainable_parameter_count() for m in X20S_LEARNED_MODES}
    assert len(set(learned_counts.values())) == 1
    assert len(set(learned_trainable.values())) == 1

    initial_gate_stats_train_only = {}
    for mode in X20S_LEARNED_MODES:
        initial_gate_stats_train_only[mode] = {}
        for nlive in TRAIN_LIVE_CARDINALITIES:
            b = make_state_instantiation_batch(32, args.train_depth, args.seed + nlive, live_cardinality=nlive, split="train")
            initial_gate_stats_train_only[mode][str(nlive)] = _selection_metrics(models[mode].soft_gates(b), b.live_mask)

    cardinality_counts = {str(n): 0 for n in TRAIN_LIVE_CARDINALITIES}
    minimum_answer_loss = {m: float("inf") for m in X20S_MODES}
    final_answer_loss = {m: float("nan") for m in X20S_MODES}
    maximum_post4000_grad_norm = {m: 0.0 for m in X20S_MODES}
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
        for mode in X20S_MODES:
            models[mode].train()
            row[mode] = train_step(models[mode], optimizers[mode], batch, mode=mode, step=step, lr=lr)
            minimum_answer_loss[mode] = min(minimum_answer_loss[mode], row[mode]["answer_loss"])
            final_answer_loss[mode] = row[mode]["answer_loss"]
            if step > 4000:
                maximum_post4000_grad_norm[mode] = max(maximum_post4000_grad_norm[mode], row[mode]["grad_norm"])
        if step == 1 or step % 500 == 0 or step == args.steps:
            if step % 1000 == 0 or step in (1, args.steps):
                row["raw_gate_diagnostics"] = {
                    mode: _selection_metrics(models[mode].soft_gates(batch), batch.live_mask)
                    for mode in X20S_LEARNED_MODES
                }
                row["training_forward_record_fraction"] = {
                    mode: float(models[mode].training_gates(batch).detach().mean())
                    for mode in X20S_LEARNED_MODES
                }
                row["schedule_lambda"] = {mode: storage_lambda(mode, step) for mode in X20S_MODES}
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
            for mode in X20S_MODES:
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
        "question": "does delayed storage onset repair the X20R all-absent hard-forward optimum?",
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
        "parameters": {m: models[m].parameter_count() for m in X20S_MODES},
        "trainable_parameters": {m: models[m].trainable_parameter_count() for m in X20S_MODES},
        "minimum_answer_loss": minimum_answer_loss,
        "final_answer_loss": final_answer_loss,
        "maximum_post4000_grad_norm": maximum_post4000_grad_norm,
        "optimizer_contract": {
            "optimizer": "AdamW",
            "weight_decay": WEIGHT_DECAY,
            "global_gradient_clip_norm": GRAD_CLIP_NORM,
            "lr_max": LR_MAX,
            "lr_min": LR_MIN,
            "steps": PREREGISTERED_STEPS,
        },
        "schedule_contract": {
            "storage_lambda_final": STORAGE_LAMBDA_FINAL,
            "warmup_steps": WARMUP_STEPS,
            "ramp_end_step": RAMP_END_STEP,
            "immediate": {str(s): storage_lambda(IMMEDIATE_MODE, s) for s in (1, 500, 1000, 1001, 1500, 2000, 12000)},
            "no_storage": {str(s): storage_lambda(NO_STORAGE_MODE, s) for s in (1, 500, 1000, 1001, 1500, 2000, 12000)},
            "delayed_abrupt": {str(s): storage_lambda(DELAYED_ABRUPT_MODE, s) for s in (1, 500, 1000, 1001, 1500, 2000, 12000)},
            "delayed_ramp": {str(s): storage_lambda(DELAYED_RAMP_MODE, s) for s in (1, 500, 1000, 1001, 1500, 2000, 12000)},
        },
        "constructor_contract": {
            "learned_live_mask_input": False,
            "learned_active_cardinality_input": False,
            "per_candidate_parameter_table": False,
            "hard_gate_threshold": 0.5,
            "straight_through_forward_binary": True,
            "same_estimator_as_x20r": True,
            "graph_conditioned_modes": list(X20S_GRAPH_MODES),
            "structure_blind_ablation": DELAYED_RAMP_BLIND_MODE,
        },
        "supervision_contract": {
            "final_answer_only": True,
            "live_mask_loss": False,
            "cardinality_loss": False,
            "intermediate_state_targets": False,
            "hidden_register_targets": False,
            "semantic_operator_labels": False,
            "pretraining_unseen_live_cardinality": False,
        },
        "initial_gate_stats_train_cardinalities_only": initial_gate_stats_train_only,
        "training_history": history,
        "evaluation": evaluation,
    }
    for values in (minimum_answer_loss, final_answer_loss, maximum_post4000_grad_norm):
        if not all(math.isfinite(v) and v >= 0.0 for v in values.values()):
            raise RuntimeError("non-finite report scalar")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("RESULT_JSON")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
