from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch

from .run_resource_competitive_binding import evaluate
from .run_stable_cardinality_executor import (
    GRAD_CLIP_NORM,
    LR_MAX,
    LR_MIN,
    PREREGISTERED_STEPS,
    WEIGHT_DECAY,
    cosine_decay_lr,
    set_optimizer_lr,
)
from .scarcity_binding import (
    CAPACITY_LAMBDA,
    LEARNED_X12_MODES,
    SCARCITY_MODES,
    SPREAD_LAMBDA,
    X12_MODES,
    cloned_x12_models,
)
from .variable_contextual_data import (
    MAX_CARDINALITY,
    MIN_CARDINALITY,
    TRAIN_CARDINALITIES,
    UNSEEN_CARDINALITIES,
    make_variable_contextual_batch,
    training_cardinality_for_step,
)

HARNESS_VERSION = "casm-x12-scarcity-binding-v0-2026-09-01"


def train_step(model, optimizer, batch, *, applied_lr: float):
    set_optimizer_lr(optimizer, applied_lr)
    optimizer.zero_grad(set_to_none=True)
    parts = model.loss_components(batch)
    values = {k: float(v.detach()) for k, v in parts.items()}
    if not math.isfinite(values["answer_loss"]) or values["answer_loss"] < 0.0:
        raise RuntimeError(f"invalid categorical answer loss: {values['answer_loss']}")
    for key in ("overlap_penalty", "spread_penalty", "capacity_penalty", "total_loss"):
        if not math.isfinite(values[key]) or values[key] < 0.0:
            raise RuntimeError(f"invalid {key}: {values[key]}")
    parts["total_loss"].backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
    optimizer.step()
    values["grad_norm"] = float(grad_norm)
    values["lr"] = float(applied_lr)
    return values


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=PREREGISTERED_STEPS)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--train-depth", type=int, default=8)
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--seed", type=int, default=20261071)
    p.add_argument("--eval-seed", type=int, default=20261151)
    p.add_argument("--eval-n", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--output", type=Path, default=Path("casm-x12-output/results.json"))
    args = p.parse_args()

    if args.steps != PREREGISTERED_STEPS:
        raise ValueError("CASM-X12 preregisters exactly 10000 optimization steps")
    if args.batch_size != 128 or args.train_depth != 8:
        raise ValueError("CASM-X12 preregisters batch size 128 and train depth 8")
    if args.weight_decay != WEIGHT_DECAY or args.eval_n != 256:
        raise ValueError("CASM-X12 frozen weight decay/eval_n mismatch")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    models = cloned_x12_models(d_model=96)
    optimizers = {
        mode: torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=args.weight_decay)
        for mode, model in models.items()
    }

    parameters = {mode: model.parameter_count() for mode, model in models.items()}
    trainable_parameters = {mode: model.trainable_parameter_count() for mode, model in models.items()}
    for left, right in (
        ("relational_independent_overlap", "relational_independent_scarcity"),
        ("relational_coordinated_overlap", "relational_coordinated_scarcity"),
        ("relational_independent_scarcity", "relational_coordinated_scarcity"),
    ):
        assert parameters[left] == parameters[right]
        assert trainable_parameters[left] == trainable_parameters[right]

    initial_bindings = {
        mode: {str(n): models[mode].binding_stats(n) for n in range(MIN_CARDINALITY, MAX_CARDINALITY + 1)}
        for mode in LEARNED_X12_MODES
    }

    minimum_answer_loss = {mode: float("inf") for mode in X12_MODES}
    minimum_total_loss = {mode: float("inf") for mode in X12_MODES}
    final_answer_loss = {mode: float("nan") for mode in X12_MODES}
    final_total_loss = {mode: float("nan") for mode in X12_MODES}
    maximum_post4000_grad_norm = {mode: 0.0 for mode in X12_MODES}
    cardinality_counts = {str(n): 0 for n in TRAIN_CARDINALITIES}
    history = []

    for step in range(1, args.steps + 1):
        num_registers = training_cardinality_for_step(step)
        cardinality_counts[str(num_registers)] += 1
        batch = make_variable_contextual_batch(
            args.batch_size,
            args.train_depth,
            args.seed * 1_000_003 + step * 97,
            num_registers=num_registers,
            split="train",
        )
        applied_lr = cosine_decay_lr(step)
        row = {"step": step, "num_registers": num_registers}
        for mode in X12_MODES:
            models[mode].train()
            row[mode] = train_step(models[mode], optimizers[mode], batch, applied_lr=applied_lr)
            minimum_answer_loss[mode] = min(minimum_answer_loss[mode], row[mode]["answer_loss"])
            minimum_total_loss[mode] = min(minimum_total_loss[mode], row[mode]["total_loss"])
            final_answer_loss[mode] = row[mode]["answer_loss"]
            final_total_loss[mode] = row[mode]["total_loss"]
            if step > 4000:
                maximum_post4000_grad_norm[mode] = max(maximum_post4000_grad_norm[mode], row[mode]["grad_norm"])
        if step == 1 or step % 100 == 0 or step == args.steps:
            if step == 1 or step % 500 == 0 or step == args.steps:
                row["binding_diagnostics"] = {
                    mode: models[mode].binding_stats(num_registers)
                    for mode in LEARNED_X12_MODES
                }
            history.append(row)
            print(json.dumps(row), flush=True)

    for model in models.values():
        model.eval()

    final_bindings = {
        mode: {str(n): models[mode].binding_stats(n) for n in range(MIN_CARDINALITY, MAX_CARDINALITY + 1)}
        for mode in LEARNED_X12_MODES
    }

    suites = [
        ("iid_depth_8", "iid", 8),
        ("composition_depth_12", "composition", 12),
        ("composition_depth_24", "composition", 24),
        ("stress_depth_48", "composition", 48),
        ("stress_depth_96", "composition", 96),
    ]
    evaluation = {}
    soft_evaluation = {}
    for num_registers in range(MIN_CARDINALITY, MAX_CARDINALITY + 1):
        nkey = str(num_registers)
        evaluation[nkey] = {}
        soft_evaluation[nkey] = {}
        for suite_index, (suite, split, depth) in enumerate(suites):
            eval_seed = args.eval_seed + num_registers * 1_000_003 + suite_index * 100_003
            evaluation[nkey][suite] = {"split": split, "depth": depth}
            soft_evaluation[nkey][suite] = {"split": split, "depth": depth}
            for mode in X12_MODES:
                evaluation[nkey][suite][mode] = evaluate(
                    models[mode], num_registers=num_registers, depth=depth, split=split,
                    n=args.eval_n, batch_size=args.eval_batch_size, seed=eval_seed,
                    discrete_binding=True,
                )
                soft_evaluation[nkey][suite][mode] = evaluate(
                    models[mode], num_registers=num_registers, depth=depth, split=split,
                    n=args.eval_n, batch_size=args.eval_batch_size, seed=eval_seed,
                    discrete_binding=False,
                )
            print(f"n={num_registers} {suite}", json.dumps(evaluation[nkey][suite], sort_keys=True), flush=True)

    report = {
        "harness_version": HARNESS_VERSION,
        "question": "does decomposed row-spread plus slot-capacity scarcity robustly solve relational allocation and unseen cardinality?",
        "seed": args.seed,
        "eval_seed": args.eval_seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "train_depth": args.train_depth,
        "eval_n": args.eval_n,
        "training_cardinalities": list(TRAIN_CARDINALITIES),
        "unseen_cardinalities": list(UNSEEN_CARDINALITIES),
        "training_cardinality_schedule": "2,3,4 repeated deterministically by optimizer step",
        "training_cardinality_counts": cardinality_counts,
        "parameters": parameters,
        "trainable_parameters": trainable_parameters,
        "minimum_answer_loss": minimum_answer_loss,
        "minimum_total_loss": minimum_total_loss,
        "final_answer_loss": final_answer_loss,
        "final_total_loss": final_total_loss,
        "maximum_post4000_grad_norm": maximum_post4000_grad_norm,
        "optimizer_contract": {
            "optimizer": "AdamW", "weight_decay": WEIGHT_DECAY,
            "global_gradient_clip_norm": GRAD_CLIP_NORM,
            "lr_max": LR_MAX, "lr_min": LR_MIN,
            "total_steps": PREREGISTERED_STEPS, "warmup_steps": 0,
            "schedule": "X9R2 cosine decay",
        },
        "scarcity_contract": {
            "spread_lambda": SPREAD_LAMBDA,
            "capacity_lambda": CAPACITY_LAMBDA,
            "spread_formula": "mean row entropy / log(8)",
            "capacity_formula": "sum_s relu(column_occupancy_s - 1)^2 / n",
            "scarcity_modes": list(SCARCITY_MODES),
            "hard_injectivity": False, "matching": False, "sinkhorn": False,
            "collision_repair": False, "binding_labels": False,
        },
        "supervision_contract": {
            "teacher_forcing": False, "semantic_operator_labels": False,
            "intermediate_state_targets": False, "fixed_answer_register": 0,
            "hidden_final_targets": False, "binding_labels": False,
            "active_cardinality_is_supplied": True,
            "validated_local_equivariant_executor": True,
            "hard_binding_evaluation": "independent row argmax without collision repair",
        },
        "initial_bindings": initial_bindings,
        "final_bindings": final_bindings,
        "training_history": history,
        "evaluation": evaluation,
        "soft_evaluation": soft_evaluation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("RESULT_JSON")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
