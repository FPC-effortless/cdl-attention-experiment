from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch

from .contextual_data import make_contextual_batch
from .explicit_compute import aggregate_metrics, state_metrics
from .run_fixed_answer_register import answer_hidden_metrics
from .unconstrained_binding import (
    BINDING_MODES,
    UnconstrainedBindingTransitionModel,
    cloned_topology_models,
)

HARNESS_VERSION = "casm-x8-unconstrained-binding-v1-2026-08-31"


def train_step(model, optimizer, batch):
    optimizer.zero_grad(set_to_none=True)
    loss = model.fixed_answer_loss(batch)
    loss_value = float(loss.detach())
    if not math.isfinite(loss_value) or loss_value < 0.0:
        raise RuntimeError(f"invalid categorical loss: {loss_value}")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {"loss": loss_value, "grad_norm": float(grad_norm)}


def evaluate(
    model,
    *,
    depth: int,
    split: str,
    n: int,
    batch_size: int,
    seed: int,
    discrete_binding: bool,
):
    rows = []
    remaining = n
    offset = 0
    while remaining:
        size = min(batch_size, remaining)
        batch = make_contextual_batch(size, depth, seed + offset * 1009, split=split)
        pred = model.rollout_hard(batch, discrete_binding=discrete_binding)
        row = state_metrics(pred, batch.target_states)
        row.update(answer_hidden_metrics(pred, batch.target_states))
        rows.append(row)
        remaining -= size
        offset += 1
    return aggregate_metrics(rows)


def _compact_stats(stats: dict[str, object]) -> dict[str, object]:
    return {
        "row_max_mean": stats["row_max_mean"],
        "row_entropy_mean": stats["row_entropy_mean"],
        "independent_argmax_assignment": stats["independent_argmax_assignment"],
        "independent_argmax_unique_slot_count": stats["independent_argmax_unique_slot_count"],
        "independent_argmax_collision_count": stats["independent_argmax_collision_count"],
        "best_injective_assignment": stats["best_injective_assignment"],
        "best_injective_score": stats["best_injective_score"],
        "max_row_sum_error": stats["max_row_sum_error"],
        "max_column_occupancy": stats["max_column_occupancy"],
        "min_column_occupancy": stats["min_column_occupancy"],
        "total_assignment_mass": stats["total_assignment_mass"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--train-depth", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--binding-temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=20261011)
    p.add_argument("--eval-seed", type=int, default=20261091)
    p.add_argument("--eval-n", type=int, default=384)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--output", type=Path, default=Path("casm-x8-output/results.json"))
    args = p.parse_args()

    if args.steps != 8000:
        raise ValueError("CASM-X8 preregisters exactly 8000 optimization steps")
    if args.train_depth != 8:
        raise ValueError("CASM-X8 preregisters fixed training depth 8")
    if args.binding_temperature != 1.0:
        raise ValueError("CASM-X8 preregisters binding temperature 1.0")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    seed_model = UnconstrainedBindingTransitionModel(
        d_model=96,
        binding_mode="learned_dense",
        binding_temperature=args.binding_temperature,
    )
    models = cloned_topology_models(seed_model)
    optimizers = {
        mode: torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for mode, model in models.items()
    }

    parameters = {mode: model.parameter_count() for mode, model in models.items()}
    trainable_parameters = {
        mode: model.trainable_parameter_count() for mode, model in models.items()
    }
    assert len(set(parameters.values())) == 1, parameters
    assert len(set(trainable_parameters.values())) == 1, trainable_parameters

    initial_dense = models["learned_dense"].binding_stats()
    initial_injective = models["learned_injective"].binding_stats()
    history = []
    minimum_training_loss = {mode: float("inf") for mode in BINDING_MODES}

    for step in range(1, args.steps + 1):
        batch = make_contextual_batch(
            args.batch_size,
            args.train_depth,
            args.seed * 1_000_003 + step * 97,
            split="train",
        )
        row = {"step": step}
        for mode in BINDING_MODES:
            models[mode].train()
            row[mode] = train_step(models[mode], optimizers[mode], batch)
            minimum_training_loss[mode] = min(
                minimum_training_loss[mode],
                row[mode]["loss"],
            )

        if step == 1 or step % 100 == 0 or step == args.steps:
            row["learned_dense_stats"] = _compact_stats(
                models["learned_dense"].binding_stats()
            )
            row["learned_injective_stats"] = _compact_stats(
                models["learned_injective"].binding_stats()
            )
            history.append(row)
            print(json.dumps(row), flush=True)

    for model in models.values():
        model.eval()

    suites = [
        ("iid_depth_8", "iid", 8),
        ("composition_depth_12", "composition", 12),
        ("composition_depth_24", "composition", 24),
        ("stress_depth_48", "composition", 48),
        ("stress_depth_96", "composition", 96),
    ]

    evaluation = {}
    soft_evaluation = {}
    for i, (suite, split, depth) in enumerate(suites):
        evaluation[suite] = {"split": split, "depth": depth}
        soft_evaluation[suite] = {"split": split, "depth": depth}
        for mode in BINDING_MODES:
            evaluation[suite][mode] = evaluate(
                models[mode],
                depth=depth,
                split=split,
                n=args.eval_n,
                batch_size=args.eval_batch_size,
                seed=args.eval_seed + i * 100_003,
                discrete_binding=True,
            )
            soft_evaluation[suite][mode] = evaluate(
                models[mode],
                depth=depth,
                split=split,
                n=args.eval_n,
                batch_size=args.eval_batch_size,
                seed=args.eval_seed + i * 100_003,
                discrete_binding=False,
            )
        print("HARD", suite, json.dumps(evaluation[suite], sort_keys=True), flush=True)
        print("SOFT", suite, json.dumps(soft_evaluation[suite], sort_keys=True), flush=True)

    report = {
        "harness_version": HARNESS_VERSION,
        "question": (
            "does fixed-answer-only supervision discover collision-free discrete variable topology "
            "without an injective binding prior?"
        ),
        "seed": args.seed,
        "eval_seed": args.eval_seed,
        "steps": args.steps,
        "train_depth": args.train_depth,
        "parameters": parameters,
        "trainable_parameters": trainable_parameters,
        "binding_temperature": args.binding_temperature,
        "minimum_training_loss": minimum_training_loss,
        "supervision_contract": {
            "teacher_forcing": False,
            "semantic_operator_labels": False,
            "intermediate_state_targets": False,
            "fixed_answer_register": 0,
            "binding_labels": False,
            "binding_regularizer": False,
            "collision_penalty": False,
            "identity_prior": False,
            "external_register_count": 4,
            "candidate_internal_slot_count": 8,
            "surplus_symbol": "EMPTY",
            "learned_dense_row_normalized": True,
            "learned_dense_injective_constraint": False,
            "learned_dense_column_capacity_constraint": False,
            "learned_dense_hard_evaluation": "independent_row_argmax_without_collision_repair",
            "learned_injective_control": True,
            "all_register_specific_access_passes_through_binding": True,
        },
        "initial_binding_stats": {
            "learned_dense": initial_dense,
            "learned_injective": initial_injective,
        },
        "final_binding_stats": {
            mode: model.binding_stats() for mode, model in models.items()
        },
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
