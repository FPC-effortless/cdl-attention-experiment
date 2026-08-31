from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch

from .contextual_data import make_contextual_batch
from .explicit_compute import aggregate_metrics, state_metrics
from .learned_binding import BINDING_MODES, BoundExplicitTransitionModel, cloned_binding_models
from .run_fixed_answer_register import answer_hidden_metrics

HARNESS_VERSION = "casm-x6r-exact-birkhoff-binding-v1-2026-08-31"


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--train-depth", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--binding-temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=20260911)
    p.add_argument("--eval-seed", type=int, default=20260991)
    p.add_argument("--eval-n", type=int, default=384)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--output", type=Path, default=Path("casm-x6r-output/results.json"))
    args = p.parse_args()

    if args.train_depth != 8:
        raise ValueError("CASM-X6R preregisters fixed training depth 8")
    if args.steps != 6000:
        raise ValueError("CASM-X6R preregisters exactly 6000 optimization steps")
    if args.binding_temperature != 1.0:
        raise ValueError("CASM-X6R preregisters binding temperature 1.0")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    seed_model = BoundExplicitTransitionModel(
        d_model=96,
        binding_mode="learned_binding",
        binding_temperature=args.binding_temperature,
    )
    models = cloned_binding_models(seed_model)
    optimizers = {
        mode: torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for mode, model in models.items()
    }
    parameters = {mode: model.parameter_count() for mode, model in models.items()}
    trainable_parameters = {
        mode: model.trainable_parameter_count() for mode, model in models.items()
    }
    assert len(set(parameters.values())) == 1, parameters

    initial_binding = models["learned_binding"].binding_stats()
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
            minimum_training_loss[mode] = min(minimum_training_loss[mode], row[mode]["loss"])
        if step == 1 or step % 100 == 0 or step == args.steps:
            learned_stats = models["learned_binding"].binding_stats()
            row["learned_binding_stats"] = {
                "row_max_mean": learned_stats["row_max_mean"],
                "column_max_mean": learned_stats["column_max_mean"],
                "row_entropy_mean": learned_stats["row_entropy_mean"],
                "best_permutation_score": learned_stats["best_permutation_score"],
                "projected_permutation": learned_stats["projected_permutation"],
                "max_row_sum_error": learned_stats["max_row_sum_error"],
                "max_column_sum_error": learned_stats["max_column_sum_error"],
            }
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
    learned_soft_evaluation = {}
    for i, (suite, split, depth) in enumerate(suites):
        evaluation[suite] = {"split": split, "depth": depth}
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
        learned_soft_evaluation[suite] = evaluate(
            models["learned_binding"],
            depth=depth,
            split=split,
            n=args.eval_n,
            batch_size=args.eval_batch_size,
            seed=args.eval_seed + i * 100_003,
            discrete_binding=False,
        )
        print(suite, json.dumps(evaluation[suite], sort_keys=True), flush=True)

    binding_stats = {mode: model.binding_stats() for mode, model in models.items()}
    report = {
        "harness_version": HARNESS_VERSION,
        "question": "can answer-only supervision learn an executable external-register to internal-slot binding?",
        "validity_repair": "exact convex mixture over all 24 permutation matrices",
        "binding_normalizer": "exact_birkhoff_permutation_mixture",
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
            "identity_prior": False,
            "binding_regularizer": False,
            "all_register_specific_access_passes_through_binding": True,
            "learned_discrete_evaluation_uses_best_one_to_one_projection": True,
        },
        "initial_learned_binding": initial_binding,
        "final_binding_stats": binding_stats,
        "training_history": history,
        "evaluation": evaluation,
        "learned_soft_evaluation": learned_soft_evaluation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("RESULT_JSON")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
