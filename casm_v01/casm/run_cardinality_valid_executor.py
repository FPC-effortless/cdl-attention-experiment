from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch

from .cardinality_valid_executor import LocalEquivariantTransitionModel, x9_absolute_slot_control
from .explicit_compute import aggregate_metrics, state_metrics
from .run_fixed_answer_register import answer_hidden_metrics
from .variable_contextual_data import (
    MAX_CARDINALITY,
    MIN_CARDINALITY,
    TRAIN_CARDINALITIES,
    UNSEEN_CARDINALITIES,
    make_variable_contextual_batch,
    training_cardinality_for_step,
)

HARNESS_VERSION = "casm-x9r-cardinality-valid-executor-v0-2026-08-31"
REGIMES = ("x9_absolute_slot_control", "local_equivariant_control")


def train_step(model, optimizer, batch):
    optimizer.zero_grad(set_to_none=True)
    loss = model.fixed_answer_loss(batch)
    value = float(loss.detach())
    if not math.isfinite(value) or value < 0.0:
        raise RuntimeError(f"invalid categorical loss: {value}")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {"loss": value, "grad_norm": float(grad_norm)}


def evaluate(model, *, num_registers: int, depth: int, split: str, n: int, batch_size: int, seed: int):
    rows = []
    remaining = n
    offset = 0
    while remaining:
        size = min(batch_size, remaining)
        batch = make_variable_contextual_batch(
            size,
            depth,
            seed + offset * 1009,
            num_registers=num_registers,
            split=split,
        )
        pred = model.rollout_hard(batch)
        row = state_metrics(pred, batch.target_states)
        row.update(answer_hidden_metrics(pred, batch.target_states))
        rows.append(row)
        remaining -= size
        offset += 1
    return aggregate_metrics(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--train-depth", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=20261031)
    p.add_argument("--eval-seed", type=int, default=20261111)
    p.add_argument("--eval-n", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--output", type=Path, default=Path("casm-x9r-output/results.json"))
    args = p.parse_args()

    if args.steps != 10000:
        raise ValueError("CASM-X9R preregisters exactly 10000 optimization steps")
    if args.train_depth != 8:
        raise ValueError("CASM-X9R preregisters train depth 8")
    if args.eval_n != 256:
        raise ValueError("CASM-X9R preregisters eval_n=256 per cardinality/suite")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    models = {
        "x9_absolute_slot_control": x9_absolute_slot_control(d_model=96),
        "local_equivariant_control": LocalEquivariantTransitionModel(d_model=96),
    }
    optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for name, model in models.items()
    }
    parameters = {name: model.parameter_count() for name, model in models.items()}
    trainable_parameters = {name: model.trainable_parameter_count() for name, model in models.items()}
    minimum_training_loss = {name: float("inf") for name in REGIMES}
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
        row = {"step": step, "num_registers": num_registers}
        for name in REGIMES:
            models[name].train()
            row[name] = train_step(models[name], optimizers[name], batch)
            minimum_training_loss[name] = min(minimum_training_loss[name], row[name]["loss"])
        if step == 1 or step % 100 == 0 or step == args.steps:
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
    for num_registers in range(MIN_CARDINALITY, MAX_CARDINALITY + 1):
        nkey = str(num_registers)
        evaluation[nkey] = {}
        for suite_index, (suite, split, depth) in enumerate(suites):
            eval_seed = args.eval_seed + num_registers * 1_000_003 + suite_index * 100_003
            evaluation[nkey][suite] = {"split": split, "depth": depth}
            for name in REGIMES:
                evaluation[nkey][suite][name] = evaluate(
                    models[name],
                    num_registers=num_registers,
                    depth=depth,
                    split=split,
                    n=args.eval_n,
                    batch_size=args.eval_batch_size,
                    seed=eval_seed,
                )
            print(
                f"n={num_registers} {suite}",
                json.dumps(evaluation[nkey][suite], sort_keys=True),
                flush=True,
            )

    report = {
        "harness_version": HARNESS_VERSION,
        "question": "can a slot-identity-invariant local executor validate unseen-cardinality execution after training only on n=2/3/4?",
        "seed": args.seed,
        "eval_seed": args.eval_seed,
        "steps": args.steps,
        "train_depth": args.train_depth,
        "eval_n": args.eval_n,
        "training_cardinalities": list(TRAIN_CARDINALITIES),
        "unseen_cardinalities": list(UNSEEN_CARDINALITIES),
        "training_cardinality_schedule": "2,3,4 repeated deterministically by optimizer step",
        "training_cardinality_counts": cardinality_counts,
        "parameters": parameters,
        "trainable_parameters": trainable_parameters,
        "minimum_training_loss": minimum_training_loss,
        "supervision_contract": {
            "teacher_forcing": False,
            "semantic_operator_labels": False,
            "intermediate_state_targets": False,
            "fixed_answer_register": 0,
            "hidden_final_targets": False,
            "binding_labels": False,
            "binding": "deterministic canonical e->slot e",
            "local_equivariant_absolute_slot_embedding": False,
            "local_equivariant_flattened_workspace_transition_input": False,
            "local_equivariant_cardinality_feature": False,
            "local_equivariant_external_id_feature": False,
            "local_equivariant_transition_inputs": ["opaque_command", "value_a", "value_b", "value_dst"],
        },
        "training_history": history,
        "evaluation": evaluation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("RESULT_JSON")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
