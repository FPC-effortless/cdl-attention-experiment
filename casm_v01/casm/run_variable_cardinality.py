from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch

from .explicit_compute import aggregate_metrics, state_metrics
from .run_fixed_answer_register import answer_hidden_metrics
from .variable_cardinality_binding import (
    BINDING_MODES,
    VariableCardinalityTransitionModel,
    cloned_cardinality_models,
)
from .variable_contextual_data import (
    MAX_CARDINALITY,
    MIN_CARDINALITY,
    TRAIN_CARDINALITIES,
    UNSEEN_CARDINALITIES,
    make_variable_contextual_batch,
    training_cardinality_for_step,
)

HARNESS_VERSION = "casm-x9-variable-cardinality-v0-2026-08-31"


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


def evaluate(
    model,
    *,
    num_registers: int,
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
        batch = make_variable_contextual_batch(
            size,
            depth,
            seed + offset * 1009,
            num_registers=num_registers,
            split=split,
        )
        pred = model.rollout_hard(batch, discrete_binding=discrete_binding)
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
    p.add_argument("--binding-temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=20261021)
    p.add_argument("--eval-seed", type=int, default=20261101)
    p.add_argument("--eval-n", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--output", type=Path, default=Path("casm-x9-output/results.json"))
    args = p.parse_args()

    if args.steps != 10000:
        raise ValueError("CASM-X9 preregisters exactly 10000 optimization steps")
    if args.train_depth != 8:
        raise ValueError("CASM-X9 preregisters train depth 8")
    if args.binding_temperature != 1.0:
        raise ValueError("CASM-X9 preregisters binding temperature 1.0")
    if args.eval_n != 256:
        raise ValueError("CASM-X9 preregisters eval_n=256 per cardinality/suite")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    seed_model = VariableCardinalityTransitionModel(
        d_model=96,
        binding_mode="shared_generator_dense",
        binding_temperature=args.binding_temperature,
    )
    models = cloned_cardinality_models(seed_model)
    optimizers = {
        mode: torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for mode, model in models.items()
    }
    parameters = {mode: model.parameter_count() for mode, model in models.items()}
    trainable_parameters = {mode: model.trainable_parameter_count() for mode, model in models.items()}
    assert len(set(parameters.values())) == 1, parameters
    assert len(set(trainable_parameters.values())) == 1, trainable_parameters

    initial_bindings = {
        str(n): models["shared_generator_dense"].binding_stats(n)
        for n in range(MIN_CARDINALITY, MAX_CARDINALITY + 1)
    }
    history = []
    minimum_training_loss = {mode: float("inf") for mode in BINDING_MODES}
    cardinality_counts = {str(n): 0 for n in TRAIN_CARDINALITIES}

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
        for mode in BINDING_MODES:
            models[mode].train()
            row[mode] = train_step(models[mode], optimizers[mode], batch)
            minimum_training_loss[mode] = min(minimum_training_loss[mode], row[mode]["loss"])
        if step == 1 or step % 100 == 0 or step == args.steps:
            row["shared_generator_binding"] = models["shared_generator_dense"].binding_stats(
                num_registers
            )
            history.append(row)
            print(json.dumps(row), flush=True)

    for model in models.values():
        model.eval()

    final_bindings = {
        str(n): models["shared_generator_dense"].binding_stats(n)
        for n in range(MIN_CARDINALITY, MAX_CARDINALITY + 1)
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
            for mode in BINDING_MODES:
                evaluation[nkey][suite][mode] = evaluate(
                    models[mode],
                    num_registers=num_registers,
                    depth=depth,
                    split=split,
                    n=args.eval_n,
                    batch_size=args.eval_batch_size,
                    seed=eval_seed,
                    discrete_binding=True,
                )
                soft_evaluation[nkey][suite][mode] = evaluate(
                    models[mode],
                    num_registers=num_registers,
                    depth=depth,
                    split=split,
                    n=args.eval_n,
                    batch_size=args.eval_batch_size,
                    seed=eval_seed,
                    discrete_binding=False,
                )
            print(
                f"n={num_registers} {suite}",
                json.dumps(evaluation[nkey][suite], sort_keys=True),
                flush=True,
            )

    report = {
        "harness_version": HARNESS_VERSION,
        "question": "can one shared descriptor-to-binding rule generalize hidden execution from trained cardinalities 2/3/4 to unseen 5/6?",
        "seed": args.seed,
        "eval_seed": args.eval_seed,
        "steps": args.steps,
        "train_depth": args.train_depth,
        "eval_n": args.eval_n,
        "binding_temperature": args.binding_temperature,
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
            "binding_regularizer": False,
            "collision_penalty": False,
            "identity_prior": False,
            "external_id_embedding": False,
            "free_per_variable_binding_rows": False,
            "active_cardinality_is_supplied": True,
            "external_variable_descriptors_are_deterministic": True,
            "candidate_internal_slot_count": 8,
            "surplus_symbol": "EMPTY",
            "shared_generator_dense_injective_constraint": False,
            "shared_generator_dense_hard_evaluation": "independent_row_argmax_without_collision_repair",
            "all_register_specific_access_passes_through_generated_binding": True,
        },
        "initial_shared_generator_bindings": initial_bindings,
        "final_shared_generator_bindings": final_bindings,
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
