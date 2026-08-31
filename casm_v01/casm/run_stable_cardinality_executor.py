from __future__ import annotations

import argparse
import json
import math
import random
from copy import deepcopy
from pathlib import Path

import torch

from .cardinality_valid_executor import LocalEquivariantTransitionModel
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

HARNESS_VERSION = "casm-x9r2-stable-optimizer-v0-2026-08-31"
REGIMES = ("fixed_lr_replication", "cosine_decay_stable")
PREREGISTERED_STEPS = 10000
LR_MAX = 2e-3
LR_MIN = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0


def cosine_decay_lr(
    step: int,
    *,
    total_steps: int = PREREGISTERED_STEPS,
    lr_max: float = LR_MAX,
    lr_min: float = LR_MIN,
) -> float:
    if total_steps < 2:
        raise ValueError("total_steps must be >=2")
    if not 1 <= step <= total_steps:
        raise ValueError((step, total_steps))
    if not 0.0 < lr_min <= lr_max:
        raise ValueError((lr_min, lr_max))
    phase = math.pi * float(step - 1) / float(total_steps - 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(phase))


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def cloned_local_models(d_model: int = 96) -> dict[str, LocalEquivariantTransitionModel]:
    seed_model = LocalEquivariantTransitionModel(d_model=d_model)
    state = deepcopy(seed_model.state_dict())
    out: dict[str, LocalEquivariantTransitionModel] = {}
    for name in REGIMES:
        model = LocalEquivariantTransitionModel(d_model=d_model)
        model.load_state_dict(state)
        out[name] = model
    return out


def train_step(model, optimizer, batch, *, applied_lr: float):
    set_optimizer_lr(optimizer, applied_lr)
    optimizer.zero_grad(set_to_none=True)
    loss = model.fixed_answer_loss(batch)
    value = float(loss.detach())
    if not math.isfinite(value) or value < 0.0:
        raise RuntimeError(f"invalid categorical loss: {value}")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
    optimizer.step()
    return {
        "loss": value,
        "grad_norm": float(grad_norm),
        "lr": float(applied_lr),
    }


def paired_train_step(models, optimizers, batch, *, step: int):
    return {
        "fixed_lr_replication": train_step(
            models["fixed_lr_replication"],
            optimizers["fixed_lr_replication"],
            batch,
            applied_lr=LR_MAX,
        ),
        "cosine_decay_stable": train_step(
            models["cosine_decay_stable"],
            optimizers["cosine_decay_stable"],
            batch,
            applied_lr=cosine_decay_lr(step),
        ),
    }


def evaluate(
    model,
    *,
    num_registers: int,
    depth: int,
    split: str,
    n: int,
    batch_size: int,
    seed: int,
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
        pred = model.rollout_hard(batch)
        row = state_metrics(pred, batch.target_states)
        row.update(answer_hidden_metrics(pred, batch.target_states))
        rows.append(row)
        remaining -= size
        offset += 1
    return aggregate_metrics(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=PREREGISTERED_STEPS)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--train-depth", type=int, default=8)
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--seed", type=int, default=20261041)
    p.add_argument("--eval-seed", type=int, default=20261121)
    p.add_argument("--eval-n", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--output", type=Path, default=Path("casm-x9r2-output/results.json"))
    args = p.parse_args()

    if args.steps != PREREGISTERED_STEPS:
        raise ValueError("CASM-X9R2 preregisters exactly 10000 optimization steps")
    if args.batch_size != 128:
        raise ValueError("CASM-X9R2 preregisters batch size 128")
    if args.train_depth != 8:
        raise ValueError("CASM-X9R2 preregisters train depth 8")
    if args.weight_decay != WEIGHT_DECAY:
        raise ValueError("CASM-X9R2 preregisters weight decay 1e-4")
    if args.eval_n != 256:
        raise ValueError("CASM-X9R2 preregisters eval_n=256 per cardinality/suite")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    models = cloned_local_models(d_model=96)
    first_state = models[REGIMES[0]].state_dict()
    second_state = models[REGIMES[1]].state_dict()
    assert first_state.keys() == second_state.keys()
    assert all(torch.equal(first_state[k], second_state[k]) for k in first_state)

    optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=args.weight_decay)
        for name, model in models.items()
    }
    parameters = {name: model.parameter_count() for name, model in models.items()}
    trainable_parameters = {name: model.trainable_parameter_count() for name, model in models.items()}
    assert len(set(parameters.values())) == 1, parameters
    assert len(set(trainable_parameters.values())) == 1, trainable_parameters

    minimum_training_loss = {name: float("inf") for name in REGIMES}
    final_training_loss = {name: float("nan") for name in REGIMES}
    maximum_post4000_grad_norm = {name: 0.0 for name in REGIMES}
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
        paired = paired_train_step(models, optimizers, batch, step=step)
        row = {"step": step, "num_registers": num_registers, **paired}
        for name in REGIMES:
            loss_value = paired[name]["loss"]
            grad_value = paired[name]["grad_norm"]
            minimum_training_loss[name] = min(minimum_training_loss[name], loss_value)
            final_training_loss[name] = loss_value
            if step > 4000:
                maximum_post4000_grad_norm[name] = max(
                    maximum_post4000_grad_norm[name], grad_value
                )
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
        "question": "does cosine learning-rate decay make the X9R local-equivariant executor robustly cardinality-valid on new seeds?",
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
        "minimum_training_loss": minimum_training_loss,
        "final_training_loss": final_training_loss,
        "maximum_post4000_grad_norm": maximum_post4000_grad_norm,
        "optimizer_contract": {
            "optimizer": "AdamW",
            "weight_decay": WEIGHT_DECAY,
            "global_gradient_clip_norm": GRAD_CLIP_NORM,
            "fixed_lr": LR_MAX,
            "cosine_lr_max": LR_MAX,
            "cosine_lr_min": LR_MIN,
            "cosine_total_steps": PREREGISTERED_STEPS,
            "cosine_warmup_steps": 0,
            "cosine_formula": "lr_min + 0.5*(lr_max-lr_min)*(1+cos(pi*(step-1)/(10000-1)))",
        },
        "supervision_contract": {
            "teacher_forcing": False,
            "semantic_operator_labels": False,
            "intermediate_state_targets": False,
            "fixed_answer_register": 0,
            "hidden_final_targets": False,
            "binding_labels": False,
            "binding": "deterministic canonical e->slot e",
            "absolute_slot_embedding": False,
            "flattened_workspace_transition_input": False,
            "cardinality_feature": False,
            "external_id_feature": False,
            "transition_inputs": ["opaque_command", "value_a", "value_b", "value_dst"],
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
