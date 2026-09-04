from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch

from .noncontractive_role_dynamics import (
    ADDRESS_BETA,
    ALPHA,
    DIAGNOSTIC_ROLE_COUNT,
    LEARNED_X19D_MODES,
    ORTHOGONAL_X19D_MODES,
    RAW_MATRIX_INIT_STD,
    RECURRENT_X19D_MODES,
    ROLE_DIM,
    X19D_MODES,
    cloned_x19d_models,
)
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
from .variable_contextual_data import (
    MAX_CARDINALITY,
    MIN_CARDINALITY,
    TRAIN_CARDINALITIES,
    UNSEEN_CARDINALITIES,
    make_variable_contextual_batch,
    training_cardinality_for_step,
)

HARNESS_VERSION = "casm-x19d-noncontractive-role-dynamics-v0-2026-09-04"


def orthogonality_error(model) -> float:
    q = model.transition_matrix().detach()
    eye = torch.eye(ROLE_DIM, device=q.device, dtype=q.dtype)
    return float((q.T @ q - eye).abs().max())


def train_step(model, optimizer, batch, *, applied_lr: float):
    set_optimizer_lr(optimizer, applied_lr)
    optimizer.zero_grad(set_to_none=True)
    parts = model.loss_components(batch)
    answer = float(parts["answer_loss"].detach())
    total = float(parts["total_loss"].detach())
    if not math.isfinite(answer) or answer < 0.0:
        raise RuntimeError(f"invalid answer loss: {answer}")
    if not math.isfinite(total) or total < 0.0:
        raise RuntimeError(f"invalid total loss: {total}")

    n = batch.initial.shape[1]
    # During optimization this is constrained by the frozen 2,3,4 schedule, so no r4+ exists yet.
    roles = model.roles(n)
    address = model.address_matrix(n, discrete=False)
    if not torch.isfinite(roles).all() or not torch.isfinite(address).all():
        raise RuntimeError("non-finite role/address state")
    if not torch.allclose(address.sum(dim=-1), torch.ones(n, device=address.device), atol=1e-5, rtol=0.0):
        raise RuntimeError("invalid role-address normalization")

    parts["total_loss"].backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
    optimizer.step()
    return {
        "answer_loss": answer,
        "total_loss": total,
        "grad_norm": float(grad_norm),
        "lr": float(applied_lr),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=PREREGISTERED_STEPS)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--train-depth", type=int, default=8)
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--seed", type=int, default=20261161)
    p.add_argument("--eval-seed", type=int, default=20261241)
    p.add_argument("--eval-n", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--output", type=Path, default=Path("casm-x19d-output/results.json"))
    args = p.parse_args()

    if args.steps != PREREGISTERED_STEPS:
        raise ValueError("CASM-X19D preregisters exactly 10000 optimization steps")
    if args.batch_size != 128 or args.train_depth != 8:
        raise ValueError("CASM-X19D preregisters batch size 128 and train depth 8")
    if args.weight_decay != WEIGHT_DECAY or args.eval_n != 256:
        raise ValueError("CASM-X19D frozen weight decay/eval_n mismatch")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    models = cloned_x19d_models(d_model=96)
    optimizers = {
        mode: torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=args.weight_decay)
        for mode, model in models.items()
    }

    parameters = {mode: model.parameter_count() for mode, model in models.items()}
    trainable_parameters = {mode: model.trainable_parameter_count() for mode, model in models.items()}
    assert parameters["unconstrained_recursive"] == parameters["orthogonal_recursive"]
    assert trainable_parameters["unconstrained_recursive"] == trainable_parameters["orthogonal_recursive"]

    frozen = models["frozen_random_orthogonal"]
    unconstrained = models["unconstrained_recursive"]
    orthogonal = models["orthogonal_recursive"]
    assert torch.equal(frozen.constructor_seed, unconstrained.constructor_seed)
    assert torch.equal(frozen.constructor_seed, orthogonal.constructor_seed)
    assert torch.equal(frozen.raw_matrix, unconstrained.raw_matrix)
    assert torch.equal(frozen.raw_matrix, orthogonal.raw_matrix)
    frozen_seed_initial = frozen.constructor_seed.detach().clone()
    frozen_matrix_initial = frozen.raw_matrix.detach().clone()

    initial_orthogonality_error = {
        mode: orthogonality_error(models[mode]) for mode in ORTHOGONAL_X19D_MODES
    }
    if any(v > 1e-5 for v in initial_orthogonality_error.values()):
        raise RuntimeError(f"orthogonality initialization failure: {initial_orthogonality_error}")

    # No role beyond r3 may be constructed before training finishes.
    initial_address_train_cardinalities_only = {
        mode: {str(n): models[mode].address_stats(n) for n in TRAIN_CARDINALITIES}
        for mode in X19D_MODES
    }

    minimum_answer_loss = {mode: float("inf") for mode in X19D_MODES}
    final_answer_loss = {mode: float("nan") for mode in X19D_MODES}
    maximum_post4000_grad_norm = {mode: 0.0 for mode in X19D_MODES}
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
        for mode in X19D_MODES:
            models[mode].train()
            row[mode] = train_step(models[mode], optimizers[mode], batch, applied_lr=applied_lr)
            minimum_answer_loss[mode] = min(minimum_answer_loss[mode], row[mode]["answer_loss"])
            final_answer_loss[mode] = row[mode]["answer_loss"]
            if step > 4000:
                maximum_post4000_grad_norm[mode] = max(
                    maximum_post4000_grad_norm[mode], row[mode]["grad_norm"]
                )
        if step == 1 or step % 100 == 0 or step == args.steps:
            if step == 1 or step % 500 == 0 or step == args.steps:
                row["address_diagnostics"] = {
                    mode: models[mode].address_stats(num_registers) for mode in X19D_MODES
                }
                row["orthogonality_error"] = {
                    mode: orthogonality_error(models[mode]) for mode in ORTHOGONAL_X19D_MODES
                }
            history.append(row)
            print(json.dumps(row), flush=True)

    if not torch.equal(frozen.constructor_seed, frozen_seed_initial):
        raise RuntimeError("frozen random constructor seed changed during optimization")
    if not torch.equal(frozen.raw_matrix, frozen_matrix_initial):
        raise RuntimeError("frozen random recurrence matrix changed during optimization")

    final_orthogonality_error = {
        mode: orthogonality_error(models[mode]) for mode in ORTHOGONAL_X19D_MODES
    }
    if any(v > 1e-5 for v in final_orthogonality_error.values()):
        raise RuntimeError(f"post-training orthogonality failure: {final_orthogonality_error}")

    for model in models.values():
        model.eval()

    # First generation/inspection of r4+ and n5/n6 occurs only after all optimization above.
    final_address = {
        mode: {
            str(n): models[mode].address_stats(n)
            for n in range(MIN_CARDINALITY, MAX_CARDINALITY + 1)
        }
        for mode in X19D_MODES
    }
    constructor_diagnostics = {
        mode: models[mode].constructor_diagnostics(
            perturb_seed=args.eval_seed + 9_000_000,
            count=DIAGNOSTIC_ROLE_COUNT,
        )
        for mode in RECURRENT_X19D_MODES
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
            for mode in X19D_MODES:
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
        "question": "does noncontractive recursive role dynamics preserve executable identity beyond the trained role horizon?",
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
        "final_answer_loss": final_answer_loss,
        "maximum_post4000_grad_norm": maximum_post4000_grad_norm,
        "optimizer_contract": {
            "optimizer": "AdamW",
            "weight_decay": WEIGHT_DECAY,
            "global_gradient_clip_norm": GRAD_CLIP_NORM,
            "lr_max": LR_MAX,
            "lr_min": LR_MIN,
            "total_steps": PREREGISTERED_STEPS,
            "warmup_steps": 0,
            "schedule": "X9R2 cosine decay",
        },
        "constructor_contract": {
            "role_dim": ROLE_DIM,
            "alpha": ALPHA,
            "address_beta": ADDRESS_BETA,
            "raw_matrix_init_std": RAW_MATRIX_INIT_STD,
            "diagnostic_role_count": DIAGNOSTIC_ROLE_COUNT,
            "unconstrained": "normalize((I + alpha*A) @ r)",
            "orthogonal": "Cayley(A-A^T) @ r",
            "learned_pair_parameter_matched": True,
            "frozen_random_constructor_trainable": False,
            "external_index_feature": False,
            "active_cardinality_feature": False,
            "step_embedding": False,
        },
        "memory_contract": {
            "one_transient_record_per_supplied_active_variable": True,
            "learned_record_creation_decision": False,
            "physical_fixed_slot_bank": False,
            "learned_role_to_slot_scorer": False,
            "read_write_decode_through_role_address_only": True,
            "soft_address": "softmax(16*cosine(query_role,key_role))",
            "hard_address": "raw argmax without repair",
            "direct_command_index_record_access": False,
            "dual_prices": False,
            "occupancy_state": False,
            "matching": False,
            "collision_repair": False,
        },
        "supervision_contract": {
            "teacher_forcing": False,
            "semantic_operator_labels": False,
            "intermediate_state_targets": False,
            "fixed_answer_register": 0,
            "hidden_final_targets": False,
            "role_labels": False,
            "address_labels": False,
            "role_separation_loss": False,
            "active_cardinality_is_supplied": True,
            "commands_are_supplied": True,
            "pretraining_unseen_role_generation": False,
        },
        "initial_orthogonality_error": initial_orthogonality_error,
        "final_orthogonality_error": final_orthogonality_error,
        "initial_address_train_cardinalities_only": initial_address_train_cardinalities_only,
        "final_address": final_address,
        "constructor_diagnostics_posttraining_only": constructor_diagnostics,
        "training_history": history,
        "evaluation": evaluation,
        "soft_evaluation": soft_evaluation,
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
