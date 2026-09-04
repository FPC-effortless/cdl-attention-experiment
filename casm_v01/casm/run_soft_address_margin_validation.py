from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch

from .noncontractive_role_dynamics import (
    LEARNED_X19D_MODES,
    ORTHOGONAL_X19D_MODES,
    RECURRENT_X19D_MODES,
    X19D_MODES,
    cloned_x19d_models,
)
from .run_noncontractive_role_dynamics import orthogonality_error, train_step
from .run_resource_competitive_binding import evaluate
from .run_stable_cardinality_executor import (
    GRAD_CLIP_NORM,
    LR_MAX,
    LR_MIN,
    PREREGISTERED_STEPS,
    WEIGHT_DECAY,
    cosine_decay_lr,
)
from .soft_address_margin_validation import (
    COUNTERFACTUAL_BETA,
    TRAIN_BETA,
    X19VAddressView,
)
from .variable_contextual_data import (
    MAX_CARDINALITY,
    MIN_CARDINALITY,
    TRAIN_CARDINALITIES,
    UNSEEN_CARDINALITIES,
    make_variable_contextual_batch,
    training_cardinality_for_step,
)

HARNESS_VERSION = "casm-x19v-soft-address-margin-v0-2026-09-04"


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
    p.add_argument("--output", type=Path, default=Path("casm-x19v-output/results.json"))
    args = p.parse_args()

    if args.steps != PREREGISTERED_STEPS:
        raise ValueError("X19V replays exactly 10000 X19D optimizer steps")
    if args.batch_size != 128 or args.train_depth != 8:
        raise ValueError("X19V replays batch size 128 and train depth 8")
    if args.weight_decay != WEIGHT_DECAY or args.eval_n != 256:
        raise ValueError("X19V replay contract mismatch")
    if TRAIN_BETA != 16.0 or COUNTERFACTUAL_BETA != 64.0:
        raise RuntimeError("X19V beta contract changed")

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
        raise RuntimeError("initial orthogonality contract failed")

    # Strictly train-cardinality-only before optimization completes: no r4+, n5, or n6.
    initial_address_train_cardinalities_only = {
        mode: {str(n): models[mode].address_stats(n) for n in TRAIN_CARDINALITIES}
        for mode in X19D_MODES
    }

    minimum_answer_loss = {mode: float("inf") for mode in X19D_MODES}
    final_answer_loss = {mode: float("nan") for mode in X19D_MODES}
    maximum_post4000_grad_norm = {mode: 0.0 for mode in X19D_MODES}
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
        applied_lr = cosine_decay_lr(step)
        for mode in X19D_MODES:
            models[mode].train()
            row = train_step(models[mode], optimizers[mode], batch, applied_lr=applied_lr)
            minimum_answer_loss[mode] = min(minimum_answer_loss[mode], row["answer_loss"])
            final_answer_loss[mode] = row["answer_loss"]
            if step > 4000:
                maximum_post4000_grad_norm[mode] = max(
                    maximum_post4000_grad_norm[mode], row["grad_norm"]
                )
        if step == 1 or step % 500 == 0 or step == args.steps:
            print(json.dumps({
                "step": step,
                "num_registers": num_registers,
                "answer_loss": final_answer_loss,
            }), flush=True)

    if not torch.equal(frozen.constructor_seed, frozen_seed_initial):
        raise RuntimeError("frozen random constructor seed changed")
    if not torch.equal(frozen.raw_matrix, frozen_matrix_initial):
        raise RuntimeError("frozen random recurrence matrix changed")

    final_orthogonality_error = {
        mode: orthogonality_error(models[mode]) for mode in ORTHOGONAL_X19D_MODES
    }
    if any(v > 1e-5 for v in final_orthogonality_error.values()):
        raise RuntimeError("final orthogonality contract failed")

    for model in models.values():
        model.eval()

    # Counterfactual views are created only after all 10,000 training steps above.
    views16 = {mode: X19VAddressView(model, beta=TRAIN_BETA) for mode, model in models.items()}
    views64 = {mode: X19VAddressView(model, beta=COUNTERFACTUAL_BETA) for mode, model in models.items()}

    final_address_beta16 = {
        mode: {str(n): views16[mode].address_stats(n) for n in range(MIN_CARDINALITY, MAX_CARDINALITY + 1)}
        for mode in X19D_MODES
    }
    final_address_beta64 = {
        mode: {str(n): views64[mode].address_stats(n) for n in range(MIN_CARDINALITY, MAX_CARDINALITY + 1)}
        for mode in X19D_MODES
    }
    for mode in X19D_MODES:
        for n in range(MIN_CARDINALITY, MAX_CARDINALITY + 1):
            if final_address_beta16[mode][str(n)]["hard_address"] != final_address_beta64[mode][str(n)]["hard_address"]:
                raise RuntimeError("beta changed hard address")

    suites = [
        ("iid_depth_8", "iid", 8),
        ("composition_depth_12", "composition", 12),
        ("composition_depth_24", "composition", 24),
        ("stress_depth_48", "composition", 48),
        ("stress_depth_96", "composition", 96),
    ]
    evaluations = {"beta16_replay": {}, "beta64_counterfactual": {}}
    for label, views in (("beta16_replay", views16), ("beta64_counterfactual", views64)):
        for num_registers in range(MIN_CARDINALITY, MAX_CARDINALITY + 1):
            nkey = str(num_registers)
            evaluations[label][nkey] = {}
            for suite_index, (suite, split, depth) in enumerate(suites):
                eval_seed = args.eval_seed + num_registers * 1_000_003 + suite_index * 100_003
                evaluations[label][nkey][suite] = {"split": split, "depth": depth, "hard": {}, "soft": {}}
                for mode in X19D_MODES:
                    evaluations[label][nkey][suite]["hard"][mode] = evaluate(
                        views[mode], num_registers=num_registers, depth=depth, split=split,
                        n=args.eval_n, batch_size=args.eval_batch_size, seed=eval_seed,
                        discrete_binding=True,
                    )
                    evaluations[label][nkey][suite]["soft"][mode] = evaluate(
                        views[mode], num_registers=num_registers, depth=depth, split=split,
                        n=args.eval_n, batch_size=args.eval_batch_size, seed=eval_seed,
                        discrete_binding=False,
                    )
            print(f"completed {label} n={num_registers}", flush=True)

    report = {
        "harness_version": HARNESS_VERSION,
        "question": "is X19D's remaining formal failure only fixed-temperature soft address margin?",
        "seed": args.seed,
        "eval_seed": args.eval_seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "train_depth": args.train_depth,
        "eval_n": args.eval_n,
        "training_cardinalities": list(TRAIN_CARDINALITIES),
        "unseen_cardinalities": list(UNSEEN_CARDINALITIES),
        "training_cardinality_counts": cardinality_counts,
        "parameters": parameters,
        "trainable_parameters": trainable_parameters,
        "minimum_answer_loss": minimum_answer_loss,
        "final_answer_loss": final_answer_loss,
        "maximum_post4000_grad_norm": maximum_post4000_grad_norm,
        "initial_orthogonality_error": initial_orthogonality_error,
        "final_orthogonality_error": final_orthogonality_error,
        "optimizer_contract": {
            "optimizer": "AdamW",
            "weight_decay": WEIGHT_DECAY,
            "global_gradient_clip_norm": GRAD_CLIP_NORM,
            "lr_max": LR_MAX,
            "lr_min": LR_MIN,
            "total_steps": PREREGISTERED_STEPS,
            "training_address_beta": TRAIN_BETA,
        },
        "validation_contract": {
            "beta16_replay": TRAIN_BETA,
            "beta64_counterfactual": COUNTERFACTUAL_BETA,
            "counterfactual_created_posttraining_only": True,
            "hard_address_must_match_between_betas": True,
            "no_beta_sweep": True,
            "no_learned_beta": True,
        },
        "supervision_contract": {
            "fixed_answer_register": 0,
            "intermediate_state_targets": False,
            "hidden_final_targets": False,
            "role_labels": False,
            "address_labels": False,
            "active_cardinality_is_supplied": True,
            "commands_are_supplied": True,
            "pretraining_unseen_role_generation": False,
        },
        "initial_address_train_cardinalities_only": initial_address_train_cardinalities_only,
        "final_address_beta16": final_address_beta16,
        "final_address_beta64": final_address_beta64,
        "evaluations": evaluations,
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
