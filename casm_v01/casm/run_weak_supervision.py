from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from .contextual_data import make_contextual_batch
from .explicit_compute import aggregate_metrics, state_metrics
from .weak_supervision import SoftExplicitTransitionModel, cloned_regime_models

HARNESS_VERSION = "casm-x3-weak-supervision-v0-2026-08-31"
REGIMES = ("process", "quarter", "final")


def train_step(model, optimizer, batch, regime):
    optimizer.zero_grad(set_to_none=True)
    out = model.training_loss(batch, regime)
    out["loss"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {
        "loss": float(out["loss"].detach()),
        "final_entropy": float(out["final_entropy"]),
        "supervised_steps": int(out["supervised_steps"].item()),
    }


def evaluate(model, *, depth: int, split: str, n: int, batch_size: int, seed: int):
    rows = []
    remaining = n
    offset = 0
    while remaining:
        size = min(batch_size, remaining)
        batch = make_contextual_batch(size, depth, seed + offset * 1009, split=split)
        pred = model.rollout_hard(batch)
        rows.append(state_metrics(pred, batch.target_states))
        remaining -= size
        offset += 1
    return aggregate_metrics(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--train-depth", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=20260881)
    p.add_argument("--eval-seed", type=int, default=20260961)
    p.add_argument("--eval-n", type=int, default=384)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--output", type=Path, default=Path("casm-x3-output/results.json"))
    args = p.parse_args()

    if args.train_depth != 8:
        raise ValueError("CASM-X3 v0 preregisters fixed training depth 8")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    seed_model = SoftExplicitTransitionModel(d_model=96)
    models = cloned_regime_models(seed_model)
    optimizers = {
        regime: torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for regime, model in models.items()
    }
    parameters = {regime: model.parameter_count() for regime, model in models.items()}
    assert len(set(parameters.values())) == 1, parameters

    history = []
    for step in range(1, args.steps + 1):
        batch = make_contextual_batch(
            args.batch_size,
            args.train_depth,
            args.seed * 1_000_003 + step * 97,
            split="train",
        )
        row = {"step": step}
        for regime in REGIMES:
            models[regime].train()
            metrics = train_step(models[regime], optimizers[regime], batch, regime)
            row[regime] = metrics
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
    for i, (suite, split, depth) in enumerate(suites):
        evaluation[suite] = {"split": split, "depth": depth}
        for regime in REGIMES:
            evaluation[suite][regime] = evaluate(
                models[regime],
                depth=depth,
                split=split,
                n=args.eval_n,
                batch_size=args.eval_batch_size,
                seed=args.eval_seed + i * 100_003,
            )
        print(suite, json.dumps(evaluation[suite], sort_keys=True), flush=True)

    report = {
        "harness_version": HARNESS_VERSION,
        "question": "how much intermediate state supervision is required when the model must roll forward its own differentiable explicit state?",
        "seed": args.seed,
        "eval_seed": args.eval_seed,
        "steps": args.steps,
        "train_depth": args.train_depth,
        "parameters": parameters,
        "supervision_contract": {
            "teacher_forcing": False,
            "semantic_operator_labels": False,
            "process": "target state after all 8 training transitions",
            "quarter": "target state after transitions 4 and 8 only",
            "final": "target state after transition 8 only",
            "all_regimes_roll_their_own_predicted_soft_state_forward": True,
            "evaluation_uses_hard_discrete_predicted_state": True,
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
