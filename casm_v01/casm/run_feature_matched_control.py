from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from .contextual_data import make_contextual_batch
from .explicit_compute import SharedTransitionModel, aggregate_metrics, state_metrics
from .state_access_controls import FeatureMatchedStateGRUControl, StateAccessGRUControl

HARNESS_VERSION = "casm-x2-feature-matched-control-v0-2026-08-31"


def train_step(model, optimizer, batch):
    optimizer.zero_grad(set_to_none=True)
    loss = model.training_loss(batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach())


def evaluate(model, depth, split, n, batch_size, seed):
    rows = []
    remaining = n
    offset = 0
    while remaining:
        size = min(batch_size, remaining)
        batch = make_contextual_batch(size, depth, seed + offset * 1009, split=split)
        rows.append(state_metrics(model.rollout(batch), batch.target_states))
        remaining -= size
        offset += 1
    return aggregate_metrics(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2400)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--train-max-depth", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=20260871)
    p.add_argument("--eval-seed", type=int, default=20260951)
    p.add_argument("--eval-n", type=int, default=384)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--output", type=Path, default=Path("casm-x2-feature-output/results.json"))
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    models = {
        "explicit_transition": SharedTransitionModel(d_model=96),
        "feature_matched_gru": FeatureMatchedStateGRUControl(d_model=88),
        "state_only_gru": StateAccessGRUControl(d_model=112),
    }
    optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for name, model in models.items()
    }
    params = {name: model.parameter_count() for name, model in models.items()}
    target = params["explicit_transition"]
    ratios = {name: count / target for name, count in params.items()}
    for name, ratio in ratios.items():
        if not 0.95 <= ratio <= 1.05:
            raise RuntimeError(f"parameter mismatch: {params}")

    history = []
    for step in range(1, args.steps + 1):
        depth = 1 + ((step + args.seed) % args.train_max_depth)
        batch = make_contextual_batch(
            args.batch_size, depth, args.seed * 1_000_003 + step * 97, split="train"
        )
        row = {"step": step, "depth": depth}
        for name, model in models.items():
            model.train()
            row[f"{name}_loss"] = train_step(model, optimizers[name], batch)
        if step == 1 or step % 100 == 0 or step == args.steps:
            history.append(row)
            print(json.dumps(row), flush=True)

    for model in models.values():
        model.eval()

    suites = [
        ("iid_depth_4", "iid", 4),
        ("composition_depth_6", "composition", 6),
        ("composition_depth_12", "composition", 12),
        ("stress_depth_24", "composition", 24),
        ("stress_depth_48", "composition", 48),
        ("stress_depth_96", "composition", 96),
    ]
    evaluation = {}
    for i, (suite, split, depth) in enumerate(suites):
        evaluation[suite] = {"split": split, "depth": depth}
        for name, model in models.items():
            evaluation[suite][name] = evaluate(
                model, depth, split, args.eval_n, args.eval_batch_size,
                args.eval_seed + i * 100_003,
            )
        print(suite, json.dumps(evaluation[suite], sort_keys=True), flush=True)

    report = {
        "harness_version": HARNESS_VERSION,
        "question": "does the explicit transition advantage survive equal previous-state and indexed-value feature access?",
        "parameters": params,
        "parameter_ratio_to_explicit": ratios,
        "seed": args.seed,
        "eval_seed": args.eval_seed,
        "steps": args.steps,
        "training_history": history,
        "evaluation": evaluation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("RESULT_JSON")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
