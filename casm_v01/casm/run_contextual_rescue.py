from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from .contextual_baselines import MatchedGRUProgramBaseline, MatchedTransformerProgramBaseline
from .contextual_data import make_contextual_batch
from .explicit_compute import SharedTransitionModel, aggregate_metrics, state_metrics

HARNESS_VERSION = "casm-x2-generic-convergence-rescue-v0-2026-08-31"


def _train_step(model, optimizer, batch) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss = model.training_loss(batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach())


def evaluate(model, depth: int, split: str, n: int, batch_size: int, seed: int):
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260851)
    parser.add_argument("--eval-seed", type=int, default=20260931)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-n", type=int, default=384)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, default=Path("casm-x2-rescue-output/results.json"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    explicit_reference = SharedTransitionModel(d_model=96)
    models = {
        "gru": MatchedGRUProgramBaseline(d_model=114),
        "transformer": MatchedTransformerProgramBaseline(d_model=92, max_depth=128),
    }
    optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for name, model in models.items()
    }
    parameter_counts = {
        "explicit_reference": explicit_reference.parameter_count(),
        **{name: model.parameter_count() for name, model in models.items()},
    }
    target = parameter_counts["explicit_reference"]
    for name in models:
        ratio = parameter_counts[name] / target
        if not 0.95 <= ratio <= 1.05:
            raise RuntimeError(f"{name} is not parameter matched: {parameter_counts}")

    # One-sided optimization rescue: only generic controls receive this extra compute.
    stages = [(1, 1500), (2, 2000), (3, 2500), (4, 4000)]
    history = []
    iid_checkpoints = []
    global_step = 0
    for depth, stage_steps in stages:
        for stage_step in range(1, stage_steps + 1):
            global_step += 1
            batch = make_contextual_batch(
                args.batch_size,
                depth,
                args.seed * 1_000_003 + global_step * 97,
                split="train",
            )
            row = {"step": global_step, "stage_depth": depth}
            for name, model in models.items():
                model.train()
                row[f"{name}_loss"] = _train_step(model, optimizers[name], batch)
            if global_step == 1 or global_step % 250 == 0:
                history.append(row)
                print(json.dumps(row), flush=True)
            if depth == 4 and stage_step % 500 == 0:
                checkpoint = {"step": global_step, "stage_depth": depth}
                for name, model in models.items():
                    model.eval()
                    checkpoint[name] = evaluate(
                        model,
                        depth=4,
                        split="iid",
                        n=256,
                        batch_size=64,
                        seed=args.eval_seed + stage_step * 17,
                    )
                iid_checkpoints.append(checkpoint)
                print("IID_CHECKPOINT", json.dumps(checkpoint, sort_keys=True), flush=True)

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
    for index, (suite, split, depth) in enumerate(suites):
        evaluation[suite] = {"split": split, "depth": depth}
        for name, model in models.items():
            evaluation[suite][name] = evaluate(
                model,
                depth=depth,
                split=split,
                n=args.eval_n,
                batch_size=args.eval_batch_size,
                seed=args.eval_seed + index * 100_003,
            )
        print(suite, json.dumps(evaluation[suite], sort_keys=True), flush=True)

    report = {
        "harness_version": HARNESS_VERSION,
        "purpose": "one-sided convergence rescue for generic controls after CASM-X2 v0 baseline underfitting",
        "seed": args.seed,
        "eval_seed": args.eval_seed,
        "curriculum": [{"depth": depth, "steps": steps} for depth, steps in stages],
        "total_steps": global_step,
        "parameters": parameter_counts,
        "history": history,
        "iid_checkpoints": iid_checkpoints,
        "evaluation": evaluation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("RESULT_JSON")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
