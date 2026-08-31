from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from .contextual_baselines import MatchedGRUProgramBaseline
from .contextual_data import make_contextual_batch
from .explicit_compute import SharedTransitionModel, aggregate_metrics, state_metrics
from .state_access_controls import StateAccessGRUControl

HARNESS_VERSION = "casm-x2-equal-state-access-v0-2026-08-31"


def _train_step(model, optimizer, batch) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss = model.training_loss(batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach())


def evaluate(model, *, depth: int, split: str, n: int, batch_size: int, seed: int):
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
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-max-depth", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260861)
    parser.add_argument("--eval-seed", type=int, default=20260941)
    parser.add_argument("--eval-n", type=int, default=384)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("casm-x2-state-access-output/results.json"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    models = {
        "explicit_transition": SharedTransitionModel(d_model=96),
        "state_access_gru": StateAccessGRUControl(d_model=112),
        "hidden_only_gru": MatchedGRUProgramBaseline(d_model=114),
    }
    optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for name, model in models.items()
    }
    parameter_counts = {name: model.parameter_count() for name, model in models.items()}
    target = parameter_counts["explicit_transition"]
    ratios = {name: count / target for name, count in parameter_counts.items()}
    for name, ratio in ratios.items():
        if not 0.95 <= ratio <= 1.05:
            raise RuntimeError(f"{name} is not parameter matched: {parameter_counts}")

    history = []
    for step in range(1, args.steps + 1):
        depth = 1 + ((step + args.seed) % args.train_max_depth)
        batch = make_contextual_batch(
            args.batch_size,
            depth,
            args.seed * 1_000_003 + step * 97,
            split="train",
        )
        row = {"step": step, "depth": depth}
        for name, model in models.items():
            model.train()
            row[f"{name}_loss"] = _train_step(model, optimizers[name], batch)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
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
        "question": "does X2's gap persist after equalizing true previous-state access during training?",
        "training_state_access": {
            "explicit_transition": "true previous state teacher-forced",
            "state_access_gru": "true previous state teacher-forced",
            "hidden_only_gru": "initial state only; previous states are targets but not inputs",
        },
        "rollout_state_access": {
            "explicit_transition": "own predicted previous state",
            "state_access_gru": "own predicted previous state plus latent GRU hidden state",
            "hidden_only_gru": "latent GRU hidden state only after initial state",
        },
        "seed": args.seed,
        "eval_seed": args.eval_seed,
        "steps": args.steps,
        "parameters": parameter_counts,
        "parameter_ratio_to_explicit": ratios,
        "training_history": history,
        "evaluation": evaluation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("RESULT_JSON")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
