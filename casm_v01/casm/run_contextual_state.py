from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict

import torch

from .contextual_baselines import MatchedGRUProgramBaseline, MatchedTransformerProgramBaseline
from .contextual_data import make_contextual_batch
from .explicit_compute import SharedTransitionModel, aggregate_metrics, state_metrics

HARNESS_VERSION = "casm-x2-contextual-state-v0-2026-08-31"


def _train_step(model, optimizer, batch) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss = model.training_loss(batch)
    scalar = loss["loss"] if isinstance(loss, dict) else loss
    scalar.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(scalar.detach())


def train_models(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    explicit = SharedTransitionModel(d_model=96)
    gru = MatchedGRUProgramBaseline(d_model=114)
    transformer = MatchedTransformerProgramBaseline(d_model=92, max_depth=args.max_eval_depth)
    models = {"explicit_state": explicit, "gru": gru, "transformer": transformer}
    optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for name, model in models.items()
    }
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
    return models, history


def evaluate_model(model, *, depth: int, split: str, n: int, batch_size: int, seed: int) -> Dict[str, float]:
    rows = []
    semantic_counts = torch.zeros(8, dtype=torch.long)
    remaining = n
    offset = 0
    while remaining:
        size = min(batch_size, remaining)
        batch = make_contextual_batch(size, depth, seed + offset * 1009, split=split)
        pred = model.rollout(batch)
        rows.append(state_metrics(pred, batch.target_states))
        semantic_counts += torch.bincount(batch.semantics.reshape(-1), minlength=8)
        remaining -= size
        offset += 1
    out = aggregate_metrics(rows)
    out["active_semantics"] = float((semantic_counts > 0).sum().item())
    out["semantic_min_count"] = float(semantic_counts.min().item())
    out["semantic_max_count"] = float(semantic_counts.max().item())
    return out


def evaluate_all(models, args):
    suites = [
        ("iid_depth_4", "iid", 4),
        ("composition_depth_6", "composition", 6),
        ("composition_depth_12", "composition", 12),
        ("stress_depth_24", "composition", 24),
        ("stress_depth_48", "composition", 48),
        ("stress_depth_96", "composition", 96),
    ]
    result = {}
    for index, (name, split, depth) in enumerate(suites):
        seed = args.eval_seed + index * 100_003
        result[name] = {"split": split, "depth": depth}
        for model_name, model in models.items():
            result[name][model_name] = evaluate_model(
                model,
                depth=depth,
                split=split,
                n=args.eval_n,
                batch_size=args.eval_batch_size,
                seed=seed,
            )
        print(name, json.dumps(result[name], sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-max-depth", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260841)
    parser.add_argument("--eval-seed", type=int, default=20260921)
    parser.add_argument("--eval-n", type=int, default=384)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--max-eval-depth", type=int, default=128)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("casm-x2-output/results.json"))
    args = parser.parse_args()

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    models, history = train_models(args)
    parameter_counts = {name: model.parameter_count() for name, model in models.items()}
    target = parameter_counts["explicit_state"]
    parameter_ratios = {name: count / target for name, count in parameter_counts.items()}
    if not 0.95 <= parameter_ratios["gru"] <= 1.05:
        raise RuntimeError(f"GRU is not parameter matched: {parameter_counts}")
    if not 0.95 <= parameter_ratios["transformer"] <= 1.05:
        raise RuntimeError(f"Transformer is not parameter matched: {parameter_counts}")

    evaluation = evaluate_all(models, args)
    report = {
        "harness_version": HARNESS_VERSION,
        "hypothesis": "explicit predicted typed state improves contextual transition execution and length extrapolation over parameter-matched generic sequence models",
        "contract": {
            "train_depths": list(range(1, args.train_max_depth + 1)),
            "same_opaque_command_has_state_dependent_semantics": True,
            "semantic_operator_ids_are_private_evaluator_data": True,
            "all_models_receive_identical_initial_state_and_instruction_sequences": True,
            "all_models_receive_identical_per_step_state_supervision": True,
            "primary_metric": "final_state_exact",
            "stress_depths": [24, 48, 96],
        },
        "seed": args.seed,
        "eval_seed": args.eval_seed,
        "parameters": parameter_counts,
        "parameter_ratio_to_explicit": parameter_ratios,
        "training_history": history,
        "evaluation": evaluation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("RESULT_JSON")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
