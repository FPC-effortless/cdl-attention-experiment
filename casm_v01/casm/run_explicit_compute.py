from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch

from .explicit_compute import (
    ExplicitOperatorMachine,
    GRUProgramBaseline,
    HELDOUT_BIGRAMS,
    OPERATOR_NAMES,
    SharedTransitionModel,
    aggregate_metrics,
    make_program_batch,
    state_metrics,
)

HARNESS_VERSION = "casm-x-explicit-compute-v0-2026-08-31"


def _train_step(model, optimizer, loss_fn) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss = loss_fn()
    scalar = loss["loss"] if isinstance(loss, dict) else loss
    scalar.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(scalar.detach())


def train_models(args) -> Tuple[ExplicitOperatorMachine, SharedTransitionModel, GRUProgramBaseline, list[dict]]:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    explicit = ExplicitOperatorMachine(d_model=args.explicit_width)
    shared = SharedTransitionModel(d_model=args.shared_width)
    gru = GRUProgramBaseline(d_model=args.gru_width)
    explicit_opt = torch.optim.AdamW(explicit.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    shared_opt = torch.optim.AdamW(shared.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    gru_opt = torch.optim.AdamW(gru.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history: list[dict] = []

    for step in range(1, args.steps + 1):
        depth = 1 + ((step + args.seed) % args.train_max_depth)
        batch = make_program_batch(
            args.batch_size,
            depth,
            args.seed * 1_000_003 + step * 97,
            split="train",
        )
        explicit.train()
        shared.train()
        gru.train()
        e = _train_step(explicit, explicit_opt, lambda: explicit.training_loss(batch))
        s = _train_step(shared, shared_opt, lambda: shared.training_loss(batch))
        g = _train_step(gru, gru_opt, lambda: gru.training_loss(batch))
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            row = {"step": step, "depth": depth, "explicit_loss": e, "shared_loss": s, "gru_loss": g}
            history.append(row)
            print(json.dumps(row), flush=True)
    return explicit.eval(), shared.eval(), gru.eval(), history


def _chunk_sizes(n: int, chunk: int) -> Iterable[int]:
    left = n
    while left:
        take = min(left, chunk)
        yield take
        left -= take


def evaluate_explicit(
    model: ExplicitOperatorMachine,
    *,
    depth: int,
    split: str,
    n: int,
    batch_size: int,
    seed: int,
    use_verifier: bool,
    oracle_routing: bool = False,
    oracle_execution: bool = False,
) -> Dict[str, float]:
    rows = []
    route_correct = 0
    route_total = 0
    offset = 0
    for size in _chunk_sizes(n, batch_size):
        batch = make_program_batch(size, depth, seed + offset * 1009, split=split)
        pred, routes = model.rollout(
            batch,
            use_verifier=use_verifier,
            oracle_routing=oracle_routing,
            oracle_execution=oracle_execution,
        )
        rows.append(state_metrics(pred, batch.target_states))
        route_correct += int((routes == batch.semantics).sum())
        route_total += routes.numel()
        offset += 1
    out = aggregate_metrics(rows)
    out["route_accuracy"] = route_correct / max(1, route_total)
    return out


def evaluate_baseline(model, *, depth: int, split: str, n: int, batch_size: int, seed: int) -> Dict[str, float]:
    rows = []
    offset = 0
    for size in _chunk_sizes(n, batch_size):
        batch = make_program_batch(size, depth, seed + offset * 1009, split=split)
        pred = model.rollout(batch)
        rows.append(state_metrics(pred, batch.target_states))
        offset += 1
    return aggregate_metrics(rows)


def profile_rollout(name: str, fn, transitions: int, repeats: int = 5) -> Dict[str, float | str]:
    for _ in range(2):
        fn()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    elapsed = (time.perf_counter() - start) / repeats
    return {
        "model": name,
        "seconds": elapsed,
        "programs_per_second": transitions[0] / max(elapsed, 1e-12),
        "transitions_per_second": transitions[1] / max(elapsed, 1e-12),
    }


def evaluate_all(explicit, shared, gru, args) -> Dict[str, object]:
    suites = [
        ("iid_depth_1", "iid", 1),
        ("iid_depth_2", "iid", 2),
        ("iid_depth_3", "iid", 3),
        ("composition_depth_4", "composition", 4),
        ("composition_depth_6", "composition", 6),
        ("stress_depth_8", "composition", 8),
        ("stress_depth_12", "composition", 12),
    ]
    result: Dict[str, object] = {}
    for index, (name, split, depth) in enumerate(suites):
        seed = args.eval_seed + index * 100_003
        result[name] = {
            "split": split,
            "depth": depth,
            "explicit_verified": evaluate_explicit(
                explicit,
                depth=depth,
                split=split,
                n=args.eval_n,
                batch_size=args.eval_batch_size,
                seed=seed,
                use_verifier=True,
            ),
            "explicit_no_verifier": evaluate_explicit(
                explicit,
                depth=depth,
                split=split,
                n=args.eval_n,
                batch_size=args.eval_batch_size,
                seed=seed,
                use_verifier=False,
            ),
            "explicit_oracle_routing": evaluate_explicit(
                explicit,
                depth=depth,
                split=split,
                n=args.eval_n,
                batch_size=args.eval_batch_size,
                seed=seed,
                use_verifier=False,
                oracle_routing=True,
            ),
            "explicit_oracle_execution": evaluate_explicit(
                explicit,
                depth=depth,
                split=split,
                n=args.eval_n,
                batch_size=args.eval_batch_size,
                seed=seed,
                use_verifier=False,
                oracle_execution=True,
            ),
            "explicit_oracle_both": evaluate_explicit(
                explicit,
                depth=depth,
                split=split,
                n=args.eval_n,
                batch_size=args.eval_batch_size,
                seed=seed,
                use_verifier=False,
                oracle_routing=True,
                oracle_execution=True,
            ),
            "shared_transition": evaluate_baseline(
                shared,
                depth=depth,
                split=split,
                n=args.eval_n,
                batch_size=args.eval_batch_size,
                seed=seed,
            ),
            "gru_program": evaluate_baseline(
                gru,
                depth=depth,
                split=split,
                n=args.eval_n,
                batch_size=args.eval_batch_size,
                seed=seed,
            ),
        }
        print(name, json.dumps(result[name], sort_keys=True), flush=True)

    profile_batch = make_program_batch(
        args.profile_batch_size,
        args.profile_depth,
        args.eval_seed + 900_001,
        split="composition",
    )
    result["efficiency"] = [
        profile_rollout(
            "explicit_verified",
            lambda: explicit.rollout(profile_batch, use_verifier=True),
            (args.profile_batch_size, args.profile_batch_size * args.profile_depth),
        ),
        profile_rollout(
            "explicit_no_verifier",
            lambda: explicit.rollout(profile_batch, use_verifier=False),
            (args.profile_batch_size, args.profile_batch_size * args.profile_depth),
        ),
        profile_rollout(
            "shared_transition",
            lambda: shared.rollout(profile_batch),
            (args.profile_batch_size, args.profile_batch_size * args.profile_depth),
        ),
        profile_rollout(
            "gru_program",
            lambda: gru.rollout(profile_batch),
            (args.profile_batch_size, args.profile_batch_size * args.profile_depth),
        ),
    ]
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1800)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--train-max-depth", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--explicit-width", type=int, default=64)
    p.add_argument("--shared-width", type=int, default=96)
    p.add_argument("--gru-width", type=int, default=96)
    p.add_argument("--seed", type=int, default=20260831)
    p.add_argument("--eval-seed", type=int, default=20260901)
    p.add_argument("--eval-n", type=int, default=512)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--profile-batch-size", type=int, default=128)
    p.add_argument("--profile-depth", type=int, default=12)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--output", type=Path, default=Path("casm-x-output/results.json"))
    p.add_argument("--checkpoint-dir", type=Path, default=None)
    args = p.parse_args()

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    explicit, shared, gru, history = train_models(args)
    evaluation = evaluate_all(explicit, shared, gru, args)
    report = {
        "harness_version": HARNESS_VERSION,
        "hypothesis": "explicit typed modular state transitions improve compositional and depth generalization over equally supervised generic recurrent computation",
        "contract": {
            "train_depths": list(range(1, args.train_max_depth + 1)),
            "heldout_ordered_operator_bigrams": sorted([list(x) for x in HELDOUT_BIGRAMS]),
            "operators": list(OPERATOR_NAMES),
            "composition_split_requires_heldout_bigram": True,
            "semantic_operator_ids_are_not_model_inputs": True,
            "all_trainable_models_receive_identical_program examples": True,
            "all_trainable_models_receive_per_step_state supervision": True,
            "oracle_both_is_a_dataset_and_evaluator integrity check": True,
            "primary_metric": "final_state_exact",
            "secondary_metrics": ["step_state_exact", "register_accuracy", "route_accuracy"],
        },
        "seed": args.seed,
        "eval_seed": args.eval_seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "parameters": {
            "explicit_operator_machine": explicit.parameter_count(),
            "shared_transition": shared.parameter_count(),
            "gru_program": gru.parameter_count(),
        },
        "training_history": history,
        "evaluation": evaluation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.checkpoint_dir is not None:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(explicit.state_dict(), args.checkpoint_dir / "explicit.pt")
        torch.save(shared.state_dict(), args.checkpoint_dir / "shared.pt")
        torch.save(gru.state_dict(), args.checkpoint_dir / "gru.pt")
    print("RESULT_JSON")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
