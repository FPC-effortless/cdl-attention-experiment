from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Callable, Dict, List

import torch

from .data import TASKS, Example, graph_reachability
from .eval_tasks import score_example
from .free_generate_eval import evaluate_group
from .model import CASMConfig
from .recurrent_model import CASMRecurrent
from .stress_eval import associative_long, state_long


def load_recurrent(path: str, override_steps: int | None = None) -> CASMRecurrent:
    ckpt = torch.load(path, map_location="cpu")
    cfg = CASMConfig(**ckpt["config"])
    steps = int(
        override_steps
        if override_steps is not None
        else ckpt.get("reasoning_steps", 1)
    )
    model = CASMRecurrent(cfg, reasoning_steps=steps)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def aggregate(vals: List[Dict[str, float]]) -> Dict[str, float]:
    return {
        k: sum(v[k] for v in vals) / len(vals)
        for k in vals[0]
    }


def eval_tasks(model: CASMRecurrent, seed: int, n: int, hard: bool):
    rng = random.Random(seed)
    rows: Dict[str, List[Dict[str, float]]] = {}
    for fn in TASKS:
        rows[fn.__name__] = [score_example(model, fn(rng, hard=hard)) for _ in range(n)]
    out = {name: aggregate(vals) for name, vals in rows.items()}
    flat = [v for vals in rows.values() for v in vals]
    out["overall"] = aggregate(flat)
    return out


def stress_groups() -> Dict[str, Callable[[random.Random], Example]]:
    return {
        "graph_easy_balanced": lambda r: graph_reachability(r, hard=False),
        "graph_hard_balanced": lambda r: graph_reachability(r, hard=True),
        "state_12": lambda r: state_long(r, 12),
        "state_24": lambda r: state_long(r, 24),
        "state_48": lambda r: state_long(r, 48),
        "state_96": lambda r: state_long(r, 96),
        "assoc_12": lambda r: associative_long(r, 12),
        "assoc_24": lambda r: associative_long(r, 24),
        "assoc_48": lambda r: associative_long(r, 48),
        "assoc_96": lambda r: associative_long(r, 96),
    }


def eval_stress(model: CASMRecurrent, seed: int, n: int):
    out = {}
    for i, (name, fn) in enumerate(stress_groups().items()):
        rng = random.Random(seed + i * 100003)
        vals = [score_example(model, fn(rng)) for _ in range(n)]
        out[name] = aggregate(vals)
    return out


def eval_freegen(
    model: CASMRecurrent,
    seed: int,
    n: int,
    max_new_tokens: int,
):
    out = {}
    for i, (name, fn) in enumerate(stress_groups().items()):
        out[name] = evaluate_group(
            model,
            fn,
            n=n,
            seed=seed + i * 100003,
            max_new_tokens=max_new_tokens,
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--mode", choices=["task", "stress", "freegen"], required=True)
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--seed", type=int, default=20261102)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument(
        "--depths",
        default="",
        help="Optional comma-separated recurrence depths. If set, every checkpoint is evaluated at each depth.",
    )
    args = p.parse_args()

    depths = [int(x) for x in args.depths.split(",") if x.strip()]
    if not depths:
        depths = [None]

    results: Dict[str, object] = {}
    for cp in args.checkpoints:
        for depth in depths:
            model = load_recurrent(cp, override_steps=depth)
            base = Path(cp).stem
            name = base if depth is None else f"{base}@depth{depth}"
            if args.mode == "task":
                results[name] = {
                    "reasoning_steps": model.reasoning_steps,
                    "easy": eval_tasks(model, args.seed, args.n, False),
                    "hard": eval_tasks(model, args.seed + 100000, args.n, True),
                }
                print(
                    name,
                    json.dumps(results[name]["hard"]["overall"]),
                    flush=True,
                )
            elif args.mode == "stress":
                results[name] = {
                    "reasoning_steps": model.reasoning_steps,
                    **eval_stress(model, args.seed, args.n),
                }
                print(name, "stress complete", flush=True)
            else:
                results[name] = {
                    "reasoning_steps": model.reasoning_steps,
                    **eval_freegen(
                        model, args.seed, args.n, args.max_new_tokens
                    ),
                }
                print(name, "free generation complete", flush=True)

    print("RESULT_JSON")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
