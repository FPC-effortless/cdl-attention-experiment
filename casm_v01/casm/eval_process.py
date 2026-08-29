from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Callable, Dict, List

from .data import Example
from .eval_recurrent import aggregate, load_recurrent
from .eval_tasks import score_example
from .free_generate_eval import evaluate_group, greedy_answer
from .process_data import (
    PROCESS_TASKS,
    ProcessExample,
    associative_long,
    corrected_state_long,
    graph_process,
)


def _ex_fn(fn: Callable[[random.Random, bool], ProcessExample], hard: bool):
    return lambda rng: fn(rng, hard).example


def task_eval(model, seed: int, n: int, hard: bool):
    rng = random.Random(seed)
    out = {}
    flat = []
    for fn in PROCESS_TASKS:
        vals = [score_example(model, fn(rng, hard).example) for _ in range(n)]
        name = fn.__name__
        out[name] = aggregate(vals)
        flat.extend(vals)
    out["overall"] = aggregate(flat)
    return out


def stress_groups() -> Dict[str, Callable[[random.Random], Example]]:
    return {
        "graph_easy": lambda r: graph_process(r, False).example,
        "graph_hard": lambda r: graph_process(r, True).example,
        "state_12": lambda r: corrected_state_long(r, 12),
        "state_24": lambda r: corrected_state_long(r, 24),
        "state_48": lambda r: corrected_state_long(r, 48),
        "state_96": lambda r: corrected_state_long(r, 96),
        "assoc_12": lambda r: associative_long(r, 12),
        "assoc_24": lambda r: associative_long(r, 24),
        "assoc_48": lambda r: associative_long(r, 48),
        "assoc_96": lambda r: associative_long(r, 96),
    }


def stress_eval(model, seed: int, n: int):
    out = {}
    for i, (name, fn) in enumerate(stress_groups().items()):
        rng = random.Random(seed + i * 100003)
        out[name] = aggregate([score_example(model, fn(rng)) for _ in range(n)])
    return out


def task_freegen(model, seed: int, n: int, max_new_tokens: int, hard: bool):
    out = {}
    for i, fn in enumerate(PROCESS_TASKS):
        out[fn.__name__] = evaluate_group(
            model,
            _ex_fn(fn, hard),
            n=n,
            seed=seed + i * 100003,
            max_new_tokens=max_new_tokens,
        )
    return out


def stress_freegen(model, seed: int, n: int, max_new_tokens: int):
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


def exact_balanced_graphs(seed: int, hard: bool, per_label: int) -> List[Example]:
    rng = random.Random(seed)
    buckets = {"yes": [], "no": []}
    attempts = 0
    while min(len(buckets["yes"]), len(buckets["no"])) < per_label:
        ex = graph_process(rng, hard).example
        if len(buckets[ex.answer]) < per_label:
            buckets[ex.answer].append(ex)
        attempts += 1
        if attempts > per_label * 100:
            raise RuntimeError("could not construct exact balanced graph set")
    rows = buckets["yes"] + buckets["no"]
    random.Random(seed + 999).shuffle(rows)
    return rows


def balanced_graph_generation(model, seed: int, per_label: int, max_new_tokens: int):
    out = {}
    for hard in (False, True):
        rows = exact_balanced_graphs(seed + int(hard) * 99991, hard, per_label)
        correct = 0
        by_gold = {"yes": {"correct": 0, "total": 0}, "no": {"correct": 0, "total": 0}}
        examples = []
        for i, ex in enumerate(rows):
            pred = greedy_answer(model, ex, max_new_tokens=max_new_tokens)
            hit = pred == ex.answer
            correct += int(hit)
            by_gold[ex.answer]["correct"] += int(hit)
            by_gold[ex.answer]["total"] += 1
            if i < 16:
                examples.append({"gold": ex.answer, "pred": pred})
        out["hard" if hard else "easy"] = {
            "exact_accuracy": correct / len(rows),
            "n": len(rows),
            "by_gold": {
                k: {**v, "accuracy": v["correct"] / v["total"]}
                for k, v in by_gold.items()
            },
            "examples": examples,
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--mode", choices=["task", "stress", "freegen-task", "freegen-stress", "balanced-graph"], required=True)
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--seed", type=int, default=20261202)
    p.add_argument("--max-new-tokens", type=int, default=20)
    p.add_argument("--hard", action="store_true")
    args = p.parse_args()

    result = {}
    for cp in args.checkpoints:
        model = load_recurrent(cp)
        name = Path(cp).stem
        if args.mode == "task":
            result[name] = task_eval(model, args.seed, args.n, args.hard)
        elif args.mode == "stress":
            result[name] = stress_eval(model, args.seed, args.n)
        elif args.mode == "freegen-task":
            result[name] = task_freegen(model, args.seed, args.n, args.max_new_tokens, args.hard)
        elif args.mode == "freegen-stress":
            result[name] = stress_freegen(model, args.seed, args.n, args.max_new_tokens)
        else:
            result[name] = balanced_graph_generation(model, args.seed, args.n, args.max_new_tokens)
        print(name, args.mode, "complete", flush=True)

    print("RESULT_JSON")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
