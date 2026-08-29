from __future__ import annotations

import argparse
import json
import random

from .eval_process import exact_balanced_graphs
from .eval_recurrent import load_recurrent
from .free_generate_eval import evaluate_group, greedy_answer
from .process_data import state_process


def balanced_graph(model, seed: int, hard: bool, per_label: int):
    rows = exact_balanced_graphs(seed, hard, per_label)
    correct = 0
    by_gold = {"yes": {"correct": 0, "total": 0}, "no": {"correct": 0, "total": 0}}
    examples = []
    for i, ex in enumerate(rows):
        pred = greedy_answer(model, ex, max_new_tokens=6)
        hit = pred == ex.answer
        correct += int(hit)
        by_gold[ex.answer]["correct"] += int(hit)
        by_gold[ex.answer]["total"] += 1
        if i < 20:
            examples.append({"gold": ex.answer, "pred": pred})
    return {
        "exact_accuracy": correct / len(rows),
        "by_gold": {k: {**v, "accuracy": v["correct"] / v["total"]} for k, v in by_gold.items()},
        "examples": examples,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--state-checkpoint", required=True)
    p.add_argument("--graph-checkpoint", required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=20261302)
    args = p.parse_args()

    state_model = load_recurrent(args.state_checkpoint)
    graph_model = load_recurrent(args.graph_checkpoint)
    result = {
        "state_easy": evaluate_group(
            state_model, lambda r: state_process(r, False).example,
            n=args.n, seed=args.seed, max_new_tokens=16,
        ),
        "state_hard": evaluate_group(
            state_model, lambda r: state_process(r, True).example,
            n=args.n, seed=args.seed + 100003, max_new_tokens=16,
        ),
        "graph_easy_exact_balanced": balanced_graph(
            graph_model, args.seed + 200003, False, args.n // 2
        ),
        "graph_hard_exact_balanced": balanced_graph(
            graph_model, args.seed + 300007, True, args.n // 2
        ),
    }
    print("RESULT_JSON")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
