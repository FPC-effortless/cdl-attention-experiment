from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Callable, Dict, List

from .data import Example, graph_reachability
from .eval_tasks import load_model, score_example


def state_long(rng: random.Random, n_events: int) -> Example:
    items = ["red", "blue", "green", "gold", "white", "black", "silver", "orange"]
    locs = ["box", "desk", "shelf", "bag", "tray", "room", "vault", "yard", "cart", "case"]
    state = {x: rng.choice(locs) for x in items}
    events = []
    for _ in range(n_events):
        obj = rng.choice(items)
        dst = rng.choice(locs)
        state[obj] = dst
        events.append(f"move {obj} to {dst}")
    q = rng.choice(items)
    return Example(
        f"task state tracking\n{' ; '.join(events)}\nwhere {q}\nanswer {state[q]}",
        "state_long",
        state[q],
    )


def associative_long(rng: random.Random, n_keys: int) -> Example:
    keys = [f"k{i}" for i in range(n_keys)]
    mapping = {k: rng.randint(10, 99) for k in keys}
    facts = [f"{k}={v}" for k, v in mapping.items()]
    rng.shuffle(facts)
    q = rng.choice(keys)
    return Example(
        f"task associative recall\nfacts {' '.join(facts)}\nquery {q}\nanswer {mapping[q]}",
        "assoc_long",
        str(mapping[q]),
    )


def graph_scaled(rng: random.Random, hard: bool) -> Example:
    # Uses the corrected balanced graph generator. `hard` changes node/edge count.
    return graph_reachability(rng, hard=hard)


def aggregate(vals: List[Dict[str, float]]) -> Dict[str, float]:
    return {k: sum(v[k] for v in vals) / len(vals) for k in vals[0]}


def run_group(model, fn: Callable[[random.Random], Example], n: int, seed: int):
    rng = random.Random(seed)
    vals = [score_example(model, fn(rng)) for _ in range(n)]
    return aggregate(vals)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--seed", type=int, default=20261001)
    args = p.parse_args()

    groups = {
        "graph_easy_balanced": lambda r: graph_scaled(r, False),
        "graph_hard_balanced": lambda r: graph_scaled(r, True),
        "state_12": lambda r: state_long(r, 12),
        "state_24": lambda r: state_long(r, 24),
        "state_48": lambda r: state_long(r, 48),
        "state_96": lambda r: state_long(r, 96),
        "assoc_12": lambda r: associative_long(r, 12),
        "assoc_24": lambda r: associative_long(r, 24),
        "assoc_48": lambda r: associative_long(r, 48),
        "assoc_96": lambda r: associative_long(r, 96),
    }

    results = {}
    for cp in args.checkpoints:
        model = load_model(cp)
        name = Path(cp).stem
        results[name] = {}
        for i, (group, fn) in enumerate(groups.items()):
            results[name][group] = run_group(model, fn, args.n, args.seed + i * 100003)
            print(name, group, json.dumps(results[name][group]), flush=True)

    print("RESULT_JSON")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
