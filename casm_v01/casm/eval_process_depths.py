from __future__ import annotations

import argparse
import json
from pathlib import Path

from .eval_process import task_eval, task_freegen
from .eval_recurrent import load_recurrent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--depths", default="1,2,3,5")
    p.add_argument("--n-likelihood", type=int, default=40)
    p.add_argument("--n-freegen", type=int, default=20)
    p.add_argument("--seed", type=int, default=20261229)
    args = p.parse_args()

    result = {}
    for depth in [int(x) for x in args.depths.split(",") if x.strip()]:
        model = load_recurrent(args.checkpoint, override_steps=depth)
        name = f"{Path(args.checkpoint).stem}@depth{depth}"
        result[name] = {
            "depth": depth,
            "hard_likelihood": task_eval(model, args.seed, args.n_likelihood, True),
            "hard_freegen": task_freegen(model, args.seed + 100000, args.n_freegen, 20, True),
        }
        print(name, "complete", flush=True)
    print("RESULT_JSON")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
