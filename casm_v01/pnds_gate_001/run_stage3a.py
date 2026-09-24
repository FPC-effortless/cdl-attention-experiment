"""Run PNDS-GATE-001 Stage 3a weight-lesion test and emit JSON.

Structure mirrors run_stage2b.py / run_stage2c.py so the stages are comparable.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from .stage3a_lesion import run


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-episodes", type=int, default=2000)
    p.add_argument("--test-episodes", type=int, default=300)
    p.add_argument("--candidates", type=int, default=8)
    p.add_argument("--dim", type=int, default=8)
    p.add_argument("--noise", type=float, default=0.10)
    p.add_argument("--train-seed", type=int, default=0)
    p.add_argument("--eval-seed", type=int, default=1)
    p.add_argument("--seeds", type=int, default=1,
                   help="number of independent training seeds to replicate")
    p.add_argument("--output", type=Path, default=Path("pnds_gate_001_stage3a.json"))
    a = p.parse_args()

    if a.seeds <= 1:
        result = run(
            train_episodes=a.train_episodes,
            test_episodes=a.test_episodes,
            n_candidates=a.candidates,
            dim=a.dim,
            noise=a.noise,
            train_seed=a.train_seed,
            eval_seed=a.eval_seed,
        )
    else:
        result = _multiseed(a)

    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    s = result["summary"]
    if "lesion_effects" in s:
        for arm in ("intact", "joint", "gated_only", "qc_only", "gated_flip",
                    "static", "anti_static", "random", "oracle"):
            if arm in s:
                print(f"{arm:12s} success={s[arm]['success_mean']:.4f}")
        for metric, val in s["lesion_effects"].items():
            print(f"effect {metric:22s} = {val:+.4f}")
    else:
        for metric, st in s.items():
            print(f"{metric:22s} mean={st['mean']:+.4f} sd={st['sd']:.4f} n={st['n']}")
    print(f"artifact={a.output}")


def _multiseed(a) -> dict:
    """Replicate over independent training seeds, keeping the same test episodes."""
    import math
    from .stage3a_lesion import run as _run

    per_seed = []
    train_seeds_used = []
    for i in range(a.seeds):
        ts = a.train_seed + i * a.train_episodes
        train_seeds_used.append(ts)
        per_seed.append(_run(
            train_episodes=a.train_episodes,
            test_episodes=a.test_episodes,
            n_candidates=a.candidates,
            dim=a.dim,
            noise=a.noise,
            train_seed=ts,
            eval_seed=a.eval_seed,
        ))

    metrics = [m for m in ("intact_minus_joint", "intact_minus_gated_only",
                           "intact_minus_qc_only", "intact_minus_gated_flip")
               if m in per_seed[0]["summary"]["lesion_effects"]]
    stats: dict[str, dict[str, float]] = {}
    for metric in metrics:
        vals = [r["summary"]["lesion_effects"][metric] for r in per_seed]
        n = len(vals)
        mean = sum(vals) / n
        sd = math.sqrt(sum((x - mean) ** 2 for x in vals) / (n - 1)) if n > 1 else 0.0
        stats[metric] = {"mean": mean, "sd": sd, "n": n, "per_seed": vals}

    return {
        "protocol": "PNDS-GATE-001",
        "stage": "3a-multiseed",
        "config": {
            "train_episodes": a.train_episodes,
            "test_episodes": a.test_episodes,
            "candidates": a.candidates,
            "dim": a.dim,
            "noise": a.noise,
            "eval_seed": a.eval_seed,
            "n_train_seeds": a.seeds,
            "train_seeds": train_seeds_used,
        },
        "summary": stats,
        "per_seed_results": [
            {"train_seed": ts, "summary": r["summary"]}
            for ts, r in zip(train_seeds_used, per_seed)
        ],
    }


if __name__ == "__main__":
    main()
