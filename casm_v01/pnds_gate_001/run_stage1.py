"""Run PNDS-GATE-001 Stage 1 and emit a JSON artifact."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .stage1_baselines import evaluate

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("pnds_gate_001_stage1.json"))
    args = parser.parse_args()
    result = evaluate(
        seeds=range(args.seed, args.seed + args.episodes),
        n_candidates=args.candidates,
        random_seed=args.seed,
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for arm, metrics in result["summary"].items():
        print(f"{arm}: top1={metrics['top1_mean']:.4f} success={metrics['success_mean']:.4f} intervention={metrics['intervention_effect_mean']:.4f}")
    print(f"artifact={args.output}")

if __name__ == "__main__":
    main()
