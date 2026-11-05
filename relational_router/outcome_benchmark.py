"""LRR-002 paired outcome-credit experiment; not a language-model benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import build_trial, select_trial
from .outcome_coder import controlled_pair_outcomes, fit_from_outcomes


def run(
    *,
    seeds: tuple[int, ...] = (101, 102, 103, 104, 105, 106),
    lengths: tuple[int, ...] = (8, 64, 256, 1024),
    trials: int = 32,
    steps: int = 800,
    n_codes: int = 6,
    balance_weight: float = 0.5,
) -> dict:
    if not seeds or not lengths or trials < 1:
        raise ValueError("seeds, lengths, and trials must not be empty")
    vocabulary, outcomes = controlled_pair_outcomes()
    report: dict = {
        "experiment": "LRR-002",
        "training_supervision": "fact-query answerability pairs, no canonical class ids",
        "runtime": "hard code + causal hash posting index; no attention",
        "entity_and_fact_segmentation": "supplied",
        "closed_vocabulary": len(vocabulary),
        "n_codes_supplied": n_codes,
        "seeds": list(seeds),
        "trials_per_length_per_seed": trials,
        "arms": {},
    }
    for arm, lam in (("balanced_pair_credit", balance_weight), ("unbalanced_pair_credit", 0.0)):
        by_length: dict[int, list[float]] = {n: [] for n in lengths}
        train_stats: list[dict] = []
        for seed in seeds:
            coder, fit = fit_from_outcomes(
                vocabulary, outcomes, n_codes=n_codes, seed=seed, steps=steps,
                balance_weight=lam,
            )
            train_stats.append({"seed": seed, **fit})
            for length in lengths:
                correct = visits = probes = 0
                for j in range(trials):
                    trial = build_trial(length, seed=seed * 10_000_000 + length * 1000 + j)
                    retrieved, _, _ = select_trial(coder, trial, k=1)
                    correct += int(trial.gold_token_index in retrieved.indices)
                    probes += retrieved.index_probes
                    visits += retrieved.posting_visits
                by_length[length].append(correct / trials)
                if arm == "balanced_pair_credit" and (probes != trials or visits != trials):
                    raise AssertionError("successful indexed arm did not use one posting per query")
        report["arms"][arm] = {
            "train_stats": train_stats,
            "heldout_recall_by_length": {
                str(n): {
                    "per_seed": scores,
                    "mean": sum(scores) / len(scores),
                    "min": min(scores),
                } for n, scores in by_length.items()
            },
        }
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="101,102,103,104,105,106")
    ap.add_argument("--lengths", default="8,64,256,1024")
    ap.add_argument("--trials", type=int, default=32)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--codes", type=int, default=6)
    ap.add_argument("--balance-weight", type=float, default=0.5)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    result = run(
        seeds=tuple(int(s) for s in args.seeds.split(",")),
        lengths=tuple(int(s) for s in args.lengths.split(",")),
        trials=args.trials, steps=args.steps,
        n_codes=args.codes, balance_weight=args.balance_weight,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
