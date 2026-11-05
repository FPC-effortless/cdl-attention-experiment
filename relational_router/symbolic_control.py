"""LRR-002A: symbolic equivalence control using *identical* pair labels.

This is a post-result diagnostic, not a preregistered neural improvement.
For a complete labeled alias-pair graph, union-find is a strong zero-parameter
competitor. Runtime still uses the same CausalRelationIndex.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .benchmark import build_trial, select_trial
from .outcome_coder import PairOutcome, controlled_pair_outcomes


class EquivalenceCodebook:
    def __init__(self, vocabulary: tuple[str, ...], outcomes: tuple[PairOutcome, ...]):
        parent = {word: word for word in vocabulary}
        def find(w: str) -> str:
            while parent[w] != w:
                parent[w] = parent[parent[w]]
                w = parent[w]
            return w

        for pair in outcomes:
            if pair.answered:
                x, y = find(pair.fact_predicate), find(pair.question_predicate)
                if x != y:
                    parent[y] = x

        for pair in outcomes:
            if not pair.answered and find(pair.fact_predicate) == find(pair.question_predicate):
                raise ValueError("inconsistent positive and negative pair graph")

        roots = tuple(sorted({find(w) for w in vocabulary}))
        root_code = {root: i for i, root in enumerate(roots)}
        self._codes = {word: root_code[find(word)] for word in vocabulary}
        self.n_codes = len(roots)
        self.trainable_parameters = 0
        self.pair_supervision_count = len(outcomes)

    def code(self, predicate: str) -> int | None:
        return self._codes.get(predicate.lower())


def run_control(
    *,
    seeds: tuple[int, ...] = (101, 102, 103, 104, 105, 106),
    lengths: tuple[int, ...] = (8, 64, 256, 1024),
    trials: int = 32,
) -> dict:
    vocabulary, outcomes = controlled_pair_outcomes()
    coder = EquivalenceCodebook(vocabulary, outcomes)
    result = {
        "experiment": "LRR-002A posthoc symbolic control",
        "codebook_size": coder.n_codes,
        "trainable_parameters": 0,
        "training_pairs": len(outcomes),
        "same_pair_supervision_as_LRR002": True,
        "seeds": list(seeds),
        "rows": [],
    }
    for n in lengths:
        per_seed: list[float] = []
        query_us: list[float] = []
        for seed in seeds:
            hits = 0
            for j in range(trials):
                t = build_trial(n, seed=seed * 10_000_000 + n * 1000 + j)
                selected, _, elapsed = select_trial(coder, t)
                query_us.append(elapsed)
                hits += int(t.gold_token_index in selected.indices)
                if selected.index_probes != 1 or selected.posting_visits != 1:
                    raise AssertionError("symbolic index missing its known key")
            per_seed.append(hits / trials)
        result["rows"].append({
            "facts": n,
            "mean_recall_at_1": sum(per_seed)/len(per_seed),
            "per_seed_recall": per_seed,
            "median_cpu_query_us": statistics.median(query_us),
            "trials_per_seed": trials,
        })
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    r = run_control()
    s = json.dumps(r, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(s + "\n")
    print(s)


if __name__ == "__main__":
    main()
