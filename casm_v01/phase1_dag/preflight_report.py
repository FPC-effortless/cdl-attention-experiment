"""Deterministic Phase 1 benchmark preflight over a seed suite.

This runs before CASM-S training. It answers whether the benchmark contains a
real edge-selection problem rather than an existence-mask shortcut.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from .diagnostics import copy_mask_report, preflight
from .generator import generate_episode


SEEDS = tuple(range(32))


def run_suite(*, seeds=SEEDS, n_inputs=2, n_ops=2, max_edges=12) -> dict:
    rows = []
    for seed in seeds:
        episode = generate_episode(seed=seed, n_inputs=n_inputs, n_ops=n_ops)
        copy = copy_mask_report(episode)
        check = preflight(episode, max_edges=max_edges)
        rows.append(
            {
                "seed": seed,
                "nodes": len(episode.nodes),
                "candidate_edges": len(episode.candidate_edges),
                "true_edges": len(episode.true_edges),
                "distractor_edges": copy["distractor_edges"],
                "copy_mask_equals_true": copy["copy_mask_equals_true"],
                "copy_mask_executable": copy["copy_mask_executable"],
                "copy_mask_exact": copy["copy_mask_exact"],
                "minimality_checked": check["minimality"]["checked"],
                "minimal": check["minimality"]["minimal"],
                "unique_minimal": check["minimality"]["unique_minimal"],
            }
        )

    total = len(rows)
    summary = {
        "seeds": total,
        "candidate_edges": {
            "min": min(r["candidate_edges"] for r in rows),
            "max": max(r["candidate_edges"] for r in rows),
            "mean": sum(r["candidate_edges"] for r in rows) / total,
        },
        "true_edges": {
            "min": min(r["true_edges"] for r in rows),
            "max": max(r["true_edges"] for r in rows),
            "mean": sum(r["true_edges"] for r in rows) / total,
        },
        "distractor_edges": {
            "min": min(r["distractor_edges"] for r in rows),
            "max": max(r["distractor_edges"] for r in rows),
            "mean": sum(r["distractor_edges"] for r in rows) / total,
        },
        "copy_mask_equals_true": sum(r["copy_mask_equals_true"] for r in rows),
        "copy_mask_executable": sum(r["copy_mask_executable"] for r in rows),
        "copy_mask_exact": sum(r["copy_mask_exact"] for r in rows),
        "minimality_checked": sum(r["minimality_checked"] for r in rows),
        "minimal": sum(r["minimal"] is True for r in rows),
        "unique_minimal": sum(r["unique_minimal"] is True for r in rows),
        "rows": rows,
    }
    return summary


def main() -> None:
    report = run_suite()
    print(json.dumps(report, indent=2, sort_keys=True))

    if report["copy_mask_equals_true"] != 0:
        raise SystemExit("FALSIFICATION: copy-mask equals true edge mask for at least one seed")
    if report["copy_mask_executable"] != 0:
        raise SystemExit("FALSIFICATION: existence-only copy-mask is executable for at least one seed")
    if report["copy_mask_exact"] != 0:
        raise SystemExit("FALSIFICATION: existence-only copy-mask exactly solves at least one seed")
    if report["minimality_checked"] != report["seeds"]:
        raise SystemExit("PRECONDITION FAILURE: exhaustive minimality was not checked for every seed")


if __name__ == "__main__":
    main()
