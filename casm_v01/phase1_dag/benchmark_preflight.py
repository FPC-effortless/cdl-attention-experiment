"""Pre-CASM benchmark validity report.

The router is not trained until these controls establish that edge selection is
both necessary and nontrivial on the generated substrate.
"""

from __future__ import annotations

import json

from .controls import nonoracle_valid_control
from .diagnostics import copy_mask_report, preflight
from .generator import generate_episode
from .oracle import evaluate_episode


SEEDS = tuple(range(32))


def run_suite(*, seeds=SEEDS, n_inputs=2, n_ops=2, max_edges=12) -> dict:
    rows = []
    for seed in seeds:
        episode = generate_episode(seed=seed, n_inputs=n_inputs, n_ops=n_ops)
        copy = copy_mask_report(episode)
        check = preflight(episode, max_edges=max_edges)
        negative = nonoracle_valid_control(episode)
        oracle_exact = evaluate_episode(episode) == episode.truth_table
        rows.append({
            "seed": seed,
            "candidate_edges": len(episode.candidate_edges),
            "true_edges": len(episode.true_edges),
            "distractor_edges": copy["distractor_edges"],
            "oracle_exact": oracle_exact,
            "copy_mask_equals_true": copy["copy_mask_equals_true"],
            "copy_mask_executable": copy["copy_mask_executable"],
            "copy_mask_exact": copy["copy_mask_exact"],
            "minimality_checked": check["minimality"]["checked"],
            "minimal": check["minimality"]["minimal"],
            "unique_minimal": check["minimality"]["unique_minimal"],
            "nonoracle_valid_found": negative["found"],
            "nonoracle_valid_checked": negative["checked"],
        })

    total = len(rows)
    summary = {
        "seeds": total,
        "oracle_exact": sum(r["oracle_exact"] for r in rows),
        "copy_mask_equals_true": sum(r["copy_mask_equals_true"] for r in rows),
        "copy_mask_executable": sum(r["copy_mask_executable"] for r in rows),
        "copy_mask_exact": sum(r["copy_mask_exact"] for r in rows),
        "minimality_checked": sum(r["minimality_checked"] for r in rows),
        "minimal": sum(r["minimal"] is True for r in rows),
        "unique_minimal": sum(r["unique_minimal"] is True for r in rows),
        "nonoracle_valid_found": sum(r["nonoracle_valid_found"] for r in rows),
        "rows": rows,
    }
    summary["ready_for_router"] = (
        summary["oracle_exact"] == total
        and summary["copy_mask_equals_true"] == 0
        and summary["copy_mask_executable"] == 0
        and summary["copy_mask_exact"] == 0
        and summary["minimality_checked"] == total
        and summary["minimal"] == total
        and summary["unique_minimal"] == total
        and summary["nonoracle_valid_found"] == total
    )
    return summary


def main() -> None:
    report = run_suite()
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ready_for_router"]:
        raise SystemExit("BENCHMARK INVALID: Phase-1 edge-selection controls failed")


if __name__ == "__main__":
    main()
