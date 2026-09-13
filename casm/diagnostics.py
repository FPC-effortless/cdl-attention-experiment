"""Mechanical Phase 1 controls and falsification diagnostics."""

from dataclasses import dataclass
from typing import Dict

from .generator import Episode, generate_episode
from .oracle import equivalent_to_target, evaluate_episode, exhaustive_minimality


@dataclass(frozen=True)
class CopyMaskReport:
    candidate_edges: int
    true_edges: int
    distractor_edges: int
    copy_mask_equals_true: bool
    copy_mask_executable: bool
    copy_mask_exact: bool


def copy_mask_report(episode: Episode) -> CopyMaskReport:
    """Evaluate the trivial control g_ij = m_i m_j.

    All nodes in the generated episode exist, so this control assigns one to
    every admissible candidate edge.  Because the candidate substrate is a
    genuine superset, it must not be mistaken for the episode's true wiring.
    """

    copy_equals_true = len(episode.candidate_edges) == len(episode.true_edges)
    try:
        copy_exact = evaluate_episode(episode, episode.candidate_edges) == episode.truth_table
        executable = True
    except (ValueError, KeyError):
        copy_exact = False
        executable = False
    return CopyMaskReport(
        candidate_edges=len(episode.candidate_edges),
        true_edges=len(episode.true_edges),
        distractor_edges=len(episode.candidate_edges) - len(episode.true_edges),
        copy_mask_equals_true=copy_equals_true,
        copy_mask_executable=executable,
        copy_mask_exact=copy_exact,
    )


def phase1_preflight(seed: int = 0, n_inputs: int = 3, n_ops: int = 4) -> Dict[str, object]:
    episode = generate_episode(seed=seed, n_inputs=n_inputs, n_ops=n_ops)
    report = copy_mask_report(episode)
    minimality = exhaustive_minimality(episode, max_edges=12)
    return {
        "candidate_edge_count": report.candidate_edges,
        "true_edge_count": report.true_edges,
        "distractor_edge_count": report.distractor_edges,
        "copy_mask_is_trivial": report.copy_mask_equals_true,
        "copy_mask_executes": report.copy_mask_executable,
        "copy_mask_exact": report.copy_mask_exact,
        "oracle_truth_table_rows": len(episode.truth_table),
        "global_minimality_check": minimality,
        "ready_for_router_training": (
            report.distractor_edges > 0
            and not report.copy_mask_equals_true
            and minimality.get("checked", False)
        ),
    }


def gate1_gate_causality(episode: Episode) -> bool:
    """Gate 1: changing a critical selected edge must change computation."""
    if not episode.true_edges:
        return False
    critical = episode.true_edges[0]
    severed = tuple(e for e in episode.true_edges if e != critical)
    try:
        changed = evaluate_episode(episode, severed) != episode.truth_table
    except (ValueError, KeyError):
        changed = True
    return changed
