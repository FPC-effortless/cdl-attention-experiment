from .generator import Episode
from .oracle import evaluate_episode, exhaustive_minimality


def copy_mask_report(episode: Episode) -> dict:
    """Control g_ij=m_i m_j; all generated nodes exist, so g=1 on A."""
    candidate = len(episode.candidate_edges)
    true = len(episode.true_edges)
    try:
        exact = evaluate_episode(episode, episode.candidate_edges) == episode.truth_table
        executable = True
    except (KeyError, ValueError):
        exact = False
        executable = False
    return {
        "candidate_edges": candidate,
        "true_edges": true,
        "distractor_edges": candidate - true,
        "copy_mask_equals_true": candidate == true,
        "copy_mask_executable": executable,
        "copy_mask_exact": exact,
    }


def preflight(episode: Episode, max_edges: int = 12) -> dict:
    report = copy_mask_report(episode)
    minimality = exhaustive_minimality(episode, max_edges=max_edges)
    return {
        **report,
        "oracle_rows": len(episode.truth_table),
        "minimality": minimality,
        "ready_for_router": (
            report["distractor_edges"] > 0
            and not report["copy_mask_equals_true"]
            and minimality["checked"]
        ),
    }
