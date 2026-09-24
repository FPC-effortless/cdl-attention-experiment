"""PNDS-GATE-001 Stage 1: oracle/static/random routing baselines.

This harness intentionally contains no learned router and no target/outcome input
to the routing functions. The hidden environment is used only to score actions.
It establishes capacity, fixed-retrieval, and chance baselines before Stage 2.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
import random
from typing import Callable, Sequence

@dataclass(frozen=True)
class Candidate:
    key: str
    action: int
    value: str
    provenance: str = "synthetic"

@dataclass(frozen=True)
class Episode:
    query: str
    candidates: tuple[Candidate, ...]
    gold_index: int

def make_episode(seed: int, n_candidates: int = 8) -> Episode:
    rng = random.Random(seed)
    concept = f"concept_{rng.randrange(10_000)}"
    # Candidate keys are opaque identifiers; lexical overlap cannot reveal the gold candidate.
    keys = [f"key_{rng.randrange(10_000)}" for _ in range(n_candidates)]
    rng.shuffle(keys)
    gold_index = rng.randrange(n_candidates)
    candidates = tuple(
        Candidate(key=k, action=i, value=f"structure for {k}")
        for i, k in enumerate(keys)
    )
    return Episode(query=f"retrieve {concept}", candidates=candidates, gold_index=gold_index)

def oracle_router(ep: Episode) -> int:
    return ep.gold_index

def static_overlap_router(ep: Episode) -> int:
    """Fixed similarity control; intentionally has no semantic key alignment."""
    return 0

def random_router(ep: Episode, rng: random.Random) -> int:
    return rng.randrange(len(ep.candidates))

def environment_success(ep: Episode, selected_index: int) -> bool:
    """Hidden environment: only the latent correct structure yields success."""
    return selected_index == ep.gold_index

def intervention_effect(ep: Episode, selected_index: int) -> float:
    """Matched do(action) effect: selected action vs a different candidate action."""
    selected = 1.0 if environment_success(ep, selected_index) else 0.0
    alternative = (selected_index + 1) % len(ep.candidates)
    intervened = 1.0 if environment_success(ep, alternative) else 0.0
    return selected - intervened

def mrr(ep: Episode, ranked: Sequence[int]) -> float:
    try:
        rank = ranked.index(ep.gold_index) + 1
    except ValueError:
        return 0.0
    return 1.0 / rank

def evaluate(
    *,
    seeds: Sequence[int],
    n_candidates: int = 8,
    random_seed: int = 0,
) -> dict:
    rng = random.Random(random_seed)
    rows = []
    for seed in seeds:
        ep = make_episode(seed, n_candidates)
        for name, selector in (
            ("oracle", lambda e: oracle_router(e)),
            ("static_overlap", lambda e: static_overlap_router(e)),
            ("random", lambda e: random_router(e, rng)),
        ):
            idx = selector(ep)
            rows.append({
                "seed": seed,
                "arm": name,
                "top1": float(idx == ep.gold_index),
                "success": float(environment_success(ep, idx)),
                "intervention_effect": intervention_effect(ep, idx),
                "mrr": mrr(ep, [idx]),
                "candidate_count": n_candidates,
            })
    summary = {}
    for arm in ("oracle", "static_overlap", "random"):
        subset = [r for r in rows if r["arm"] == arm]
        summary[arm] = {
            "top1_mean": sum(r["top1"] for r in subset) / len(subset),
            "success_mean": sum(r["success"] for r in subset) / len(subset),
            "intervention_effect_mean": sum(r["intervention_effect"] for r in subset) / len(subset),
            "mrr_mean": sum(r["mrr"] for r in subset) / len(subset),
            "candidate_count": n_candidates,
            "n_episodes": len(subset),
        }
    return {"protocol": "PNDS-GATE-001", "stage": 1, "rows": rows, "summary": summary}

if __name__ == "__main__":
    result = evaluate(seeds=range(30), n_candidates=8)
    print(json.dumps(result, indent=2, sort_keys=True))
