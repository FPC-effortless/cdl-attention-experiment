"""PNDS-GATE-001 Stage 2b: held-out routing comparison against controls.

Single environment shared by every arm so the comparison is made on identical
held-out episodes. The learned router is trained on train seeds only and frozen
for evaluation; controls receive no training.

Arms:
  learned   - LinearBanditRouter trained on outcomes only (no gold labels)
  static    - fixed bit-agreement scorer (no learned weights)
  random    - uniform candidate choice
  oracle    - capacity ceiling; uses the hidden gold index (not a control)

Deliberately adds no persistence, verifier, recurrence, or new architecture.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
import random
from typing import Sequence

from .stage2_bandit import Candidate, Episode, make_episode, environment_success


def _agreement(query: tuple[int, ...], descriptor: tuple[int, ...], dim: int) -> float:
    """Fixed, non-learned similarity: fraction of agreeing bits."""
    return sum(1.0 if query[j] == descriptor[j] else 0.0 for j in range(dim)) / dim


def static_select(ep: Episode, dim: int) -> int:
    """Argmax over the fixed bit-agreement scorer. Ties broken deterministically."""
    scores = [_agreement(ep.query, c.descriptor, dim) for c in ep.candidates]
    best = max(scores)
    return scores.index(best)


def random_select(ep: Episode, rng: random.Random) -> int:
    return rng.randrange(len(ep.candidates))


@dataclass
class _FrozenRouter:
    """A trained LinearBanditRouter whose weights are frozen for evaluation."""

    router: object  # stage2_bandit.LinearBanditRouter

    def select(self, ep: Episode, rng: random.Random, temperature: float = 0.05) -> int:
        idx, _ = self.router.select(ep, rng, temperature)
        return idx


def _train_router(train_seeds: Sequence[int], n_candidates: int, dim: int, noise: float,
                  train_seed: int, lr: float) -> object:
    from .stage2_bandit import LinearBanditRouter
    router = LinearBanditRouter(dim, lr=lr)
    # Episode generation order is shuffled so the router does not see episodes in
    # a fixed seed order; the episodes themselves are unchanged.
    order = list(train_seeds)
    random.Random(train_seed).shuffle(order)
    for s in order:
        ep = make_episode(s, n_candidates, dim, noise)
        idx, probs = router.select(ep, random.Random(s), temperature=0.5)
        reward = float(environment_success(ep, idx))
        router.update(ep, idx, reward, probs)
    return router


def evaluate_heldout(
    *,
    train_seeds: Sequence[int],
    test_seeds: Sequence[int],
    n_candidates: int = 8,
    dim: int = 8,
    noise: float = 0.10,
    train_seed: int = 0,
    eval_seed: int = 1,
    lr: float = 0.05,
) -> dict:
    """All arms are evaluated on the identical held-out `test_seeds` episodes."""
    router = _train_router(train_seeds, n_candidates, dim, noise, train_seed, lr)
    frozen = _FrozenRouter(router)
    rng = random.Random(eval_seed)

    rows = []
    for s in test_seeds:
        ep = make_episode(s, n_candidates, dim, noise)
        arms: dict[str, int] = {
            "learned": frozen.select(ep, random.Random(s + 1_000_000)),
            "static": static_select(ep, dim),
            "random": random_select(ep, rng),
            "oracle": ep.gold_index,
        }
        for name, idx in arms.items():
            rows.append({
                "seed": s,
                "arm": name,
                "selected": idx,
                "gold_index": ep.gold_index if name == "oracle" else None,
                "top1": float(idx == ep.gold_index),
                "success": float(environment_success(ep, idx)),
            })

    summary: dict[str, dict[str, float]] = {}
    for arm in ("learned", "static", "random", "oracle"):
        subset = [r for r in rows if r["arm"] == arm]
        n = len(subset)
        summary[arm] = {
            "n_episodes": n,
            "top1_mean": sum(r["top1"] for r in subset) / n,
            "success_mean": sum(r["success"] for r in subset) / n,
            "chance": 1.0 / n_candidates,
        }

    summary["deltas"] = {
        "learned_minus_static": summary["learned"]["success_mean"] - summary["static"]["success_mean"],
        "learned_minus_random": summary["learned"]["success_mean"] - summary["random"]["success_mean"],
        "learned_minus_chance": summary["learned"]["success_mean"] - 1.0 / n_candidates,
    }

    return {
        "protocol": "PNDS-GATE-001",
        "stage": "2b",
        "config": {
            "train_episodes": len(train_seeds),
            "test_episodes": len(test_seeds),
            "candidates": n_candidates,
            "dim": dim,
            "noise": noise,
            "train_seed": train_seed,
            "eval_seed": eval_seed,
            "lr": lr,
        },
        "summary": summary,
        "rows": rows,
    }


def run(train_episodes: int = 2000, test_episodes: int = 300,
        n_candidates: int = 8, dim: int = 8, noise: float = 0.10,
        train_seed: int = 0, eval_seed: int = 1) -> dict:
    train_seeds = list(range(train_seed, train_seed + train_episodes))
    test_seeds = list(range(10_000, 10_000 + test_episodes))
    return evaluate_heldout(
        train_seeds=train_seeds,
        test_seeds=test_seeds,
        n_candidates=n_candidates,
        dim=dim,
        noise=noise,
        train_seed=train_seed,
        eval_seed=eval_seed,
    )


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
