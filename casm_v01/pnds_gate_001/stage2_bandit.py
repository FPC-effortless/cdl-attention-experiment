"""PNDS-GATE-001 Stage 2: outcome-trained relevance router.

The router observes query/candidate properties but never receives the hidden
gold index. Learning occurs only from the environment outcome after selection.
This is deliberately a small contextual-bandit primitive, not the final PNDS
router.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
import math
import random
from typing import Sequence

@dataclass(frozen=True)
class Candidate:
    key: str
    descriptor: tuple[int, ...]
    action: int
    provenance: str = "synthetic"

@dataclass(frozen=True)
class Episode:
    query: tuple[int, ...]
    candidates: tuple[Candidate, ...]
    gold_index: int  # evaluator/environment only; never exposed to router

def _flip(rng: random.Random, bits: tuple[int, ...], p: float) -> tuple[int, ...]:
    return tuple((b ^ (1 if rng.random() < p else 0)) for b in bits)

def make_episode(seed: int, n_candidates: int = 8, dim: int = 8, noise: float = 0.10) -> Episode:
    rng = random.Random(seed)
    latent = tuple(rng.randrange(2) for _ in range(dim))
    candidates = []
    for i in range(n_candidates):
        # Exactly one candidate is generated from the latent query concept;
        # distractors come from independent latent concepts. The environment's
        # hidden gold index is computed from the same latent relation, not sampled
        # independently. The router never receives the latent concept or gold index.
        source = latent if i == 0 else tuple(rng.randrange(2) for _ in range(dim))
        desc = _flip(rng, source, noise)
        candidates.append(Candidate(
            key=f"key_{rng.randrange(1_000_000)}",
            descriptor=desc,
            action=i,
        ))
    query = _flip(rng, latent, noise)
    # Randomize candidate order only after identifying the structurally correct candidate.
    gold = 0
    candidates = list(candidates)
    rng.shuffle(candidates)
    gold = next(i for i, c in enumerate(candidates) if c.action == 0)
    return Episode(query=query, candidates=tuple(candidates), gold_index=gold)

class LinearBanditRouter:
    """Small shared scorer over query/candidate bit agreement."""

    def __init__(self, dim: int, lr: float = 0.05):
        self.dim = dim
        self.lr = lr
        self.w = [0.0] * dim
        self.bias = 0.0

    def score(self, query: tuple[int, ...], candidate: Candidate) -> float:
        agreement = [1.0 if query[j] == candidate.descriptor[j] else -1.0 for j in range(self.dim)]
        return self.bias + sum(self.w[j] * agreement[j] for j in range(self.dim))

    def probabilities(self, ep: Episode, temperature: float = 0.5) -> list[float]:
        scores = [self.score(ep.query, c) / max(temperature, 1e-6) for c in ep.candidates]
        m = max(scores)
        exps = [math.exp(s - m) for s in scores]
        z = sum(exps)
        return [x / z for x in exps]

    def select(self, ep: Episode, rng: random.Random, temperature: float = 0.5) -> tuple[int, list[float]]:
        p = self.probabilities(ep, temperature)
        u = rng.random()
        acc = 0.0
        for i, pi in enumerate(p):
            acc += pi
            if u <= acc:
                return i, p
        return len(p) - 1, p

    def update(self, ep: Episode, selected: int, reward: float, probs: list[float]) -> None:
        # REINFORCE on a scalar reward. The gold index is never consulted.
        centered = reward - 1.0 / len(ep.candidates)
        for j in range(self.dim):
            agreement = 1.0 if ep.query[j] == ep.candidates[selected].descriptor[j] else -1.0
            self.w[j] += self.lr * centered * agreement * (1.0 - probs[selected])
        self.bias += self.lr * centered * (1.0 - probs[selected])

def environment_success(ep: Episode, selected: int) -> bool:
    return selected == ep.gold_index

def intervention_effect(ep: Episode, selected: int) -> float:
    before = float(environment_success(ep, selected))
    alternative = (selected + 1) % len(ep.candidates)
    after = float(environment_success(ep, alternative))
    return before - after

def evaluate(router: LinearBanditRouter, episodes: Sequence[Episode], seed: int,
             temperature: float = 0.15) -> dict:
    rng = random.Random(seed)
    rows = []
    for ep in episodes:
        selected, probs = router.select(ep, rng, temperature)
        success = float(environment_success(ep, selected))
        router.update(ep, selected, success, probs)
        rows.append({
            "success": success,
            "selected": selected,
            "candidate_count": len(ep.candidates),
            "intervention_effect": intervention_effect(ep, selected),
        })
    return {
        "success_mean": sum(r["success"] for r in rows) / len(rows),
        "intervention_effect_mean": sum(r["intervention_effect"] for r in rows) / len(rows),
        "rows": rows,
    }

def run(train_episodes: int = 2000, test_episodes: int = 300,
        n_candidates: int = 8, dim: int = 8, noise: float = 0.10,
        train_seed: int = 0, test_seed: int = 10_000) -> dict:
    train = [make_episode(train_seed + i, n_candidates, dim, noise) for i in range(train_episodes)]
    test = [make_episode(test_seed + i, n_candidates, dim, noise) for i in range(test_episodes)]
    router = LinearBanditRouter(dim)
    train_result = evaluate(router, train, train_seed)
    # Freeze router for held-out evaluation.
    rng = random.Random(test_seed)
    test_rows = []
    for ep in test:
        selected, probs = router.select(ep, rng)
        success = float(environment_success(ep, selected))
        test_rows.append({
            "success": success,
            "selected": selected,
            "candidate_count": n_candidates,
            "intervention_effect": intervention_effect(ep, selected),
        })
    return {
        "protocol": "PNDS-GATE-001",
        "stage": 2,
        "training": {"episodes": train_episodes, "seed": train_seed},
        "evaluation": {"episodes": test_episodes, "seed": test_seed},
        "environment": {"candidates": n_candidates, "dim": dim, "noise": noise},
        "training_result": {k: v for k, v in train_result.items() if k != "rows"},
        "test_success_mean": sum(r["success"] for r in test_rows) / len(test_rows),
        "test_intervention_effect_mean": sum(r["intervention_effect"] for r in test_rows) / len(test_rows),
        "test_rows": test_rows,
    }

if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
