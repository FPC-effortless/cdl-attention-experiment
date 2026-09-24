"""PNDS-GATE-001 Stage 3a: weight-lesion test of the Stage 2c learned mechanism.

This is a mechanism-ablation test, not a causal-intervention test. It asks one
question:

    Are the learned weights actually necessary for the learned behaviour?

Stage 2c showed the learned router solves a context-conditioned relation the
fixed similarity scorer cannot express, and that its converged weights form a
signed signature of the intended rule: context-gated agreement positive,
total agreement negative, offset strongly positive. That is correlational
evidence from a converged weight vector. This stage tests it by damaging the
weights and measuring the behavioural consequence.

Scope discipline: the router is trained ONCE on Stage 2c episodes and frozen.
Lesions are applied to copies of the frozen weights and evaluated on the same
held-out test episodes. Nothing is retrained after lesioning. The question is
whether the behaviour was *dependent* on the mechanism, so retraining would
destroy the question.

Why the naive single-block lesion is uninterpretable here
---------------------------------------------------------

The learned router has two blocks that both encode agreement:

    gated_j = x_j * [q_j == c_j]      (weight ~ +1.4, the 2c relation feature)
    qc_j    = [q_j == c_j]            (weight ~ -1.6, total agreement)

and the Stage 2c environment is constructed so that the gold candidate
deliberately *anti-matches* the query on unmarked positions. Consequence:
"pick the candidate with the least total query agreement" is by itself a
0.55-scoring shortcut on this environment. So lesioning the gated block alone
leaves the qc block behind as a functioning anti-agreement scorer, and the
lesion effect is confounded with that shortcut. Conversely, lesioning the qc
block alone does not degrade the router at all -- it *improves* it to 1.0,
because the qc block is the only component working against the correct rule.

A single-block lesion therefore cannot isolate the mechanism. The correct
necessity test is the *joint* lesion: destroy both the gated signal and the
anti-agreement block, and confirm the behaviour collapses.

The five arms are intact / joint-lesion / gated-only / qc-only / anti-static,
all on identical held-out episodes, so the numbers are directly comparable.

Deliberately adds no persistence, recurrence, verifier, or new architecture.
This stage is still a bandit; it makes no causal-intervention claim.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
import math
import random
from typing import Sequence

from .stage2c_relational import (
    make_episode, environment_success, static_select, random_select,
    RelationalBanditRouter,
)


@dataclass(frozen=True)
class LesionSpec:
    """A named mutation of a frozen router's weight vector."""
    name: str
    gated: float = 1.0          # multiplier applied to the gated block
    gated_shift: float = 0.0    # additive shift applied to the gated block
    qc: float = 1.0             # multiplier applied to the total-agreement block
    qc_shift: float = 0.0       # additive shift applied to the qc block


def apply_lesion(router: RelationalBanditRouter, spec: LesionSpec) -> RelationalBanditRouter:
    """Return a router with the same frozen weights, minus the lesioned blocks.

    Never mutates the input: lesioned routers must not contaminate the intact
    reference, and the intact router must remain the single trained object.
    """
    dim = router.dim
    off_qc = 1
    off_gated = 1 + 3 * dim
    clone = RelationalBanditRouter(dim, lr=router.lr)
    clone.w = list(router.w)
    for j in range(off_gated, off_gated + dim):
        clone.w[j] = clone.w[j] * spec.gated + spec.gated_shift
    for j in range(off_qc, off_qc + dim):
        clone.w[j] = clone.w[j] * spec.qc + spec.qc_shift
    return clone


def anti_static_select(ep: Episode_t, dim: int) -> int:
    """The *anti*-agreement shortcut: pick the least query-agreeing candidate.

    This is a necessary fifth arm. Because the Stage 2c gold candidate
    deliberately anti-matches the query on unmarked positions, this rule
    scores well above chance on its own, and it is what the qc block
    degenerates into when the gated block is lesioned. Without it the
    gated-only lesion number is uninterpretable.
    """
    scores = [sum(1.0 if ep.query[j] == c.descriptor[j] else 0.0 for j in range(dim))
              for c in ep.candidates]
    worst = min(scores)
    return scores.index(worst)


# Keep a type alias so the annotation above resolves without a circular import.
Episode_t = "object"


def train_frozen(train_seeds: Sequence[int], n_candidates: int, dim: int, noise: float,
                 train_seed: int, lr: float) -> RelationalBanditRouter:
    """Train the router once and freeze it. Lesions act on copies of this object."""
    router = RelationalBanditRouter(dim, lr=lr)
    order = list(train_seeds)
    random.Random(train_seed).shuffle(order)
    for s in order:
        ep = make_episode(s, n_candidates, dim, noise)
        idx, probs = router.select(ep, random.Random(s), temperature=0.5)
        reward = float(environment_success(ep, idx))
        router.update(ep, idx, reward, probs)
    return router


def evaluate_lesions(
    *,
    train_seeds: Sequence[int],
    test_seeds: Sequence[int],
    n_candidates: int = 8,
    dim: int = 8,
    noise: float = 0.10,
    train_seed: int = 0,
    eval_seed: int = 1,
    lr: float = 0.05,
    lesions: Sequence[LesionSpec] | None = None,
) -> dict:
    """All arms on identical held-out episodes. The router is frozen, then lesioned."""
    if lesions is None:
        lesions = DEFAULT_LESIONS

    router = train_frozen(train_seeds, n_candidates, dim, noise, train_seed, lr)
    rng = random.Random(eval_seed)
    lesioned = {spec.name: apply_lesion(router, spec) for spec in lesions}

    rows = []
    for s in test_seeds:
        ep = make_episode(s, n_candidates, dim, noise)
        arms: dict[str, int] = {
            "intact": router.select(ep, random.Random(s + 1_000_000), temperature=0.05)[0],
            "static": static_select(ep, dim),
            "anti_static": anti_static_select(ep, dim),
            "random": random_select(ep, rng),
            "oracle": ep.gold_index,
        }
        for spec in lesions:
            arms[spec.name] = lesioned[spec.name].select(
                ep, random.Random(s + 1_000_000), temperature=0.05)[0]
        for name, idx in arms.items():
            rows.append({
                "seed": s,
                "arm": name,
                "selected": idx,
                "gold_index": ep.gold_index if name == "oracle" else None,
                "top1": float(idx == ep.gold_index),
                "success": float(environment_success(ep, idx)),
            })

    arm_order = ["intact"] + [spec.name for spec in lesions] + \
        ["static", "anti_static", "random", "oracle"]
    summary: dict[str, dict[str, float]] = {}
    for arm in arm_order:
        subset = [r for r in rows if r["arm"] == arm]
        n = len(subset)
        summary[arm] = {
            "n_episodes": n,
            "top1_mean": sum(r["top1"] for r in subset) / n,
            "success_mean": sum(r["success"] for r in subset) / n,
            "chance": 1.0 / n_candidates,
        }

    summary["lesion_effects"] = {
        f"intact_minus_{spec.name}":
            summary["intact"]["success_mean"] - summary[spec.name]["success_mean"]
        for spec in lesions
    }

    return {
        "protocol": "PNDS-GATE-001",
        "stage": "3a",
        "config": {
            "train_episodes": len(train_seeds),
            "test_episodes": len(test_seeds),
            "candidates": n_candidates,
            "dim": dim,
            "noise": noise,
            "train_seed": train_seed,
            "eval_seed": eval_seed,
            "lr": lr,
            "lesions": [spec.name for spec in lesions],
        },
        "frozen_weights": {
            "bias": router.w[0],
            "qc": list(router.w[1:1 + dim]),
            "xc": list(router.w[1 + dim:1 + 2 * dim]),
            "xq": list(router.w[1 + 2 * dim:1 + 3 * dim]),
            "gated": list(router.w[1 + 3 * dim:1 + 4 * dim]),
        },
        "summary": summary,
        "rows": rows,
    }


DEFAULT_LESIONS: tuple[LesionSpec, ...] = (
    # The joint lesion: destroy both the gated signal and the anti-agreement
    # block. This is the interpretable necessity test; each alone is not.
    LesionSpec(name="joint", gated=0.0, qc=0.0),
    # Gated only: confounded, reported for transparency. It degenerates the
    # qc block into the anti_static shortcut, so it must NOT be read as the
    # mechanism effect.
    LesionSpec(name="gated_only", gated=0.0),
    # qc only: the surprising one. Removing the anti-agreement block does not
    # degrade the router; it improves it, because that block is the residual
    # interference limiting the intact model below the relation oracle.
    LesionSpec(name="qc_only", qc=0.0),
    # Sign-flip of the gated block: the harsher test, which has no benign
    # fallback because it actively rewards marked-position disagreement.
    LesionSpec(name="gated_flip", gated=-1.0),
)


def run(train_episodes: int = 2000, test_episodes: int = 300,
        n_candidates: int = 8, dim: int = 8, noise: float = 0.10,
        train_seed: int = 0, eval_seed: int = 1) -> dict:
    train_seeds = list(range(train_seed, train_seed + train_episodes))
    test_seeds = list(range(10_000, 10_000 + test_episodes))
    return evaluate_lesions(
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
