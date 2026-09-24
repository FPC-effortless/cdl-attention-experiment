"""PNDS-GATE-001 Stage 2c: relational relevance.

Changes exactly one thing relative to Stage 2b: the source of relevance.

Stage 2b:
    R(q, c) ~ feature agreement(q, c)
The gold candidate's descriptor is a noisy copy of the query, so a fixed
bit-agreement scorer is near-optimal and the learned router cannot beat it.

Stage 2c:
    R(q, c, x) = agreement(q, c) restricted to the positions x marks
The context is no longer inert. It selects a small set of "marked" positions, and
relevance is defined *only* on those positions: the gold candidate matches the
query on every marked position, and deliberately mismatches the query on the
unmarked ones. Neither pairwise input determines relevance:

  - (q, c) alone is not enough: which positions count is decided by x.
  - (x, c) alone is not enough: the required values come from q.
  - total query-agreement is actively misleading: the gold candidate has *low*
    total agreement with the query, because it anti-matches on the unmarked
    positions, while the query-deceptive distractors agree strongly with the
    query everywhere except the marked positions.

This is the decisive control: the static scorer from Stage 2b is unchanged and
scores total query-agreement, so it must collapse here, and the question the
experiment answers is whether outcome training recovers the context-gated rule.

Representability note. The relation is deliberately kept linearly representable
in the router's hand-designed feature basis, by including one context-gated
query-agreement feature per position:

    gated_j = x_j * [q_j == c_j]

Setting every gated weight to +1 recovers the relation exactly, so the correct
solution lives in the hypothesis class. What makes it hard is that relevance is a
*conditional* rule -- which features matter is decided per episode by the context
-- so a fixed similarity rule (the static arm) cannot express it. The experiment
therefore tests learning from outcomes, not model-class expressiveness.

Everything else is identical to Stage 2b: the same four arms, the same
train/test methodology, the same candidate counts, the same evaluation
discipline and metrics.

Deliberately adds no persistence, verifier, recurrence, or new architecture.
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
    context: tuple[int, ...]
    candidates: tuple[Candidate, ...]
    gold_index: int  # evaluator/environment only; never exposed to the router


def _rand_bits(rng: random.Random, dim: int) -> tuple[int, ...]:
    return tuple(rng.randrange(2) for _ in range(dim))


def _n_marked(dim: int) -> int:
    """How many positions the context marks. Scales with dim, fixed across episodes."""
    return max(2, dim // 4)


def marked_positions(context: tuple[int, ...], dim: int) -> tuple[int, ...]:
    """The positions the context marks. Fully observable: context bit == 1."""
    return tuple(j for j in range(dim) if context[j] == 1)


def satisfies_relation(query: tuple[int, ...], context: tuple[int, ...],
                       descriptor: tuple[int, ...], dim: int) -> bool:
    """The relevance relation, exactly as the environment defines it.

    Public and observable so that tests can verify gold is the unique satisfier
    without consulting ``gold_index`` to *construct* anything.
    """
    return all(descriptor[j] == query[j] for j in marked_positions(context, dim))


def make_episode(seed: int, n_candidates: int = 8, dim: int = 8, noise: float = 0.10) -> Episode:
    """Context-gated relational relevance.

    The context marks ``n_marked`` positions (its 1 bits). The gold candidate
    matches the query on every marked position and anti-matches the query on the
    unmarked positions with probability ``1 - noise``.

    Distractor modes, each forced to violate the relation on at least one marked
    position so that gold is the *unique* satisfier:
      - query-deceptive: agrees strongly with the query on unmarked positions but
        violates every marked position. Beats the static scorer, fails the relation.
      - near-miss: matches the query on all but one marked position.
      - random: random bits, then forced to violate one marked position.
    """
    rng = random.Random(seed)

    query = _rand_bits(rng, dim)

    # The context marks exactly n_marked positions, chosen per episode. Which
    # positions are marked is the relational variable the router must condition on.
    n_marked = _n_marked(dim)
    marks = rng.sample(range(dim), n_marked)
    context_list = [0] * dim
    for j in marks:
        context_list[j] = 1
    context = tuple(context_list)

    # Gold: satisfies the relation on the marked positions, and anti-matches the
    # query elsewhere so that *total* query agreement is uninformative.
    gold_desc: list[int] = []
    for j in range(dim):
        if j in marks:
            gold_desc.append(query[j])
        else:
            gold_desc.append(1 - query[j] if rng.random() < 1.0 - noise else rng.randrange(2))
    gold_desc_t = tuple(gold_desc)

    def agree_with(target: tuple[int, ...], strength: float) -> list[int]:
        d = list(_rand_bits(rng, dim))
        for j in range(dim):
            if rng.random() < strength:
                d[j] = target[j]
        return d

    def force_violation(desc: list[int]) -> list[int]:
        """Guarantee `desc` fails the relation on one marked position.

        Without this the random modes can accidentally satisfy the relation, which
        would make the gold candidate non-unique and cap the oracle arm below 1.0.
        The environment must be unambiguous for the control to be interpretable.
        """
        j = rng.choice(marks)
        desc[j] = 1 - query[j]
        return desc

    candidates: list[Candidate] = []
    gold = Candidate(
        key=f"key_{rng.randrange(1_000_000)}",
        descriptor=gold_desc_t,
        action=0,
    )
    candidates.append(gold)

    made = 1
    guard = 0
    while made < n_candidates and guard < 100 * n_candidates:
        guard += 1
        mode = rng.randrange(3)
        if mode == 0:
            # Query-deceptive: looks similar to the query overall, fails the relation.
            desc = agree_with(query, 0.9)
            for j in marks:
                desc[j] = 1 - query[j]
        elif mode == 1:
            # Near-miss: right on all but one marked position.
            desc = list(_rand_bits(rng, dim))
            miss = rng.choice(marks)
            for j in marks:
                desc[j] = 1 - query[j] if j == miss else query[j]
        else:
            # Random, then forced to violate the relation.
            desc = force_violation(list(_rand_bits(rng, dim)))
        desc_t = tuple(desc)
        if desc_t == gold_desc_t:
            continue
        candidates.append(Candidate(
            key=f"key_{rng.randrange(1_000_000)}",
            descriptor=desc_t,
            action=made,
        ))
        made += 1

    while made < n_candidates:
        candidates.append(Candidate(
            key=f"key_{rng.randrange(1_000_000)}",
            descriptor=tuple(force_violation(list(_rand_bits(rng, dim)))),
            action=made,
        ))
        made += 1

    # Randomize candidate order only after identifying the gold candidate.
    candidates = list(candidates)
    rng.shuffle(candidates)
    gold_index = next(i for i, c in enumerate(candidates) if c.action == 0)
    return Episode(query=query, context=context, candidates=tuple(candidates), gold_index=gold_index)


def environment_success(ep: Episode, selected: int) -> bool:
    return selected == ep.gold_index


# --------------------------------------------------------------------------- #
# Arms
# --------------------------------------------------------------------------- #

def static_select(ep: Episode, dim: int) -> int:
    """The *same* fixed total-agreement rule used in Stage 2b.

    The key control: the static scorer is unchanged, so if it collapses here it
    is because the environment no longer rewards total feature agreement. In
    Stage 2c the gold candidate deliberately anti-matches the query on the
    unmarked positions, so this rule is actively anti-correlated with relevance.
    """
    scores = [sum(1.0 if ep.query[j] == c.descriptor[j] else 0.0 for j in range(dim))
              for c in ep.candidates]
    best = max(scores)
    return scores.index(best)


def random_select(ep: Episode, rng: random.Random) -> int:
    return rng.randrange(len(ep.candidates))


class RelationalBanditRouter:
    """Outcome-trained scorer over query/context/candidate features.

    Deliberately kept as a small linear model over a *hand-designed feature
    basis* so that the experiment isolates "can relevance be learned from
    outcomes" rather than "is this model class expressive". The basis contains
    one context-gated query-agreement feature per position, which is exactly
    what the Stage 2c relation is built from, so the correct solution lives in
    the hypothesis class.

    Nothing here is given the gold index, the relation, or the answer.
    """

    def __init__(self, dim: int, lr: float = 0.05):
        self.dim = dim
        self.lr = lr
        # Basis: [bias, q_j==c_j, x_j==c_j, x_j==q_j, x_j * (q_j==c_j)]  (each per j)
        self.n_bias = 1
        self.n_qc = dim
        self.n_xc = dim
        self.n_xq = dim
        self.n_gated = dim
        self.n = self.n_bias + self.n_qc + self.n_xc + self.n_xq + self.n_gated
        self.w = [0.0] * self.n

    def _features(self, query: tuple[int, ...], context: tuple[int, ...],
                  descriptor: tuple[int, ...]) -> list[float]:
        dim = self.dim
        feats = [1.0]
        feats += [1.0 if query[j] == descriptor[j] else -1.0 for j in range(dim)]
        feats += [1.0 if context[j] == descriptor[j] else -1.0 for j in range(dim)]
        feats += [1.0 if context[j] == query[j] else -1.0 for j in range(dim)]
        # The relation feature: query-agreement gated on by the context marker.
        # Zero on unmarked positions, +-1 on marked ones.
        feats += [float(context[j]) * (1.0 if query[j] == descriptor[j] else -1.0)
                  for j in range(dim)]
        return feats

    def score(self, query, context, descriptor) -> float:
        feats = self._features(query, context, descriptor)
        return sum(w * f for w, f in zip(self.w, feats))

    def probabilities(self, ep: Episode, temperature: float = 0.5) -> list[float]:
        scores = [self.score(ep.query, ep.context, c.descriptor) / max(temperature, 1e-6)
                  for c in ep.candidates]
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
        feats = self._features(ep.query, ep.context, ep.candidates[selected].descriptor)
        grad = self.lr * centered * (1.0 - probs[selected])
        for k in range(self.n):
            self.w[k] += grad * feats[k]


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def _train_router(train_seeds: Sequence[int], n_candidates: int, dim: int, noise: float,
                  train_seed: int, lr: float) -> RelationalBanditRouter:
    router = RelationalBanditRouter(dim, lr=lr)
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
    rng = random.Random(eval_seed)

    rows = []
    for s in test_seeds:
        ep = make_episode(s, n_candidates, dim, noise)
        arms: dict[str, int] = {
            "learned": router.select(ep, random.Random(s + 1_000_000), temperature=0.05)[0],
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
        "stage": "2c",
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
