"""PNDS-GATE-001 Stage 3b: persistent state and the first causal intervention.

This is the first stage in which a persistent state S_t exists to be read and
damaged. Stages 2b/2c/3a were all bandits: stateless, i.i.d. episodes, one
decision per episode, no memory. None of them could address the PNDS loop
claim, because no S_t existed to intervene upon.

    S_t -> R_t -> C_t -> A_t -> O_t -> V_t -> S_{t+1}

Here S_t is a stream-level latent that is stable across many episodes within a
stream and resampled between streams. The router must READ it to route.

Corrected environment (Amendment 1)
-----------------------------------

The environment registered in 3a03225 was defective: a router that read the
target t and ignored the key K scored 0.6965 at 8 candidates, because gold's
total t-agreement was systematically higher than every distractor's. That is
failure mode B (state redundancy) -- the registered design checked that the
QUERY could not leak the key, but never checked that t read WITHOUT the key
was already sufficient. It was.

The correction makes the principal stateless shortcut structurally impossible
rather than empirically unlikely:

    for every candidate c:   sum_j 1[c_j == t_j] == T

so f_total-agreement(c, t) = T for all candidates, and the score vector is
CONSTANT. A t-only router provably cannot separate gold from distractors. The
per-position marginals equalise at T/dim for gold and distractors alike, so
the leak cannot reappear through a different statistic.

What the corrected design tests
-------------------------------

The original preregistration tested whether the state contained useful target
information. It did. That is not the PNDS hypothesis.

The corrected design tests whether the decision mechanism must access the
KEY-INDEXED persistent state, because the target itself has been made
non-discriminative under every observable feature statistic. The only
separating quantity is

    sum_{j in K} 1[c_j == t_j]

which is a function of (c, t, K), and K exists nowhere outside S_t.

Deliberately does not implement verification, repair, or S_{t+1} learning.
The state is read and damaged, not updated from outcomes. Protocol section 29
(verified-only commit falsification) is a separate experiment; conflating it
with the first intervention would make both uninterpretable.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import json
import math
import random
from typing import Sequence

# Import the two invariants so a generator change cannot silently drop them.
from .test_stage3b_invariants import (
    exact_agreement_total, per_position_marginal,
)


@dataclass(frozen=True)
class PersistentState:
    """S_t. Stream-level latent; never part of any per-episode observation.

    ``target`` and ``key`` are drawn once per stream and held fixed across many
    episodes. They appear in no observation field; a router receives them only
    through an explicit routing read.
    """
    target: tuple[int, ...]
    key: frozenset[int]

    @property
    def k(self) -> int:
        return len(self.key)


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
    gold_index: int          # environment only; never exposed to the router
    stream_id: int


def default_T(dim: int, k: int) -> int:
    """Agreement total every candidate must hit.

    Must satisfy  k <= T <= dim  and  T - k <= dim - k  and  T - (k-1) <= dim - k,
    i.e. it must be achievable by gold (k key matches + T-k unmarked matches)
    and by a maximal distractor (k-1 key matches + T-(k-1) unmarked matches).
    """
    return k + max(1, (dim - k) // 2)


def _feasible_a(state: PersistentState, dim: int, T: int) -> list[int]:
    """Key-agreement counts a distractor may take: 0..k-1, subject to the
    unmarked positions being able to absorb the remainder T - a."""
    k = state.k
    n_unmarked = dim - k
    return [a for a in range(0, k) if 0 <= T - a <= n_unmarked]


def make_stream(seed: int, dim: int = 8, k: int = 4) -> PersistentState:
    """Draw the stream-level latent. Called ONCE per stream, not per episode."""
    rng = random.Random(seed)
    target = tuple(rng.randrange(2) for _ in range(dim))
    key = frozenset(rng.sample(range(dim), k))
    return PersistentState(target=target, key=key)


def make_episode_in_stream(seed: int, state: PersistentState, dim: int = 8,
                           n_candidates: int = 8, T: int | None = None) -> Episode:
    """One episode within a stream.

    Gold and every distractor have EXACTLY T agreements with the target t.
    Gold places all k of its key-position agreements on the key; each
    distractor takes a_d <= k-1 key agreements and makes the remainder up on
    unmarked positions. Gold is therefore the unique satisfier of the relation

        relevance(c)  <=>  all( c_j == t_j  for j in K )

    while being statistically indistinguishable from distractors under any
    total- or per-position statistic.
    """
    if T is None:
        T = default_T(dim, state.k)
    k = state.k
    if not (k <= T <= dim):
        raise ValueError(f"T={T} not achievable with dim={dim}, k={k}")
    t = state.target
    key = sorted(state.key)
    off = [j for j in range(dim) if j not in state.key]

    rng = random.Random(seed)

    # Query is a pure nuisance variable: independent of (t, K).
    query = tuple(rng.randrange(2) for _ in range(dim))

    # Gold: all k key positions agree, plus exactly T-k unmarked agreements.
    gold = [0] * dim
    for j in key:
        gold[j] = t[j]
    chosen_off = rng.sample(off, T - k)
    for j in off:
        gold[j] = t[j] if j in chosen_off else 1 - t[j]

    feas = _feasible_a(state, dim, T)
    if k - 1 not in feas:
        raise ValueError("no distractor can both hit T and violate the key")

    cands: list[tuple[int, ...]] = [tuple(gold)]
    seen = {tuple(gold)}
    made, guard = 1, 0
    while made < n_candidates and guard < 2000 * n_candidates:
        guard += 1
        a_d = rng.choice(feas)
        if a_d >= k:
            a_d = k - 1            # guarantee >=1 key violation
        need = T - a_d

        d = [0] * dim
        kpos = list(key)
        rng.shuffle(kpos)
        for j in kpos[:a_d]:
            d[j] = t[j]
        for j in kpos[a_d:]:
            d[j] = 1 - t[j]

        opos = list(off)
        rng.shuffle(opos)
        for j in opos[:need]:
            d[j] = t[j]
        for j in opos[need:]:
            d[j] = 1 - t[j]

        dt = tuple(d)
        if dt in seen:
            continue
        seen.add(dt)
        cands.append(dt)
        made += 1

    # Randomize candidate order AFTER identifying gold, then locate it.
    order = list(range(len(cands)))
    rng.shuffle(order)
    shuffled = [cands[i] for i in order]
    gold_desc = tuple(gold)
    gold_index = next(i for i, c in enumerate(shuffled) if c == gold_desc)

    candidates = tuple(
        Candidate(key=f"key_{rng.randrange(1_000_000)}", descriptor=d, action=i)
        for i, d in enumerate(shuffled)
    )
    return Episode(
        query=query,
        candidates=candidates,
        gold_index=gold_index,
        stream_id=seed,
    )


def satisfies_relation(descriptor: tuple[int, ...], state: PersistentState) -> bool:
    """The relevance relation, public and observable so tests can verify gold is
    the unique satisfier without consulting gold_index to construct anything."""
    return all(descriptor[j] == state.target[j] for j in state.key)


def environment_success(ep: Episode, selected: int) -> bool:
    return selected == ep.gold_index


# --------------------------------------------------------------------------- #
# Stateless arms. None of these reads S_t.
# --------------------------------------------------------------------------- #

def _tied_index(scores: list[float], want_max: bool, rng: random.Random) -> int:
    """Random tie-breaking under a seeded RNG (design rule R5, 3a03225 s3.5).

    Without this, scores.index(max) returns the FIRST index among ties and
    manufactures a position artifact. The corrected environment has many ties
    by construction, so this matters.
    """
    best = max(scores) if want_max else min(scores)
    tied = [i for i, v in enumerate(scores) if v == best]
    return rng.choice(tied)


def t_static_select(ep: Episode, state: PersistentState, rng: random.Random) -> int:
    """Reads the target t but IGNORES the key K.

    This is the arm that exposed the registered defect. Under the corrected
    environment every candidate has exactly T agreements with t, so this score
    vector is constant and the arm is at chance -- provably, not approximately.
    """
    dim = len(ep.query)
    scores = [sum(1.0 for j in range(dim) if c.descriptor[j] == state.target[j])
              for c in ep.candidates]
    return _tied_index(scores, True, rng)


def static_select(ep: Episode, dim: int, rng: random.Random) -> int:
    """The Stage 2b/2c fixed rule: total agreement with the QUERY.

    The query is a pure nuisance variable here, independent of (t, K), so this
    arm must be at chance. Retained because it is the arm that collapsed to
    0.0000 in Stage 2c and because it anchors the control discipline.
    """
    scores = [sum(1.0 if ep.query[j] == c.descriptor[j] else 0.0 for j in range(dim))
              for c in ep.candidates]
    return _tied_index(scores, True, rng)


def anti_static_select(ep: Episode, dim: int, rng: random.Random) -> int:
    """The anti-agreement shortcut that scored 0.5567 in Stage 3a. Here it must
    be at chance, because total query agreement carries no information."""
    scores = [sum(1.0 if ep.query[j] == c.descriptor[j] else 0.0 for j in range(dim))
              for c in ep.candidates]
    return _tied_index(scores, False, rng)


def random_select(ep: Episode, rng: random.Random) -> int:
    return rng.randrange(len(ep.candidates))


def relation_oracle_select(ep: Episode, state: PersistentState) -> int:
    """Follows the relation using the TRUE state. Ceiling reference only.

    This arm KNOWS the relation. It is not a competitor for the learned arm;
    it measures whether the environment is unambiguous.
    """
    hits = [i for i, c in enumerate(ep.candidates)
            if satisfies_relation(c.descriptor, state)]
    if not hits:
        return random_select(ep, random.Random(0))
    return hits[0]


# --------------------------------------------------------------------------- #
# The learned router.
# --------------------------------------------------------------------------- #

class PersistentStateRouter:
    """Outcome-trained scorer over (query, candidate, S_t) features.

    Deliberately kept as a small linear model over a FIXED feature basis, so
    the experiment isolates "can key-indexed routing be learned from outcomes"
    rather than "is this model class expressive".

    The basis is KEY-RELATIVE, one dim-length block per possible key:

        phi_{K', i}(c, t) = +[c_i == t_i]    if i in K'
                          = -[c_i == t_i]    if i not in K'

    For the true key K the block's linear score is

        sum_{i in K} agree_i  -  sum_{i not in K} agree_i  =  a - (T - a) = 2a - T

    where a is the key-agreement count. Gold has a = k, every distractor has
    a <= k - 1, so the margin is exactly 2 and the relation is representable
    for every k. (A naive "+1 on key positions, 0 elsewhere" basis does NOT
    work and produced an earlier wrong conclusion: summed over all keys it
    reduces to C(dim-1,k-1) * total-agreement, the constant C*T, and cannot
    separate anything. The negative features are load-bearing, not cosmetic.)

    Nothing here is given the gold index, the relation, or the answer.
    """

    def __init__(self, dim: int, k: int, all_keys: Sequence[frozenset[int]],
                 lr: float = 0.05):
        self.dim = dim
        self.k = k
        self.lr = lr
        self.all_keys = [frozenset(x) for x in all_keys]
        self.key_index = {kk: i for i, kk in enumerate(self.all_keys)}
        self.n_bias = 1
        self.n_keyed = len(self.all_keys) * dim
        self.n = self.n_bias + self.n_keyed
        self.w = [0.0] * self.n

    def _features(self, query, descriptor, state: PersistentState) -> list[float]:
        """One dim-length block per possible key; only the state's key block is
        active. Within the active block, key positions get +agreement and
        non-key positions get -agreement."""
        dim = self.dim
        t = state.target
        agree = [1.0 if descriptor[j] == t[j] else -1.0 for j in range(dim)]
        feats = [1.0]
        for kk in self.all_keys:
            feats.extend(agree[j] if j in kk else -agree[j] for j in range(dim))
        return feats

    def score(self, query, descriptor, state: PersistentState) -> float:
        feats = self._features(query, descriptor, state)
        return sum(w * f for w, f in zip(self.w, feats))

    def probabilities(self, ep: Episode, state: PersistentState,
                      temperature: float = 0.5) -> list[float]:
        scores = [self.score(ep.query, c.descriptor, state) / max(temperature, 1e-6)
                  for c in ep.candidates]
        m = max(scores)
        exps = [math.exp(s - m) for s in scores]
        z = sum(exps)
        return [x / z for x in exps]

    def select(self, ep: Episode, state: PersistentState, rng: random.Random,
               temperature: float = 0.5) -> tuple[int, list[float]]:
        p = self.probabilities(ep, state, temperature)
        u = rng.random()
        acc = 0.0
        for i, pi in enumerate(p):
            acc += pi
            if u <= acc:
                return i, p
        return len(p) - 1, p

    def update(self, ep: Episode, state: PersistentState, selected: int,
               reward: float, probs: list[float]) -> None:
        # REINFORCE on a scalar reward. The gold index is never consulted.
        centered = reward - 1.0 / len(ep.candidates)
        feats = self._features(ep.query, ep.candidates[selected].descriptor, state)
        grad = self.lr * centered * (1.0 - probs[selected])
        for kk in range(self.n):
            self.w[kk] += grad * feats[kk]


class StatelessRouter:
    """learned_no_state: same training budget, S_t features withheld.

    This is the arm that makes Stage 3b a STATE test rather than a model test.
    It sees only (query, candidate). Since the query is independent of (t, K)
    and every candidate has identical total t-agreement, its best achievable
    score is provably chance. If ``learned`` does not beat this arm, the
    persistent state contributes nothing measurable regardless of how well it
    scores.
    """

    def __init__(self, dim: int, lr: float = 0.05):
        self.dim = dim
        self.lr = lr
        self.n_bias = 1
        self.n_qc = dim
        self.n = self.n_bias + self.n_qc
        self.w = [0.0] * self.n

    def _features(self, query, descriptor) -> list[float]:
        return [1.0] + [1.0 if query[j] == descriptor[j] else -1.0
                        for j in range(self.dim)]

    def probabilities(self, ep: Episode, temperature: float = 0.5) -> list[float]:
        scores = [sum(w * f for w, f in zip(self.w, self._features(ep.query, c.descriptor)))
                  / max(temperature, 1e-6) for c in ep.candidates]
        m = max(scores)
        exps = [math.exp(s - m) for s in scores]
        z = sum(exps)
        return [x / z for x in exps]

    def select(self, ep: Episode, rng: random.Random,
               temperature: float = 0.5) -> tuple[int, list[float]]:
        p = self.probabilities(ep, temperature)
        u = rng.random()
        acc = 0.0
        for i, pi in enumerate(p):
            acc += pi
            if u <= acc:
                return i, p
        return len(p) - 1, p

    def update(self, ep: Episode, selected: int, reward: float,
               probs: list[float]) -> None:
        centered = reward - 1.0 / len(ep.candidates)
        feats = self._features(ep.query, ep.candidates[selected].descriptor)
        grad = self.lr * centered * (1.0 - probs[selected])
        for kk in range(self.n):
            self.w[kk] += grad * feats[kk]


# --------------------------------------------------------------------------- #
# The intervention.
# --------------------------------------------------------------------------- #

def replace_key(state: PersistentState, rng: random.Random, dim: int,
                disjoint: bool = False) -> PersistentState:
    """do(K_t <- K'_t): replace the key with an independent key of the same size.

    The router reads the corrupted state and routes against it. Because t can
    no longer independently identify gold, this destroys the ONLY separating
    signal the environment contains.
    """
    if disjoint:
        pool = [j for j in range(dim) if j not in state.key]
    else:
        pool = list(range(dim))
    if disjoint and len(pool) < state.k:
        raise ValueError("cannot form a disjoint key")
    return PersistentState(
        target=state.target,
        key=frozenset(rng.sample(pool, state.k)),
    )


def corrupt_target(state: PersistentState, rng: random.Random,
                   frac: float) -> PersistentState:
    """do(t <- t'): flip a fraction of the target bits. Secondary diagnostic arm."""
    t = list(state.target)
    n_flip = int(round(frac * len(t)))
    for j in rng.sample(range(len(t)), n_flip):
        t[j] = 1 - t[j]
    return PersistentState(target=tuple(t), key=state.key)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def all_keys(dim: int, k: int) -> list[frozenset[int]]:
    from itertools import combinations
    return [frozenset(c) for c in combinations(range(dim), k)]


def _train(train_seeds: Sequence[int], dim: int, k: int, n_candidates: int,
           T: int, train_seed: int, lr: float,
           keys: Sequence[frozenset[int]]) -> tuple[PersistentStateRouter, StatelessRouter]:
    """Train on streams. Each stream has its own latent; episodes within a
    stream share it, which is what makes the state worth reading."""
    router = PersistentStateRouter(dim, k, keys, lr=lr)
    no_state = StatelessRouter(dim, lr=lr)
    order = list(train_seeds)
    random.Random(train_seed).shuffle(order)
    for s in order:
        state = make_stream(s, dim, k)
        # A short stream per latent keeps episodes correlated with their state.
        for e in range(8):
            ep = make_episode_in_stream(s * 100 + e, state, dim, n_candidates, T)
            idx, probs = router.select(ep, state, random.Random(s * 100 + e), temperature=0.5)
            reward = float(environment_success(ep, idx))
            router.update(ep, state, idx, reward, probs)
            i2, p2 = no_state.select(ep, random.Random(s * 100 + e), temperature=0.5)
            no_state.update(ep, i2, float(environment_success(ep, i2)), p2)
    return router, no_state


def evaluate_intervention(
    *,
    train_seeds: Sequence[int],
    test_seeds: Sequence[int],
    n_candidates: int = 8,
    dim: int = 8,
    k: int = 4,
    noise: float = 0.0,
    train_seed: int = 0,
    eval_seed: int = 1,
    lr: float = 0.05,
    T: int | None = None,
) -> dict:
    """All arms on identical held-out episodes. Frozen router, then intervened."""
    if T is None:
        T = default_T(dim, k)
    keys = all_keys(dim, k)

    router, no_state = _train(train_seeds, dim, k, n_candidates, T, train_seed, lr, keys)
    rng = random.Random(eval_seed)

    rows = []
    for s in test_seeds:
        state = make_stream(s, dim, k)
        for e in range(1):
            ep = make_episode_in_stream(s * 100 + e, state, dim, n_candidates, T)

            # Intervened states: the SAME episodes, the SAME frozen router.
            wrong_state = replace_key(state, rng, dim)
            corrupt_state = corrupt_target(state, rng, 1.0)

            arms: dict[str, int] = {
                "learned": router.select(ep, state, random.Random(s + 1_000_000), temperature=0.05)[0],
                "learned_no_state": no_state.select(ep, random.Random(s + 1_000_000), temperature=0.05)[0],
                "true_key": relation_oracle_select(ep, state),
                "wrong_key": relation_oracle_select(ep, wrong_state),
                "corrupt_key": relation_oracle_select(ep, corrupt_state),
                "t_static": t_static_select(ep, state, rng),
                "static": static_select(ep, dim, rng),
                "anti_static": anti_static_select(ep, dim, rng),
                "random": random_select(ep, rng),
                "oracle": ep.gold_index,
            }
            for name, idx in arms.items():
                rows.append({
                    "seed": s, "arm": name, "selected": idx,
                    "gold_index": ep.gold_index if name == "oracle" else None,
                    "top1": float(idx == ep.gold_index),
                    "success": float(environment_success(ep, idx)),
                })

    arm_order = ["learned", "learned_no_state", "true_key", "wrong_key",
                 "corrupt_key", "t_static", "static", "anti_static", "random", "oracle"]
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

    summary["intervention_effect"] = {
        "true_key_minus_wrong_key":
            summary["true_key"]["success_mean"] - summary["wrong_key"]["success_mean"],
        "true_key_minus_corrupt_key":
            summary["true_key"]["success_mean"] - summary["corrupt_key"]["success_mean"],
    }
    summary["state_necessity"] = {
        "learned_minus_learned_no_state":
            summary["learned"]["success_mean"] - summary["learned_no_state"]["success_mean"],
    }

    return {
        "protocol": "PNDS-GATE-001",
        "stage": "3b",
        "config": {
            "train_streams": len(train_seeds),
            "test_streams": len(test_seeds),
            "episodes_per_stream": 1,
            "candidates": n_candidates,
            "dim": dim, "k": k, "T": T,
            "noise": noise,
            "train_seed": train_seed,
            "eval_seed": eval_seed,
            "lr": lr,
        },
        "summary": summary,
        "rows": rows,
    }


def run(train_streams: int = 250, test_streams: int = 300,
        n_candidates: int = 8, dim: int = 8, k: int = 4,
        train_seed: int = 0, eval_seed: int = 1) -> dict:
    train_seeds = list(range(train_seed, train_seed + train_streams))
    test_seeds = list(range(10_000, 10_000 + test_streams))
    return evaluate_intervention(
        train_seeds=train_seeds,
        test_seeds=test_seeds,
        n_candidates=n_candidates,
        dim=dim, k=k,
        train_seed=train_seed,
        eval_seed=eval_seed,
    )


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
