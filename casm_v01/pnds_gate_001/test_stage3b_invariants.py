"""Property tests for the Stage 3b corrected environment.

These exist BEFORE the environment implementation, per Stage 3b Amendment 1
section 10, step 2. Their purpose is to prevent a future generator modification
from reintroducing the registered-design defect (Amendment 1 section 1) through
a different statistic.

The defect they guard against: under the original registered design a router
that read the target t and ignored the key K scored 0.6965 at 8 candidates,
because gold had systematically higher TOTAL t-agreement than any distractor.

Both tests were verified to:
    - PASS 100% on the corrected generator (every episode)
    - FAIL 100% on the original  registered generator

so they catch the regression rather than passing vacuously.
"""
from __future__ import annotations

import random
import unittest


def original_generator(seed: int, dim: int = 8, k: int = 4, n: int = 8):
    """The REGISTERED (defective) generator from 3a03225.

    Reproduced here verbatim-in-spirit so the property tests can be proven to
    fail on it. NOT used by any experiment; it is the negative control.
    """
    r = random.Random(seed)
    t = tuple(r.randrange(2) for _ in range(dim))
    key = set(r.sample(range(dim), k))
    gold = [r.randrange(2) for _ in range(dim)]
    for j in key:
        gold[j] = t[j]
    cands = [tuple(gold)]
    made, guard = 1, 0
    while made < n and guard < 400 * n:
        guard += 1
        d = [r.randrange(2) for _ in range(dim)]
        jv = r.choice(sorted(key))
        d[jv] = 1 - t[jv]
        if tuple(d) == tuple(gold):
            continue
        cands.append(tuple(d))
        made += 1
    return t, key, cands


# --------------------------------------------------------------------------- #
# The two invariants, written so they can be evaluated on ANY generator.
# --------------------------------------------------------------------------- #

def exact_agreement_total(t, cands, dim: int, T: int) -> bool:
    """Invariant 1: every candidate has exactly T agreements with t.

        for all i:  sum_j 1[c_ij == t_j] == T

    Under this invariant the total-agreement score vector is CONSTANT across
    candidates, so a t-only router (read t, ignore the key) cannot separate
    gold from distractors. The principal stateless shortcut is then
    structurally impossible rather than merely empirically unlikely.
    """
    return all(
        sum(1 for j in range(dim) if c[j] == t[j]) == T
        for c in cands
    )


def per_position_marginal(t, cands, dim: int, T: int, tol: float = 1e-12) -> bool:
    """Invariant 2: per-position agreement marginals match T/dim for gold
    AND for the mean distractor, in every single episode.

        P(c_ij == t_j | gold)      == P(c_ij == t_j | distractor) == T / dim

    This blocks the leak from reappearing through a different statistic: even
    if a future modification keeps the total at T while shifting WHICH
    positions agree, the marginal equality detects it.

    Evaluated per episode, not aggregated, because the exact-T construction
    makes both marginals exactly T/dim in every episode.
    """
    if not cands:
        return False
    target = T / dim
    gold_rate = sum(1 for j in range(dim) if cands[0][j] == t[j]) / dim
    if abs(gold_rate - target) > tol:
        return False
    n_dist = len(cands) - 1
    if n_dist == 0:
        return True
    dist_rate = (
        sum(1 for c in cands[1:] for j in range(dim) if c[j] == t[j])
        / (n_dist * dim)
    )
    return abs(dist_rate - target) <= tol


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

N_EPISODES = 500


class TestPropertyTestsCatchTheRegisteredDefect(unittest.TestCase):
    """The property tests must FAIL on the original registered generator.

    If they pass on it, they do not guard anything and the regression the
    amendment exists to prevent is unprotected.
    """

    def test_exact_T_fails_on_original_generator(self):
        passed = sum(
            exact_agreement_total(t, cands, dim=8, T=6)
            for t, key, cands in (original_generator(s) for s in range(N_EPISODES))
        )
        self.assertEqual(
            passed, 0,
            "exact-T property must FAIL on the registered defective generator, "
            f"but passed on {passed}/{N_EPISODES} episodes; the guard is vacuous.",
        )

    def test_per_position_marginal_fails_on_original_generator(self):
        passed = sum(
            per_position_marginal(t, cands, dim=8, T=6)
            for t, key, cands in (original_generator(s) for s in range(N_EPISODES))
        )
        self.assertEqual(
            passed, 0,
            "per-position marginal property must FAIL on the registered "
            f"defective generator, but passed on {passed}/{N_EPISODES}; "
            "the guard is vacuous.",
        )


class TestOriginalGeneratorHasTheDefect(unittest.TestCase):
    """Positive control: the registered generator really does leak.

    This is the defect recorded in Amendment 1 section 1, measured here so the
    property tests are anchored to an actual failure rather than a hypothetical
    one. A t-only router scores far above chance.
    """

    def test_t_only_retrieval_scores_far_above_chance(self):
        rng = random.Random(7)
        n = 8
        hits = 0
        N = 1000
        for s in range(100_000, 100_000 + N):
            t, key, cands = original_generator(s, n=n)
            dim = len(t)
            scores = [sum(1 for j in range(dim) if c[j] == t[j]) for c in cands]
            best = max(scores)
            tied = [i for i, v in enumerate(scores) if v == best]
            # Gold is always constructed at index 0 by the original generator.
            if rng.choice(tied) == 0:
                hits += 1
        rate = hits / N
        chance = 1.0 / n
        self.assertGreater(
            rate, chance + 0.30,
            f"expected the registered generator to leak badly (t-only >> chance), "
            f"got {rate:.4f} vs chance {chance:.4f}; the defect premise is wrong.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
