"""Representability probe for the Stage 3b learned router.

Added by Amendment 3. This is the check that was missing from `3a03225`,
`528ad89`, `dd8f63c` and `b8b873b`, and whose absence let a key-blind feature
map hide behind a perfect `true_key` oracle arm for four commits.

What it does
------------
Instantiate the router, set its weights to the ANALYTICALLY IDEAL values, and
assert that gold outscores every distractor on held-out episodes.

The ideal weights are, per key block:

    w[j] = +1   if j in that block's key
    w[j] = -1   otherwise

so the active block scores `a - (T - a) = 2a - T`, which is `+2` for gold
(a = k) and `<= 0` for every distractor (a <= k - 1).

Why it is not just another arm
------------------------------
Every representability check in the registered design was an ARM: `true_key`,
`static`, `anti_static`, `t_static`. An arm that reads the relation directly
never passes through the learned feature map, so it scores 1.0000 regardless of
whether the router could ever represent the relation. A basis defect therefore
hides indefinitely behind a perfect oracle. This probe tests the map itself.

Verified to
-----------
    FAIL loudly on the committed key-blind map of dd8f63c
    PASS on the gated map of Amendment 3

so it does not pass vacuously. The key-blind map scores the ideal weights at
exactly 0.0 for every candidate, because

    sum_{K'} (2 a_{K'} - T) = 2 * C(dim-1, k-1) * T - n_keys * T = 0

for every candidate under the exact-T constraint.
"""
from __future__ import annotations

import random
import unittest
from itertools import combinations

from .stage3b_persistent import (
    PersistentStateRouter, all_keys, default_T, make_episode_in_stream,
    make_stream,
)


def ideal_weights(dim: int, k: int, all_keys_list) -> list[float]:
    """The analytically optimal weight vector: +1 on key positions, -1 off,
    in every key block, with a zero bias."""
    w = [0.0]
    for key in all_keys_list:
        w.extend(1.0 if j in key else -1.0 for j in range(dim))
    return w


def key_blind_features(router: PersistentStateRouter, descriptor, state):
    """The committed (defective) feature map of dd8f63c, reproduced here so the
    probe can be PROVEN to fail on it. NOT used by any experiment."""
    dim = router.dim
    agree = [1.0 if descriptor[j] == state.target[j] else -1.0 for j in range(dim)]
    feats = [1.0]
    for key in router.all_keys:
        feats.extend(agree[j] if j in key else -agree[j] for j in range(dim))
    return feats


def ideal_goldscore_margin(router: PersistentStateRouter, ep, state) -> float:
    """Gold's score under the ideal weights, minus the best distractor's.

    Positive means the relation is representable. Zero or negative means it is
    not, and every learned-arm number measured on this map is void.
    """
    w = ideal_weights(router.dim, router.k, router.all_keys)
    feats = [router._features(ep.query, c.descriptor, state) for c in ep.candidates]
    scores = [sum(a * b for a, b in zip(w, f)) for f in feats]
    gold = scores[ep.gold_index]
    best_dist = max(s for i, s in enumerate(scores) if i != ep.gold_index)
    return gold - best_dist


class TestIdealWeightsAreRepresentable(unittest.TestCase):
    """The relation must be representable in the router's feature map before
    any learned-arm measurement is meaningful."""

    def test_ideal_weights_separate_gold_from_distractors_k4(self):
        dim, k, T, n = 8, 4, 6, 8
        keys = all_keys(dim, k)
        router = PersistentStateRouter(dim, k, keys)
        worst = float("inf")
        for s in range(300):
            state = make_stream(40_000 + s, dim, k)
            ep = make_episode_in_stream(40_000 + s, state, dim, n, T)
            worst = min(worst, ideal_goldscore_margin(router, ep, state))
        self.assertGreater(
            worst, 0,
            f"ideal weights fail to separate gold from distractors "
            f"(min margin {worst}); the feature map cannot represent the "
            f"relation and every learned-arm number is void. See "
            f"STAGE_3B_AMENDMENT_3.md.",
        )

    def test_ideal_weights_separate_gold_from_distractors_k2(self):
        dim, k, T, n = 8, 2, 5, 8
        keys = all_keys(dim, k)
        router = PersistentStateRouter(dim, k, keys)
        worst = float("inf")
        for s in range(300):
            state = make_stream(50_000 + s, dim, k)
            ep = make_episode_in_stream(50_000 + s, state, dim, n, T)
            worst = min(worst, ideal_goldscore_margin(router, ep, state))
        self.assertGreater(
            worst, 0,
            f"ideal weights fail at k=2 (min margin {worst}); "
            f"STAGE_3B_PROBE1_RESULT.md numbers are also void.",
        )

    def test_gated_map_uses_state_key(self):
        """The correction of Amendment 3: the feature map must depend on
        state.key, or the map is key-blind and the score is constant."""
        dim, k, T, n = 8, 4, 6, 8
        keys = all_keys(dim, k)
        router = PersistentStateRouter(dim, k, keys)
        state = make_stream(5, dim, k)
        ep = make_episode_in_stream(5, state, dim, n, T)
        other_key = next(kk for kk in keys if kk != state.key)
        other = type(state)(target=state.target, key=other_key)
        f1 = router._features(ep.query, ep.candidates[0].descriptor, state)
        f2 = router._features(ep.query, ep.candidates[0].descriptor, other)
        self.assertNotEqual(
            f1, f2,
            "features are identical under a key swap: the map is key-blind "
            "and the relation is unrepresentable (STAGE_3B_AMENDMENT_3.md).",
        )

    def test_only_the_states_key_block_is_nonzero(self):
        """All 69 non-state key blocks must be exactly zero."""
        dim, k, T, n = 8, 4, 6, 8
        keys = all_keys(dim, k)
        router = PersistentStateRouter(dim, k, keys)
        state = make_stream(5, dim, k)
        ep = make_episode_in_stream(5, state, dim, n, T)
        f = router._features(ep.query, ep.candidates[0].descriptor, state)
        b = router.key_index[state.key]
        for other in range(len(keys)):
            if other == b:
                continue
            block = f[1 + other * dim : 1 + (other + 1) * dim]
            self.assertEqual(
                sum(abs(x) for x in block), 0.0,
                f"block {other} is non-zero; only the state's key block may "
                f"be active (STAGE_3B_AMENDMENT_3.md).",
            )


class TestProbeFailsOnTheKeyBlindMap(unittest.TestCase):
    """The probe must FAIL on the committed defective map, or it guards
    nothing. This is its negative control."""

    def test_key_blind_ideal_weights_score_a_constant(self):
        dim, k, T, n = 8, 4, 6, 8
        keys = all_keys(dim, k)
        router = PersistentStateRouter(dim, k, keys)
        w = ideal_weights(dim, k, keys)
        constant_episodes = 0
        N = 200
        for s in range(30_000, 30_000 + N):
            state = make_stream(s, dim, k)
            ep = make_episode_in_stream(s, state, dim, n, T)
            feats = [key_blind_features(router, c.descriptor, state)
                     for c in ep.candidates]
            scores = [sum(a * b for a, b in zip(w, f)) for f in feats]
            if abs(max(scores) - min(scores)) < 1e-9:
                constant_episodes += 1
        self.assertEqual(
            constant_episodes, N,
            f"the key-blind map of dd8f63c must score a constant in every "
            f"episode; got {constant_episodes}/{N}. If it does not, the "
            f"retraction in STAGE_3B_AMENDMENT_3.md is misstated.",
        )

    def test_gated_probe_fails_on_key_blind_map(self):
        """Run the representability probe through the DEFECTIVE map to prove
        the probe catches the regression."""
        dim, k, T, n = 8, 4, 6, 8
        keys = all_keys(dim, k)
        router = PersistentStateRouter(dim, k, keys)

        worst = float("inf")
        for s in range(300):
            state = make_stream(40_000 + s, dim, k)
            ep = make_episode_in_stream(40_000 + s, state, dim, n, T)
            w = ideal_weights(dim, k, keys)
            feats = [key_blind_features(router, c.descriptor, state)
                     for c in ep.candidates]
            scores = [sum(a * b for a, b in zip(w, f)) for f in feats]
            gold = scores[ep.gold_index]
            best = max(x for i, x in enumerate(scores) if i != ep.gold_index)
            worst = min(worst, gold - best)
        self.assertLessEqual(
            worst, 0.0,
            f"the probe must fail on the key-blind map but got min margin "
            f"{worst}; the negative control is broken.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
