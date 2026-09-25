"""PNDS-GATE-001 Stage 3b, Probe 2: dense-gradient signal at the full 70 keys.

RETRACTED AS DESIGNED -- see STAGE_3B_AMENDMENT_3.md and
STAGE_3B_RESULT_CORRECTED.md.

This probe was registered in STAGE_3B_AMENDMENT_2.md (d545e62) to distinguish
sample allocation from sparse credit assignment as the cause of the Stage 3b
F1 failure. It ran, and it is the run that exposed the real defect:
`PersistentStateRouter._features` never read `state.key`, so the feature map
was key-blind and the relation sat in its null space. Every number this probe
produced measured an impossible learning problem.

Amendment 2's decision table is therefore moot. With the corrected gated map,
sparse REINFORCE at 70 keys reaches 1.0000 (STAGE_3B_RESULT_CORRECTED.md s5),
so neither hypothesis the probe was built to separate was the problem, and
re-running it would measure nothing of interest.

This file is retained in the record, unretracted as a document, because the
anomaly in its own output -- a dense gradient failing where the fixed-key
REINFORCE arm had succeeded -- is what made the defect visible. Deleting it
would erase the trace. It is NOT run by any test or experiment.

What follows is the original module docstring, left intact as the registered
design.
--------------------------------------------------------------------------
Registered in STAGE_3B_AMENDMENT_2.md (d545e62). Run only after Probe 1's
result was recorded (b8b873b).

Purpose
-------
`dd8f63c` established that sparse binary REINFORCE cannot acquire the 70-way
key-indexed read, and Probe 1 established that halving the key space to 28
softens but does not resolve the failure. Both are consistent with two
hypotheses:

    H_card  : the failure is sample allocation. Each of the 70 key blocks
              sees too few updates; a dense signal would not help.
    H_grad  : the failure is sparse binary credit assignment. A single reward
              bit carries too little information per update; a dense gradient
              that identifies the correct key would recover acquisition.

This probe separates them. It changes **one** thing relative to `dd8f63c`: the
source of gradient.

    dd8f63c :  grad = lr * (r - 1/n) * (1 - p_sel) * phi(selected)
              one scalar bit, multiplied across the whole active block

    Probe 2 :  grad = lr * d/dw  CE(p(w), gold_distribution)
              dense, per-candidate, label-derived

Everything else is identical: the state read, the feature basis, the parameter
count, the key space, the stream structure, the train/test boundary and the
10-seed protocol.

What this is NOT
----------------
This is a learner probe, not a PNDS mechanism. The gold index is used only to
compute the gradient; it is never given to the router at test time, and the
state is still read only through the routing interface. No architectural
change, no new component, no state-representation change, no environment
redesign, no additional tuning -- the binding constraints of Amendment 2 s4.

A dense signal is strictly more informative than a sparse one. If acquisition
still fails, the conclusion is that the learner problem is broader than sparse
REINFORCE, and the substrate/learner separation itself needs revisiting.
"""
from __future__ import annotations

import math
import random
from typing import Sequence

from .stage3b_persistent import (
    PersistentState, Episode, all_keys, default_T,
    environment_success, make_episode_in_stream, make_stream,
    relation_oracle_select, t_static_select, static_select,
    anti_static_select, random_select, replace_key, corrupt_target,
    PersistentStateRouter, StatelessRouter,
)


class DenseGradientRouter(PersistentStateRouter):
    """Same features, same parameters as the committed router.

    Only ``update`` changes: cross-entropy against the gold distribution rather
    than REINFORCE on a scalar reward. The gold index is consumed here and only
    here; it never reaches ``select`` or ``probabilities``.
    """

    def update(self, ep: Episode, state: PersistentState, selected: int,
               reward: float, probs: list[float]) -> None:
        del selected, reward            # unused; the gradient is not outcome-derived
        p = probs
        gold = ep.gold_index
        n = len(ep.candidates)

        # d CE / d logit_k = p_k - 1[k == gold], computed at the CURRENT weights.
        for i, cand in enumerate(ep.candidates):
            err = p[i] - (1.0 if i == gold else 0.0)
            feats = self._features(ep.query, cand.descriptor, state)
            for kk in range(self.n):
                self.w[kk] -= self.lr * err * feats[kk]

        # NOTE: this iterates all candidates, so each of the 70 key blocks
        # receives gradient from every candidate in the episode, not only from
        # the one that was sampled. That is the dense part: per-candidate error
        # rather than one scalar reward bit. Only the block matching the
        # episode's key is non-zero, so the other 69 blocks are unchanged --
        # but the active block gets 8x the signal REINFORCE gave it.


def _train_dense(train_seeds: Sequence[int], dim: int, k: int, n_candidates: int,
                 T: int, train_seed: int, lr: float,
                 keys: Sequence[frozenset[int]]):
    """Same stream structure and episode budget as stage3b_persistent._train.

    Only the update rule differs.
    """
    router = DenseGradientRouter(dim, k, keys, lr=lr)
    no_state = StatelessRouter(dim, lr=lr)
    order = list(train_seeds)
    random.Random(train_seed).shuffle(order)
    for s in order:
        state = make_stream(s, dim, k)
        for e in range(8):
            ep = make_episode_in_stream(s * 100 + e, state, dim, n_candidates, T)
            probs = router.probabilities(ep, state, temperature=0.5)
            router.update(ep, state, -1, -1.0, probs)
            i2, p2 = no_state.select(ep, random.Random(s * 100 + e), temperature=0.5)
            no_state.update(ep, i2, float(environment_success(ep, i2)), p2)
    return router, no_state


def evaluate_dense(
    *,
    train_seeds: Sequence[int],
    test_seeds: Sequence[int],
    n_candidates: int = 8,
    dim: int = 8,
    k: int = 4,
    train_seed: int = 0,
    eval_seed: int = 1,
    lr: float = 0.05,
    T: int | None = None,
) -> dict:
    """Probe 2. The same 70-key environment as dd8f63c, dense gradient.

    The stateless comparator is UNCHANGED: it still trains by REINFORCE, so the
    F1 comparison isolates the change to the stateful arm. Giving the stateless
    arm a dense signal would destroy the comparison, because its features carry
    no information about gold either way and the point is to hold the control
    fixed while varying the arm under test.
    """
    if T is None:
        T = default_T(dim, k)
    keys = all_keys(dim, k)

    router, no_state = _train_dense(train_seeds, dim, k, n_candidates, T,
                                    train_seed, lr, keys)
    rng = random.Random(eval_seed)

    rows = []
    for s in test_seeds:
        state = make_stream(s, dim, k)
        ep = make_episode_in_stream(s * 100, state, dim, n_candidates, T)

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
        "probe": 2,
        "signal": "dense-supervised-gradient",
        "config": {
            "train_streams": len(train_seeds),
            "test_streams": len(test_seeds),
            "episodes_per_stream": 8,
            "candidates": n_candidates,
            "dim": dim, "k": k, "T": T,
            "train_seed": train_seed,
            "eval_seed": eval_seed,
            "lr": lr,
        },
        "summary": summary,
        "rows": rows,
    }
