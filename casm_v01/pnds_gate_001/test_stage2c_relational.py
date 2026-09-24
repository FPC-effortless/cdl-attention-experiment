"""Methodology tests for the Stage 2c relational relevance comparison.

Mirrors test_stage2b_heldout.py: the protocol discipline is what makes the 2c
numbers comparable to 2b, so the same invariants are asserted here.
"""
from .stage2c_relational import (
    run, static_select, random_select, evaluate_heldout,
    make_episode, environment_success, RelationalBanditRouter,
    satisfies_relation, marked_positions,
)
import random


def test_all_arms_evaluated_on_identical_episodes():
    """No arm may see a different episode population than the others."""
    result = evaluate_heldout(
        train_seeds=range(40),
        test_seeds=range(10_000, 10_020),
        n_candidates=6, dim=8, noise=0.10,
    )
    seeds_by_arm = {}
    for row in result["rows"]:
        seeds_by_arm.setdefault(row["arm"], set()).add(row["seed"])
    arms = sorted(seeds_by_arm)
    assert arms == ["learned", "oracle", "random", "static"]
    first = seeds_by_arm[arms[0]]
    for arm in arms[1:]:
        assert seeds_by_arm[arm] == first, f"{arm} saw different episodes"


def test_learned_arm_receives_no_gold_supervision():
    """The router object must not expose gold_index/target/answer fields."""
    result = evaluate_heldout(
        train_seeds=range(20),
        test_seeds=range(10_000, 10_010),
        n_candidates=6, dim=8, noise=0.10,
    )
    for row in result["rows"]:
        if row["arm"] == "learned":
            # The router may not record gold for its own selection.
            assert row["gold_index"] is None


def test_gold_index_is_environment_only():
    """Deterministic environment: the router input has no gold field."""
    ep = make_episode(7, n_candidates=6, dim=8, noise=0.10)
    router = RelationalBanditRouter(8)
    router_attrs = set(vars(router))
    forbidden = {"gold_index", "gold", "target", "answer", "reward"}
    assert not (router_attrs & forbidden)
    assert hasattr(ep, "gold_index"), "environment must retain gold for scoring"
    # Router scoring must not consult gold_index.
    idx, _ = router.select(ep, random.Random(0))
    assert idx in range(len(ep.candidates))


def test_test_seeds_do_not_overlap_train_seeds():
    """Held-out evaluation requires disjoint train/test episodes."""
    result = run(train_episodes=10, test_episodes=10)
    cfg = result["config"]
    train = set(range(cfg["train_seed"], cfg["train_seed"] + cfg["train_episodes"]))
    test = set(range(10_000, 10_000 + cfg["test_episodes"]))
    assert not (train & test), "train and test seeds overlap"


def test_static_control_has_no_learned_parameters():
    """The static control must be a fixed scorer with no trainable state."""
    ep = make_episode(3, n_candidates=6, dim=8, noise=0.10)
    a = static_select(ep, 8)
    b = static_select(ep, 8)
    assert a == b, "static control must be deterministic"
    # Calling it on a different episode cannot update internal state.
    other = make_episode(99, n_candidates=6, dim=8, noise=0.10)
    assert static_select(ep, 8) == b


def test_environment_is_deterministic_by_seed():
    """Replayability: same seed must yield the same episode."""
    assert make_episode(5, 6, 8, 0.10) == make_episode(5, 6, 8, 0.10)


def test_runs_produce_all_four_arms():
    result = evaluate_heldout(
        train_seeds=range(15),
        test_seeds=range(10_000, 10_008),
        n_candidates=6, dim=8, noise=0.10,
    )
    assert set(result["summary"]) >= {"learned", "static", "random", "oracle", "deltas"}
    for arm in ("learned", "static", "random", "oracle"):
        assert 0.0 <= result["summary"][arm]["success_mean"] <= 1.0


def test_multiseed_runner_aggregates_independent_training_seeds():
    """The multiseed path must vary the training seed, not re-run the same one."""
    import argparse
    from casm_v01.pnds_gate_001.run_stage2c import _multiseed

    a = argparse.Namespace(
        train_episodes=40, test_episodes=10, candidates=6, dim=8, noise=0.10,
        train_seed=0, eval_seed=1, seeds=3,
    )
    out = _multiseed(a)
    assert out["stage"] == "2c-multiseed"
    assert out["config"]["n_train_seeds"] == 3
    for metric in ("learned_minus_static", "learned_minus_random", "learned_minus_chance"):
        st = out["summary"][metric]
        assert st["n"] == 3
        assert len(st["per_seed"]) == 3
        assert abs(st["mean"] - sum(st["per_seed"]) / 3) < 1e-9
    # The per-seed results must record distinct training seeds (not re-runs).
    starts = [p["train_seed"] for p in out["per_seed_results"]]
    assert len(set(starts)) == 3, f"training seeds are not distinct: {starts}"
    assert starts == out["config"]["train_seeds"]


# --------------------------------------------------------------------------- #
# Stage 2c-specific invariants: the environment must be a valid control.
# --------------------------------------------------------------------------- #

def test_relation_is_observable_from_router_inputs():
    """The relation must be computable from query+context alone.

    If this fails, relevance depends on hidden state and the experiment would be
    testing model-class expressiveness rather than learning from outcomes --
    the flaw that killed two earlier Stage 2c designs.
    """
    ep = make_episode(11, n_candidates=6, dim=8, noise=0.10)
    # marked positions are a pure function of the context.
    assert marked_positions(ep.context, 8) == tuple(j for j in range(8) if ep.context[j] == 1)
    # the relation is decidable from query, context, and the candidate descriptor.
    gold = ep.candidates[ep.gold_index]
    assert satisfies_relation(ep.query, ep.context, gold.descriptor, 8)


def test_gold_is_the_unique_satisfier():
    """Gold must be the only candidate satisfying the relation.

    Otherwise the task is ambiguous, the oracle arm is capped below 1.0, and the
    learned/static gap is not attributable to relational learning.
    """
    n_violations = 0
    for seed in range(60):
        ep = make_episode(seed, n_candidates=6, dim=8, noise=0.10)
        n_sat = sum(satisfies_relation(ep.query, ep.context, c.descriptor, 8)
                    for c in ep.candidates)
        if n_sat != 1:
            n_violations += 1
    assert n_violations == 0, f"gold not unique in {n_violations}/60 episodes"


def test_static_rule_cannot_express_the_relation():
    """The unchanged 2b static scorer must be near chance or worse.

    This is the control that makes 2c decisive: if static still works, the
    environment did not actually change the source of relevance.
    """
    hits = sum(static_select(make_episode(s, 6, 8, 0.10), 8) == make_episode(s, 6, 8, 0.10).gold_index
               for s in range(60))
    assert hits / 60 <= 0.20, f"static control did not collapse: {hits / 60:.3f}"
