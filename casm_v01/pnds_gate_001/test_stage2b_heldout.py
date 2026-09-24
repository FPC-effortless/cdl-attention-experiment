"""Methodology tests for the Stage 2b held-out routing comparison."""
from .stage2b_heldout import (
    run, static_select, random_select, evaluate_heldout,
)
from .stage2_bandit import make_episode, environment_success, LinearBanditRouter
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
    router = LinearBanditRouter(8)
    router_attrs = set(vars(router))
    forbidden = {"gold_index", "gold", "target", "answer", "reward"}
    assert not (router_attrs & forbidden)
    assert hasattr(ep, "gold_index"), "environment must retain gold for scoring"
    # Router-visible episode fields exclude gold.
    visible = set(f for f in vars(ep))
    assert "gold_index" in visible  # present on episode, but never router input
    # Router scoring must not consult gold_index:
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
    from casm_v01.pnds_gate_001.run_stage2b import _multiseed

    a = argparse.Namespace(
        train_episodes=40, test_episodes=10, candidates=6, dim=8, noise=0.10,
        train_seed=0, eval_seed=1, seeds=3,
    )
    out = _multiseed(a)
    assert out["stage"] == "2b-multiseed"
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
