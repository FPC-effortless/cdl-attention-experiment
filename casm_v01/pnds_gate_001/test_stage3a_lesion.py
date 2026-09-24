"""Methodology tests for the Stage 3a weight-lesion experiment.

Mirrors the discipline of the Stage 2b/2c tests: identical episodes across arms,
no gold supervision of any router-derived arm, disjoint train/test seeds, seed
determinism, distinct multi-seed training seeds. Adds invariants specific to
lesioning: lesions must not damage the intact reference, and the lesioned router
must be a frozen copy rather than a retrained one.
"""
from .stage3a_lesion import (
    run, evaluate_lesions, apply_lesion, train_frozen, LesionSpec,
    anti_static_select, DEFAULT_LESIONS,
)
from .stage2c_relational import make_episode
import random


def test_all_arms_evaluated_on_identical_episodes():
    """No arm may see a different episode population than the others."""
    result = evaluate_lesions(
        train_seeds=range(40),
        test_seeds=range(10_000, 10_020),
        n_candidates=6, dim=8, noise=0.10,
    )
    seeds_by_arm = {}
    for row in result["rows"]:
        seeds_by_arm.setdefault(row["arm"], set()).add(row["seed"])
    expected = {"intact", "static", "anti_static", "random", "oracle"}
    expected |= {spec.name for spec in DEFAULT_LESIONS}
    assert set(seeds_by_arm) == expected, f"unexpected arms: {set(seeds_by_arm) ^ expected}"
    first = seeds_by_arm[next(iter(seeds_by_arm))]
    for arm in seeds_by_arm:
        assert seeds_by_arm[arm] == first, f"{arm} saw different episodes"


def test_lesioned_arms_receive_no_gold_supervision():
    """A lesioned router is a frozen copy; it must not record gold."""
    result = evaluate_lesions(
        train_seeds=range(20),
        test_seeds=range(10_000, 10_010),
        n_candidates=6, dim=8, noise=0.10,
    )
    for row in result["rows"]:
        if row["arm"] in ("intact", "joint", "gated_only", "qc_only", "gated_flip"):
            assert row["gold_index"] is None


def test_lesion_does_not_mutate_the_intact_router():
    """The reference router must be untouched by lesioning.

    This is the core invariant of the design: the question is whether behaviour
    was dependent on the frozen weights, so the frozen weights must not change
    when a lesion is applied.
    """
    router = train_frozen(range(60), 6, 8, 0.10, train_seed=0, lr=0.05)
    before = list(router.w)
    for spec in DEFAULT_LESIONS:
        apply_lesion(router, spec)
    assert router.w == before, "lesioning mutated the intact router"


def test_lesioned_router_is_not_retrained():
    """A lesioned router must be a copy with frozen (already-converged) weights."""
    router = train_frozen(range(60), 6, 8, 0.10, train_seed=0, lr=0.05)
    clone = apply_lesion(router, LesionSpec(name="t", gated=0.0, qc=0.0))
    assert clone is not router
    assert clone.w is not router.w
    # An evaluation pass must not change the lesioned weights.
    before = list(clone.w)
    ep = make_episode(1, 6, 8, 0.10)
    clone.select(ep, random.Random(0))
    assert clone.w == before, "lesioned router was modified during evaluation"


def test_joint_lesion_is_the_default_and_is_destructive():
    """The joint lesion must exist by default and must be the strongest.

    The single-block lesions are confounded on this environment, so the joint
    lesion is the interpretable necessity test. If it stops being destructive,
    the mechanism claim is void.
    """
    names = {spec.name for spec in DEFAULT_LESIONS}
    assert names >= {"joint", "gated_only", "qc_only", "gated_flip"}
    result = run(train_episodes=300, test_episodes=60,
                 n_candidates=8, dim=8, noise=0.10)
    s = result["summary"]
    # The joint lesion must fall well below the anti-agreement shortcut, which
    # is what the qc block degenerates into. Otherwise the collapse is just the
    # shortcut and not a loss of the learned relation.
    assert s["joint"]["success_mean"] < s["anti_static"]["success_mean"], (
        "joint lesion did not fall below the anti_static shortcut; the necessity "
        "claim is confounded"
    )
    assert s["joint"]["success_mean"] < 0.5, "joint lesion was not destructive"


def test_gated_only_lesion_matches_the_anti_static_shortcut():
    """The confound must be visible in the data, not just asserted in prose.

    Lesioning the gated block alone leaves the anti-agreement block, which
    degenerates into the anti_static shortcut. Observing that degeneracy in the
    numbers is the evidence that the confound is real and understood, rather
    than an artefact of a particular seed.
    """
    result = run(train_episodes=300, test_episodes=60,
                 n_candidates=8, dim=8, noise=0.10)
    s = result["summary"]
    assert abs(s["gated_only"]["success_mean"] - s["anti_static"]["success_mean"]) < 0.15, (
        "gated_only did not degenerate to the anti_static shortcut, so the "
        "confound diagnosis is wrong"
    )


def test_test_seeds_do_not_overlap_train_seeds():
    result = run(train_episodes=10, test_episodes=10)
    cfg = result["config"]
    train = set(range(cfg["train_seed"], cfg["train_seed"] + cfg["train_episodes"]))
    test = set(range(10_000, 10_000 + cfg["test_episodes"]))
    assert not (train & test), "train and test seeds overlap"


def test_static_control_has_no_learned_parameters():
    ep = make_episode(3, n_candidates=6, dim=8, noise=0.10)
    a = static_and_anti_check(ep)
    assert a == static_and_anti_check(ep), "static controls must be deterministic"


def static_and_anti_check(ep):
    """Both fixed scorers must be stateless: identical on repeated calls."""
    from .stage2c_relational import static_select
    return (static_select(ep, 8), anti_static_select(ep, 8))


def test_environment_is_deterministic_by_seed():
    assert make_episode(5, 6, 8, 0.10) == make_episode(5, 6, 8, 0.10)


def test_multiseed_runner_aggregates_independent_training_seeds():
    """The multiseed path must vary the training seed, not re-run the same one."""
    import argparse
    from casm_v01.pnds_gate_001.run_stage3a import _multiseed

    a = argparse.Namespace(
        train_episodes=120, test_episodes=30, candidates=6, dim=8, noise=0.10,
        train_seed=0, eval_seed=1, seeds=3,
    )
    out = _multiseed(a)
    assert out["stage"] == "3a-multiseed"
    assert out["config"]["n_train_seeds"] == 3
    for metric in out["summary"]:
        st = out["summary"][metric]
        assert st["n"] == 3
        assert len(st["per_seed"]) == 3
        assert abs(st["mean"] - sum(st["per_seed"]) / 3) < 1e-9
    starts = [p["train_seed"] for p in out["per_seed_results"]]
    assert len(set(starts)) == 3, f"training seeds are not distinct: {starts}"
    assert starts == out["config"]["train_seeds"]


def test_frozen_weights_are_recorded_for_provenance():
    """The frozen weight vector must be in the artifact.

    The whole point of the stage is that the lesion acts on the converged
    weights, so the reader must be able to see exactly which weights were
    lesioned.
    """
    result = run(train_episodes=60, test_episodes=10,
                 n_candidates=6, dim=8, noise=0.10)
    w = result["frozen_weights"]
    for block in ("bias", "qc", "xc", "xq", "gated"):
        assert block in w
    assert len(w["gated"]) == 8
    assert len(w["qc"]) == 8
