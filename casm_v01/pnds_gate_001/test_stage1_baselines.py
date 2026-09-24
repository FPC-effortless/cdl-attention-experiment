from .stage1_baselines import (
    evaluate, make_episode, oracle_router, random_router,
    static_overlap_router, environment_success, intervention_effect,
)

def test_oracle_is_capacity_ceiling():
    ep = make_episode(1)
    idx = oracle_router(ep)
    assert environment_success(ep, idx)

def test_static_router_has_no_gold_input():
    ep = make_episode(2)
    idx = static_overlap_router(ep)
    assert 0 <= idx < len(ep.candidates)

def test_random_router_is_bounded():
    ep = make_episode(3)
    idx = random_router(ep, __import__("random").Random(0))
    assert 0 <= idx < len(ep.candidates)

def test_intervention_changes_success_for_correct_selection():
    ep = make_episode(4)
    assert intervention_effect(ep, ep.gold_index) == 1.0

def test_stage1_is_reproducible():
    a = evaluate(seeds=range(10), n_candidates=8, random_seed=7)
    b = evaluate(seeds=range(10), n_candidates=8, random_seed=7)
    assert a == b
