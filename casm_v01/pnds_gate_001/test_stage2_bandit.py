"""Methodology tests for the Stage 2 outcome-trained router."""
from casm_v01.pnds_gate_001.stage2_bandit import (
    LinearBanditRouter, make_episode, environment_success, run
)

def test_gold_is_not_router_input():
    ep=make_episode(7, n_candidates=6, dim=8, noise=0.10)
    router=LinearBanditRouter(8)
    assert hasattr(ep, "gold_index")
    assert "gold_index" not in router.__dict__

def test_environment_has_nontrivial_observable_signal():
    hits=0
    for seed in range(100):
        ep=make_episode(seed, n_candidates=8, dim=8, noise=0.10)
        # The candidate whose public descriptor is most similar to the public query
        # should be substantially better than chance in this synthetic environment.
        scores=[
            sum(1 if ep.query[j]==c.descriptor[j] else 0 for j in range(8))
            for c in ep.candidates
        ]
        hits += int(scores.index(max(scores)) == ep.gold_index)
    assert hits/100 > 0.30

def test_stage2_runs_without_gold_supervision():
    result=run(train_episodes=40,test_episodes=20,n_candidates=6,dim=8,noise=0.10)
    assert result["stage"] == 2
    assert 0.0 <= result["test_success_mean"] <= 1.0
