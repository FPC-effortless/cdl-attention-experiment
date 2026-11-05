"""Exploratory training tests; confirmatory seeds are deliberately reserved."""
from dataclasses import replace

import pytest

from .benchmark import build_trial, select_trial
from .outcome_coder import controlled_pair_outcomes, fit_from_outcomes, PairOutcome
from .outcome_benchmark import run


def test_pair_outcomes_contain_only_answerability_no_numeric_class_labels():
    vocabulary, outcomes = controlled_pair_outcomes()
    assert len(vocabulary) == 24
    assert len(outcomes) == 276
    assert sum(pair.answered for pair in outcomes) == 36
    assert all(isinstance(pair.answered, bool) for pair in outcomes)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_balanced_coder_recovers_discrete_separation_on_exploratory_seeds(seed):
    vocabulary, outcomes = controlled_pair_outcomes()
    coder, record = fit_from_outcomes(vocabulary, outcomes, n_codes=6,
                                      seed=seed, steps=800, balance_weight=0.5)
    assert record["unique_hard_codes"] == 6
    assert record["positive_pair_accuracy"] == 1.0
    assert record["negative_pair_accuracy"] == 1.0
    assert record["training_pair_count"] == 276
    assert record["positive_training_pairs"] == 36
    trial = build_trial(256, seed=seed + 250_001)
    selected, _, _ = select_trial(coder, trial)
    assert selected.indices == (trial.gold_token_index,)


def test_training_refuses_non_discriminating_outcome_stream():
    vocabulary, outcomes = controlled_pair_outcomes()
    all_positive = tuple(replace(pair, answered=True) for pair in outcomes)
    with pytest.raises(ValueError):
        fit_from_outcomes(vocabulary, all_positive, n_codes=6, seed=0, steps=1)


def test_negative_outcome_is_reported_not_excluded_as_invalid():
    report = run(seeds=(29,), lengths=(8,), trials=4, steps=1)
    assert "balanced_pair_credit" in report["arms"]
    assert "unbalanced_pair_credit" in report["arms"]
    for arm in report["arms"].values():
        assert len(arm["train_stats"]) == 1
        assert 0.0 <= arm["heldout_recall_by_length"]["8"]["mean"] <= 1.0
