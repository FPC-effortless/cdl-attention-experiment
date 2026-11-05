"""Capability and anti-leakage tests for the LRR v0.1 pilot."""
from __future__ import annotations

import pytest

from .router import CausalRelationIndex, RELATION_ALIASES, fit_coder
from .benchmark import benchmark, build_trial, select_trial


@pytest.fixture(scope="module")
def coder():
    model, report = fit_coder(seed=17, steps=120)
    assert report["train_lexical_accuracy"] == 1.0
    return model


def test_codes_align_paraphrases_not_unrelated_relations(coder):
    for family, aliases in RELATION_ALIASES.items():
        codes = {coder.code(alias) for alias in aliases}
        assert len(codes) == 1, family
    assert len({coder.code(aliases[0]) for aliases in RELATION_ALIASES.values()}) == len(RELATION_ALIASES)


def test_learned_selection_points_at_actual_prior_value_token(coder):
    t = build_trial(64, seed=9821)
    selected, index, _ = select_trial(coder, t)
    assert selected.indices == (t.gold_token_index,)
    assert selected.values == (t.gold_value,)
    assert t.tokens[selected.indices[0]] == t.gold_value
    assert selected.index_probes == 1 and selected.posting_visits == 1
    assert index.observed_facts == 64
    assert index.distinct_addresses <= 64


@pytest.mark.parametrize("n_facts", [8, 32, 256, 1024])
def test_retrieval_remains_correct_when_distractors_grow(coder, n_facts):
    for seed in (3, 4, 5):
        trial = build_trial(n_facts, seed=seed * 10000 + n_facts)
        result, index, _ = select_trial(coder, trial)
        assert result.indices == (trial.gold_token_index,)
        assert result.index_probes == 1
        assert result.posting_visits == 1
        assert index.observed_facts == n_facts
        assert result.indices[0] < trial.query_index


def test_alias_swap_needs_semantic_alignment(coder):
    trial = build_trial(32, seed=77)
    gold = next(f for f in trial.facts if f.token_index == trial.gold_token_index)
    assert trial.query_predicate != gold.predicate
    selection, _, _ = select_trial(coder, trial)
    assert selection.values == (gold.value,)
    # An exact lexical baseline would not find this fact.
    assert not any(f.entity == trial.query_entity and f.predicate == trial.query_predicate
                   for f in trial.facts)


def test_unknown_relation_abstains_without_hallucinating_a_position(coder):
    index = CausalRelationIndex(coder)
    index.observe(token_index=3, entity="mira", predicate="owns", value="violin")
    assert index.select(query_index=5, entity="mira", predicate="UNSEEN").indices == ()
    index.observe(token_index=8, entity="mira", predicate="UNSEEN", value="irrelevant")
    assert index.select(query_index=10, entity="mira", predicate="possesses").values == ("violin",)


def test_causal_api_blocks_future_reads_and_out_of_order_writes(coder):
    index = CausalRelationIndex(coder)
    index.observe(token_index=3, entity="mira", predicate="owns", value="violin")
    with pytest.raises(ValueError):
        index.select(query_index=3, entity="mira", predicate="owns")
    with pytest.raises(ValueError):
        index.observe(token_index=2, entity="bob", predicate="owns", value="lamp")
    with pytest.raises(ValueError):
        index.select(query_index=5, entity="mira", predicate="owns", k=0)


def test_posting_budget_and_newest_first_order(coder):
    index = CausalRelationIndex(coder)
    for i in range(12):
        index.observe(token_index=5 * i + 3, entity="mira",
                      predicate="owns", value=f"value{i}")
    selection = index.select(query_index=60, entity="mira",
                             predicate="possesses", k=3)
    assert selection.values == ("value11", "value10", "value9")
    assert selection.indices == (58, 53, 48)
    assert selection.posting_visits == 3
    assert selection.index_probes == 1


def test_entity_key_prevents_same_relation_interference(coder):
    index = CausalRelationIndex(coder)
    index.observe(token_index=3, entity="mira", predicate="owns", value="violin")
    index.observe(token_index=8, entity="bob", predicate="holds", value="lantern")
    selection = index.select(query_index=10, entity="mira", predicate="possesses")
    assert selection.values == ("violin",)


def test_metrics_are_separate_and_baselines_can_fail():
    report = benchmark(lengths=(8, 64), trials=15, seed=17, steps=120)
    assert report["training"]["train_lexical_accuracy"] == 1.0
    for row in report["rows"]:
        assert row["learned_recall_at_k"] == 1.0
        assert row["raw_predicate_recall_at_k"] == 0.0
        assert row["oracle_recall_at_k"] == 1.0
        assert row["mean_index_probes"] == 1.0
        assert row["mean_postings_visited"] == 1.0
        assert row["mean_ingest_writes"] == row["facts"]
    assert report["rows"][1]["relation_only_recall_at_k"] < 0.5
