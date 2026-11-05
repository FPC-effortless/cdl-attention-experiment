"""Hard integrity gates for the LRR-004 locality and negative control protocol."""
import pytest
import torch

from .raw_causal import ALIASES, VOCAB, CausalRoleIndexer, examples_to_batch
from .lrr004_locality import (
    make_example, locality_loss, train_arm, KnownLexiconGuard,
    known_absent_challenge, unknown_predicate_challenge,
    distant_predicate_challenge,
)
from .symbolic_raw import parse_raw, SymbolicRawIndex
from .run_lrr004 import evaluate_one


def test_train_dev_test_ordered_pairs_are_disjoint():
    for seed in range(400):
        for split in ("train", "dev", "test"):
            e = make_example(seed + 2000, 6, split)
            aliases = ALIASES[e.family]
            pair = (e.fact_alias, e.question_alias)
            if split == "train":
                assert pair not in ((aliases[1], aliases[2]), (aliases[3], aliases[0]))
            elif split == "dev":
                assert pair == (aliases[3], aliases[0])
            else:
                assert pair == (aliases[1], aliases[2])
            assert e.tokens[e.gold_index] == e.answer
            assert len(e.tokens) == 4 * 6 + 5
            assert "<F>" not in e.text and "<Q>" not in e.text


def test_zero_locality_equals_frozen_source_loss_and_gradient():
    torch.manual_seed(4)
    m = CausalRoleIndexer()
    examples = [make_example(i+10000, 6, "train") for i in range(4)]
    x, answer, query = examples_to_batch(examples)
    original = m.answer_position_loss(x, answer, query, balance_weight=0.2)
    replicated, statistics = locality_loss(
        m, x, answer, query, distance_weight=0.0, balance_weight=0.2)
    assert torch.allclose(original, replicated, atol=1e-6, rtol=1e-6)
    old_gradient = torch.autograd.grad(
        original, m.relation_offset.weight, retain_graph=True)[0]
    new_gradient = torch.autograd.grad(
        replicated, m.relation_offset.weight)[0]
    assert torch.allclose(old_gradient, new_gradient, atol=1e-5, rtol=1e-5)
    assert 0 <= float(statistics["distance"]) <= 3


def test_locality_is_a_soft_prior_not_hard_role_labels():
    torch.manual_seed(7)
    m = CausalRoleIndexer()
    x, y, q = examples_to_batch([make_example(900, 7, "train")])
    base, _ = locality_loss(m, x, y, q, distance_weight=0.0)
    changed, _ = locality_loss(m, x, y, q, distance_weight=0.15)
    assert changed.item() >= base.item()
    assert torch.isfinite(changed)
    with pytest.raises(ValueError):
        locality_loss(m, x, y, q, distance_weight=-0.1)


def test_unknown_predicate_guard_rejects_without_neural_confidence_claim():
    m = CausalRoleIndexer()
    support = frozenset(word for words in ALIASES.values() for word in words)
    s = KnownLexiconGuard(m, support)
    e = make_example(55666, 32, "test")
    unknown = unknown_predicate_challenge(e)
    assert s.route(unknown)["token_index"] is None
    assert s.route(unknown)["abstained_reason"] == "predicate_not_in_training_lexicon"
    assert s.route(e.text)["token_index"] == m.route(e.text)["token_index"]
    assert "unseenverb" not in VOCAB
    with pytest.raises(ValueError):
        s.route("what does mira possess")


def test_known_absent_has_no_true_fact_and_guard_cannot_trivially_reject_it():
    e = make_example(31232, 64, "test")
    changed = known_absent_challenge(e)
    facts, entity, relation = parse_raw(changed)
    assert len(facts) == 64
    assert relation == e.question_alias
    assert not any(fact_entity == entity and predicate in ALIASES[e.family]
                   for fact_entity, predicate, _, _ in facts)
    support = frozenset(word for words in ALIASES.values() for word in words)
    m = CausalRoleIndexer()
    guarded = KnownLexiconGuard(m, support)
    assert guarded.route(changed)["abstained_reason"] is None


def test_distant_predicate_challenge_is_nonlocal_and_preserves_gold_value():
    e = make_example(909090, 32, "test")
    changed, new_gold = distant_predicate_challenge(e)
    tokens = changed.split()
    assert tokens[new_gold] == e.answer
    assert tokens[new_gold - 1] == "now"
    assert tokens[new_gold - 2] == e.fact_alias
    assert new_gold == e.gold_index + 1


def test_short_fresh_seed_runs_and_preserves_failures():
    m, meta, support = train_arm(987, distance_weight=0.15, steps=2, batch_size=4)
    assert meta["steps"] == 2 and meta["parameters"] == 23429
    assert 0 < len(support) <= 16
    symbolic = SymbolicRawIndex()
    for i in range(24):
        e = make_example(400000+i, 6, "train")
        symbolic.observe_training(e.text, e.answer)
    row = evaluate_one(987, 6, 3, m, frozenset(
        word for words in ALIASES.values() for word in words
    ), symbolic)
    assert row["learned"]["total"] == 3
    assert row["unknown_predicate"]["lexical_guard"]["false_answer_rate"] == 0
    assert 0 <= row["known_relation_absent"]["raw"]["false_answer_rate"] <= 1
    assert row["nonlocal_adverb_challenge"]["total"] == 3
