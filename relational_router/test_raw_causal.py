"""LRR-003 raw-causal, answer-only training and no future-token leakage gates."""
from __future__ import annotations

import random

import pytest
import torch

from .attention_raw import CausalAttentionPointer
from .raw_causal import (
    ALIASES, CausalRoleIndexer, VOCAB, examples_to_batch, generate, tokenize,
)
from .symbolic_raw import SymbolicRawIndex, parse_raw


def test_train_dev_and_test_are_disjoint_ordered_pairs():
    for split in ("train", "dev", "test"):
        for seed in range(100):
            ex = generate(seed, 6, split)
            aliases = ALIASES[ex.family]
            pair = (ex.fact_alias, ex.question_alias)
            if split == "train":
                assert pair != (aliases[0], aliases[1])
                assert pair != (aliases[2], aliases[3])
            elif split == "dev":
                assert pair == (aliases[0], aliases[1])
            else:
                assert pair == (aliases[2], aliases[3])


def test_generated_text_has_no_fact_tags_or_gold_position_hints():
    ex = generate(300, 32, "test")
    words = tokenize(ex.text)
    assert "<F>" not in ex.text and "<Q>" not in ex.text
    assert words[-1] == "?"
    assert words.count(ex.answer) == 1
    assert words[ex.gold_index] == ex.answer
    assert ex.gold_index < len(words) - 5
    assert len([w for w in words if w.startswith("value_")]) == 32


def test_all_generated_values_are_out_of_vocab_and_no_future_gold():
    data = [generate(seed, 6 + seed % 3, "train") for seed in range(10, 18)]
    ids, answers, queries = examples_to_batch(data)
    for i, ex in enumerate(data):
        assert ids[i, answers[i]].item() == VOCAB.index("<unk>")
        assert answers[i].item() < queries[i].item()
        assert len(ex.tokens) == 4 * (6 + (10+i) % 3) + 5


def test_causal_local_encoder_no_effect_from_future_suffix():
    torch.manual_seed(7)
    model = CausalRoleIndexer().eval()
    a = tokenize("mira owns value_1 .")
    b = tokenize("mira owns value_1 . niko fixes value_2 . what does mira possess ?")
    ia = torch.tensor([[model_id(t) for t in a]])
    ib = torch.tensor([[model_id(t) for t in b]])
    with torch.no_grad():
        short = model(ia)
        long = model(ib)
    for s, l in zip(short[:4], long[:4]):
        assert torch.allclose(s, l[:, :len(a)], atol=1e-6)


def model_id(word):
    from .raw_causal import IDS
    return IDS.get(word, IDS["<unk>"])


def test_symbolic_same_answer_labels_recover_relations_without_gold_family():
    symbol = SymbolicRawIndex()
    for n in range(60):
        sample = generate(900_000 + n, 8, "train")
        symbol.observe_training(sample.text, sample.answer)
    example = generate(720, 32, "test")
    facts, entity, predicate = parse_raw(example.text)
    assert len(facts) == 32
    assert entity and predicate
    assert symbol.trainable_parameters == 0
    assert symbol.training_documents == 60
    assert symbol.route(example.text)["index_probes"] == 1


def test_neural_route_receives_only_text_and_produces_prior_indices():
    torch.manual_seed(13)
    model = CausalRoleIndexer()
    example = generate(321, 16, "test")
    result = model.route(example.text)
    assert result["token_index"] is None or (
        0 <= result["token_index"] < len(example.tokens) - 1)
    assert result["index_probes"] in (0, 1)
    assert isinstance(result["admissions"], int)
    assert "gold_index" not in model.route.__code__.co_varnames


def test_matching_parameter_budget_and_causal_attention():
    model = CausalRoleIndexer()
    attn = CausalAttentionPointer()
    n1 = sum(p.numel() for p in model.parameters())
    n2 = sum(p.numel() for p in attn.parameters())
    assert n1 == 23741
    assert abs(n2-n1)/n1 < 0.05
    example = generate(13, 8, "dev")
    result = attn.route(example.text)
    assert 0 <= result["token_index"] < len(example.tokens) - 1


@pytest.mark.parametrize("split", ["train", "dev", "test"])
def test_conflict_update_has_only_one_newest_gold_and_valid_positions(split):
    ex = generate(700, 16, split, update=True)
    assert ex.tokens[ex.gold_index] == ex.answer
    assert len([w for w in ex.tokens if w.startswith("value_")]) == 16
    assert ex.gold_index == len(ex.tokens) - 7
    facts, _, _ = parse_raw(ex.text)
    assert facts[-1][2] == ex.answer


def test_unseen_entity_is_literal_surface_not_externally_assigned_identity():
    ex = generate(3131, 8, "test", new_entity=True)
    facts, entity, _ = parse_raw(ex.text)
    assert entity.startswith("nonce_")
    assert any(f[0] == entity for f in facts)
    assert entity not in VOCAB
