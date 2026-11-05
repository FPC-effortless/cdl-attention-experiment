"""Mechanistic correctness of incremental LRR causal receptive-field cache."""
import copy

import pytest
import torch

from .raw_causal import CausalRoleIndexer, generate, tokenize
from .stream_cache import CachedCausalIndex


def split_raw_example(example):
    tokens = example.tokens
    assert tokens[-5:] == ("what", "does", tokens[-3], tokens[-2], "?")
    return " ".join(tokens[:-5]), " ".join(tokens[-5:])


@pytest.mark.parametrize("facts", [6, 32, 128])
def test_append_in_chunks_equals_batch_ingest_and_one_query(facts):
    torch.manual_seed(13)
    model = CausalRoleIndexer().eval()
    example = generate(101 + facts, facts, "test")
    history, question = split_raw_example(example)

    batch = CachedCausalIndex(model)
    receipt = batch.append(history)
    assert receipt["observed_total"] == 4*facts
    cached = batch.query(question)
    assert cached["encoded_token_count"] == min(14, 4*facts) + 5
    assert cached["observed_history_length"] == 4*facts
    assert cached["index_probes"] == 1

    chunked = CachedCausalIndex(model)
    tokens = tokenize(history)
    for i in range(0, len(tokens), 4):
        chunked.append(" ".join(tokens[i:i+4]))
    assert chunked.history_tokens == batch.history_tokens
    assert chunked.postings == batch.postings
    assert chunked.write_count == batch.write_count
    assert chunked.query(question)["token_indices"] == cached["token_indices"]

    # The last question's causal context, and hence its encoded address,
    # must match a full recomputation even when full recomputation is long.
    full = model.route(example.text)
    assert cached["query_address"] == full["query_address"]
    assert all(i < len(batch.history_tokens) for i in cached["token_indices"])


def test_repeated_questions_read_without_rebuilding_or_mutating_history():
    torch.manual_seed(19)
    m = CausalRoleIndexer().eval()
    e = generate(55, 64, "test")
    history, question = split_raw_example(e)
    state = CachedCausalIndex(m)
    state.append(history)
    original_postings = copy.deepcopy(state.postings)
    original_tokens = tuple(state.history_tokens)
    results = [state.query(question) for _ in range(6)]
    assert all(x["token_indices"] == results[0]["token_indices"] for x in results)
    assert state.postings == original_postings
    assert tuple(state.history_tokens) == original_tokens
    assert all(x["encoded_token_count"] == 19 for x in results)
    assert state.tail_size == 14


def test_invalid_queries_and_history_fail_closed():
    torch.manual_seed(21)
    state = CachedCausalIndex(CausalRoleIndexer())
    with pytest.raises(ValueError):
        state.append("what does mira own ?")
    with pytest.raises(ValueError):
        state.query("what does mira own")
    with pytest.raises(ValueError):
        state.query("what does mira own ?", k=0)
