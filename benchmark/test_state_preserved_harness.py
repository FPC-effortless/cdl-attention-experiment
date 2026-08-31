from __future__ import annotations

import random

import torch

from benchmark.state_preserved_harness import (
    TASKS,
    _carried_tokens,
    _query_only_tokens,
    make_batch,
    make_example,
    target_only_weights,
)


def test_all_tasks_are_deterministic() -> None:
    for task in TASKS:
        a = make_example(task, random.Random(1234))
        b = make_example(task, random.Random(1234))
        assert a == b
        context, query, target = a
        assert len(context) == 24
        assert len(query) == 7
        assert 8 <= target <= 63


def test_multi_hop_requires_two_edges() -> None:
    context, query, target = make_example("multi_hop", random.Random(44))
    start = query[1]
    triples = [context[i : i + 3] for i in range(0, 24, 3)]
    edges = [(x[1], x[2]) for x in triples if len(x) == 3 and x[0] == 0]
    first = [b for a, b in edges if a == start]
    assert first
    assert target not in first
    assert any(a in first and b == target for a, b in edges)


def test_carried_query_crosses_chunk_boundary() -> None:
    contexts, queries, targets = make_batch("single_key", 4, 11)
    carried = _carried_tokens(contexts, queries, targets)
    reset = _query_only_tokens(queries, targets)
    assert carried.shape == (4, 32)
    assert reset.shape == (4, 8)
    assert torch.equal(carried[:, :24], contexts)
    assert torch.equal(carried[:, 24:31], queries)
    assert torch.equal(carried[:, -1], targets)


def test_target_weight_only_trains_recall_token() -> None:
    contexts, queries, targets = make_batch("multi_key", 3, 23)
    tokens = _carried_tokens(contexts, queries, targets)
    w = target_only_weights(tokens)
    assert w.shape == (3, 31)
    assert torch.equal(w[:, -1], torch.ones(3))
    assert float(w[:, :-1].sum()) == 0.0


def test_shuffled_context_changes_stream_without_changing_query_or_label() -> None:
    contexts, queries, targets = make_batch("noisy_key", 8, 37)
    shuffled = torch.roll(contexts, shifts=1, dims=0)
    assert not torch.equal(contexts, shuffled)
    a = _carried_tokens(contexts, queries, targets)
    b = _carried_tokens(shuffled, queries, targets)
    assert torch.equal(a[:, 24:], b[:, 24:])
    assert not torch.equal(a[:, :24], b[:, :24])
