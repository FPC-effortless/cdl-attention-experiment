"""Executable semantics for causal records and compositional multi-hop programs.

No oracle labels are passed to the neural model by this module. It is a reference
specification and proposed *expressivity target*, not an empirical learning claim.

The relation maps are fully available only in this oracle/reference layer. In a
learned architecture they must be induced from raw input without leakage.
"""
from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_left, bisect_right
from math import log, isfinite
from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np

@dataclass(frozen=True)
class Occurrence:
    position: int
    namespace: int
    address: int


def causal_candidates(history_newest_first: Sequence[Occurrence], *,
                      query_position: int, namespace: int, address: int,
                      cap: int) -> tuple[Occurrence, ...]:
    """Denotational meaning of a capped read; never reads current/future records."""
    if cap < 1 or query_position < 0:
        raise ValueError('invalid budget or query position')
    relevant = (o for o in history_newest_first
                if o.position < query_position and o.namespace == namespace
                and o.address == address)
    matches = []
    for occurrence in relevant:
        matches.append(occurrence)
        if len(matches) == cap:
            break
    return tuple(matches)


class VersionedRelations:
    """Explicit typed, versioned oracle graph: distinct relations per hop.

    Each (relation, subject) has independently versioned *functional* edges.
    Updates are sorted by global time, and lookup at time T observes only <=T.
    The caller may choose a historical snapshot explicitly.
    """
    def __init__(self):
        self._edges: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        self._last_time = -1

    def append(self, time: int, relation: int, subject: int, destination: int):
        if time <= self._last_time or min(time, relation, subject, destination) < 0:
            raise ValueError('writes must be globally ordered, nonnegative')
        self._edges[relation, subject].append((time, destination))
        self._last_time = time

    def resolve(self, relation: int, subject: int, *, as_of: int) -> int | None:
        versions = self._edges.get((relation, subject), ())
        # Binary search over time; O(log versions). tuples are ordered by time.
        j = bisect_right(versions, (as_of, float('inf'))) - 1
        return versions[j][1] if j >= 0 else None

    def follow(self, start: int, relations: Sequence[int], *, as_of: int) -> int | None:
        current = start
        for r in relations:
            if current is None:
                return None
            current = self.resolve(r, current, as_of=as_of)
        return current


def follow_tables(start: int, relations: Sequence[int], tables: Mapping[int, np.ndarray]) -> int:
    """Serial functional composition with O(hops) dependent reads."""
    node = int(start)
    for r in relations:
        node = int(tables[r][node])
    return node


def compile_parallel(relations: Sequence[int], tables: Mapping[int, np.ndarray],
                     *, count: int) -> np.ndarray:
    """Parallelizable associative reduction of *fully materialized* functions.

    If f and g are successor arrays, (g∘f)[v] = g[f[v]]. Each tree level
    composes disjoint pairs; levels have O(log H) dependency depth with O(N H)
    total array work (assuming parallel array gathers). No promise of improved
    neural routing, no free preprocessing for frequently updated relation maps.
    """
    if count < 1:
        raise ValueError('entity count must be positive')
    identity = np.arange(count, dtype=np.int64)
    if not relations:
        return identity
    funcs = []
    for r in relations:
        f = np.asarray(tables[r], dtype=np.int64)
        if f.shape != (count,) or (f < 0).any() or (f >= count).any():
            raise ValueError('each relation must be total on 0..count-1')
        funcs.append(f.copy())
    while len(funcs) > 1:
        nxt = []
        for i in range(0, len(funcs), 2):
            if i + 1 == len(funcs):
                nxt.append(funcs[i])
            else:
                first, second = funcs[i], funcs[i + 1]
                nxt.append(second[first])
        funcs = nxt
    return funcs[0]


def schedule_cost(*, inference_ms: float, baseline_inference_ms: float,
                  training_ms: float, baseline_training_ms: float,
                  inference_weight: float = 12.0) -> float:
    """User-selected 12:1 log-ratio *preference*, meaningful only above quality gates.

    It penalizes multiplicative slowdowns, comparing dimensionless ratios.
    Inference speed dominates training speed, but never compensates for failed
    causal/expressivity/accuracy requirements.
    """
    if (not all(isfinite(v) for v in (inference_ms, baseline_inference_ms,
                training_ms, baseline_training_ms, inference_weight))
            or min(inference_ms, baseline_inference_ms, training_ms, baseline_training_ms) <= 0):
        raise ValueError('timings must be strictly positive')
    if inference_weight < 0:
        raise ValueError('inference weight must be nonnegative')
    return (inference_weight * log(inference_ms / baseline_inference_ms)
            + log(training_ms / baseline_training_ms))
