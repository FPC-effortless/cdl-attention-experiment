"""Exact executable multi-hop semantics for a materialized FUNCTION graph.

This is an oracle/reference execution backend, not a trained text parser or
learned function compiler. It compares fair vectorized serial batching with
balanced parallel function composition, and invalidates caches on every update.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Sequence
from types import MappingProxyType
import numpy as np
from .math_spec import compile_parallel


def serial_batch(starts: Sequence[int] | np.ndarray, path: Sequence[int],
                 tables: Mapping[int, np.ndarray]) -> np.ndarray:
    result = np.asarray(starts, dtype=np.int64).copy()
    for relation in path:
        result = tables[relation][result]
    return result


@dataclass(frozen=True)
class ExecutionInfo:
    mode: str
    epoch: int
    compiled_cache_hit: bool
    predicted_serial_gathers: int
    predicted_compile_gathers: int


class VersionedProgramEngine:
    """Incremental graph updates with epoch-dependent compiled programs.

    Auto dispatch uses a transparent, configurable work heuristic. The priority
    remains correctness; any target-hardware throughput claim needs measurement.
    """
    def __init__(self, tables: Mapping[int, np.ndarray]):
        if not tables:
            raise ValueError('at least one relation map is needed')
        self._tables = {int(r): np.asarray(values, dtype=np.int64).copy()
                       for r, values in tables.items()}
        n = len(next(iter(self._tables.values())))
        if n < 1:
            raise ValueError('empty entity universe')
        for relation, arr in self._tables.items():
            if relation < 0 or arr.shape != (n,) or (arr < 0).any() or (arr >= n).any():
                raise ValueError('relations must be total maps over 0..n-1')
        self.entities = n
        self.epoch = 0
        self._compiled: dict[tuple[int, ...], np.ndarray] = {}
        self.compile_count = 0

    @property
    def tables(self):
        # Snapshot copies: external code cannot silently invalidate cache via
        # in-place edits to public table values. Execution uses private arrays.
        result = {r: values.copy() for r, values in self._tables.items()}
        for values in result.values():
            values.flags.writeable = False
        return MappingProxyType(result)

    def update(self, relation: int, subject: int, dest: int):
        if relation not in self._tables or not (0 <= subject < self.entities and 0 <= dest < self.entities):
            raise ValueError('invalid relation update')
        self._tables[relation][subject] = dest
        self.epoch += 1
        # Unrelated programs remain valid after updates, improving reuse.
        self._compiled = {word: plan for word, plan in self._compiled.items()
                          if relation not in word}

    def execute(self, starts: Sequence[int] | np.ndarray, path: Sequence[int],
                *, mode: str = 'auto', anticipated_queries: int = 0):
        if mode not in ('auto', 'serial', 'compiled'):
            raise ValueError('unsupported execution mode')
        word = tuple(path)
        if any(r not in self._tables for r in word):
            raise ValueError('unknown relation')
        x = np.asarray(starts, dtype=np.int64)
        if x.ndim != 1 or (x < 0).any() or (x >= self.entities).any():
            raise ValueError('starts must be in the entity universe')
        count = len(x)
        work_serial = count * len(word)
        # Each composition level touches roughly H * |V| entries; effective
        # cost is hardware-dependent. Treat these as work proxies, not latency.
        work_compile = self.entities * max(0, len(word) - 1) + count
        cached = word in self._compiled
        if mode == 'auto':
            if cached:
                mode = 'compiled'
            elif len(word) < 2 or work_compile >= (count + anticipated_queries) * len(word):
                mode = 'serial'
            else:
                mode = 'compiled'
        if mode == 'serial':
            answer = serial_batch(x, word, self._tables)
        else:
            if not cached:
                self._compiled[word] = compile_parallel(word, self._tables, count=self.entities)
                self.compile_count += 1
            answer = self._compiled[word][x]
        return answer, ExecutionInfo(mode, self.epoch, cached,
                                     work_serial, work_compile)
