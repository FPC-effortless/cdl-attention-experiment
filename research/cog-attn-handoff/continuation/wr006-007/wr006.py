"""Persistent valid-time tree with one root per entity transaction."""
from array import array
from bisect import bisect_right, insort
import json
import statistics
import time
from pathlib import Path

import numpy as np


class History:
    def __init__(self, horizon):
        self.horizon = horizon
        self.left = array("I", [0])
        self.right = array("I", [0])
        self.value = array("i", [-1])
        self.times = {}
        self.roots = {}

    def insert(self, entity, valid, tx, value):
        roots = self.roots.setdefault(entity, array("I"))
        times = self.times.setdefault(entity, array("I"))
        roots.append(self._insert(roots[-1] if roots else 0, 0, self.horizon, valid, value))
        times.append(tx)

    def _insert(self, node, lo, hi, valid, value):
        left, right = self.left[node], self.right[node]
        if hi - lo > 1:
            mid = (lo + hi) // 2
            if valid < mid:
                left = self._insert(left, lo, mid, valid, value)
            else:
                right = self._insert(right, mid, hi, valid, value)
        result = len(self.left)
        self.left.append(left)
        self.right.append(right)
        self.value.append(value if hi - lo == 1 else -1)
        return result

    def lookup(self, entity, valid, cutoff):
        times = self.times.get(entity, ())
        i = bisect_right(times, cutoff) - 1
        if i < 0 or valid < 0:
            return -1, 0
        visits = 0

        def predecessor(node, lo, hi):
            nonlocal visits
            if not node or lo > valid:
                return -1
            visits += 1
            if hi - lo == 1:
                return self.value[node]
            mid = (lo + hi) // 2
            result = predecessor(self.right[node], mid, hi)
            return result if result >= 0 else predecessor(self.left[node], lo, mid)

        return predecessor(self.roots[entity][i], 0, self.horizon), visits

    def payload(self):
        arrays = [self.left, self.right, self.value, *self.times.values(), *self.roots.values()]
        return sum(len(a) * a.itemsize for a in arrays)


def scan(events, entity, valid, cutoff):
    best = (-1, -1, -1)
    for k, vt, tx, value in events:
        if k == entity and vt <= valid and tx <= cutoff and (vt, tx) > best[:2]:
            best = vt, tx, value
    return best[2]


def one_dimension(index, entity, valid, cutoff):
    rows = index[entity]
    i = bisect_right(rows, (valid, float("inf"), float("inf"))) - 1
    checks = 0
    while i >= 0:
        checks += 1
        if rows[i][1] <= cutoff:
            return rows[i][2], checks
        i -= 1
    return -1, checks


def benchmark(seed, size):
    rng = np.random.default_rng(seed)
    events = []
    for tx in range(size):
        entity = 0 if rng.random() < .75 else int(rng.integers(1, 32))
        valid = max(0, tx - int(rng.integers(1, size // 2))) if rng.random() < .30 else tx
        events.append((entity, valid, tx, tx))
    queries = [(0 if rng.random() < .75 else int(rng.integers(1, 32)),
                int(rng.integers(size)), int(rng.integers(size))) for _ in range(511)]
    queries += [(0, size - 1, 0)]
    index = [[] for _ in range(32)]
    tree = History(size)
    t = time.perf_counter_ns()
    for entity, valid, tx, value in events:
        insort(index[entity], (valid, tx, value))
    one_build = time.perf_counter_ns() - t
    t = time.perf_counter_ns()
    for event in events:
        tree.insert(*event)
    tree_build = time.perf_counter_ns() - t
    ref = [scan(events, *query) for query in queries]
    t = time.perf_counter_ns()
    one = [one_dimension(index, *query) for query in queries]
    one_time = time.perf_counter_ns() - t
    t = time.perf_counter_ns()
    two = [tree.lookup(*query) for query in queries]
    tree_time = time.perf_counter_ns() - t
    assert ref == [value for value, _ in one] == [value for value, _ in two]
    return dict(seed=seed, size=size, queries=len(queries), tree_payload_bytes=tree.payload(),
                one_payload_bytes=12 * size, one_build_ns=one_build, tree_build_ns=tree_build,
                one_query_ns=one_time, tree_query_ns=tree_time,
                one_checks=[count for _, count in one], tree_visits=[count for _, count in two])


def main():
    fixture = History(8)
    for event in [(0, 1, 0, 10), (0, 6, 1, 11), (0, 1, 2, 12), (0, 0, 3, 13)]:
        fixture.insert(*event)
    for valid, cutoff, value in [(1, 0, 10), (1, 2, 12), (0, 2, -1),
                                  (0, 3, 13), (7, 3, 11), (7, -1, -1)]:
        assert fixture.lookup(0, valid, cutoff)[0] == value
    rows = [benchmark(seed, size) for size in (128, 2048, 32768) for seed in range(201, 209)]
    for size in (128, 2048, 32768):
        group = [r for r in rows if r["size"] == size]
        print(size, max(max(r["one_checks"]) for r in group), max(max(r["tree_visits"]) for r in group),
              statistics.mean(r["one_query_ns"] for r in group) / statistics.mean(r["tree_query_ns"] for r in group),
              group[0]["tree_payload_bytes"], flush=True)
    Path(__file__).with_name("wr006.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
