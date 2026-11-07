import unittest
import numpy as np
import random

from relational_composition.math_spec import (
    Occurrence, causal_candidates, VersionedRelations,
    follow_tables, compile_parallel, schedule_cost)
from relational_composition.index import CausalIndex

class MathematicalSpecTests(unittest.TestCase):
    def test_semantics_matches_index_and_cap(self):
        rng = random.Random(1234)
        for seed in range(30):
            length = rng.randrange(1, 140)
            cap = rng.randrange(1, 9)
            writes = [Occurrence(i, rng.randrange(4), rng.randrange(8)) for i in range(length)]
            index = CausalIndex(cap)
            for w in writes:
                index.append(w.position, w.namespace, w.address)
            for _ in range(20):
                t, r, a = rng.randrange(length + 1), rng.randrange(4), rng.randrange(8)
                selected = causal_candidates(writes[::-1], query_position=t,
                                             namespace=r, address=a, cap=cap)
                actual = index.lookup(r, a, t)
                self.assertEqual(tuple(w.position for w in selected), actual.candidates)
                self.assertLessEqual(len(selected), cap)
                self.assertTrue(all(w.position < t for w in selected))

    def test_multihop_and_parallel_compiler(self):
        rng = np.random.default_rng(72)
        n = 101
        maps = {i: rng.permutation(n) for i in range(4)}
        for h in [0, 1, 2, 3, 4, 7, 8, 11, 17, 32]:
            rels = [int(x) for x in rng.integers(0, 4, size=h)]
            compiled = compile_parallel(rels, maps, count=n)
            for start in range(n):
                self.assertEqual(int(compiled[start]), follow_tables(start, rels, maps))

    def test_explicit_snapshots_and_distinct_relation_roles(self):
        graph = VersionedRelations()
        graph.append(1, 0, 0, 1)
        graph.append(2, 1, 1, 3)
        graph.append(3, 0, 0, 2)
        graph.append(4, 1, 2, 4)
        self.assertEqual(graph.follow(0, [0, 1], as_of=2), 3)
        self.assertIsNone(graph.follow(0, [0, 1], as_of=3))
        self.assertEqual(graph.follow(0, [0, 1], as_of=4), 4)
        with self.assertRaises(ValueError):
            graph.append(4, 0, 0, 2)

    def test_weighted_objective(self):
        a = schedule_cost(inference_ms=5, baseline_inference_ms=10,
                          training_ms=20, baseline_training_ms=10)
        b = schedule_cost(inference_ms=10, baseline_inference_ms=10,
                          training_ms=5, baseline_training_ms=10)
        self.assertLess(a, b)
        with self.assertRaises(ValueError):
            schedule_cost(inference_ms=0, baseline_inference_ms=10,
                          training_ms=20, baseline_training_ms=10)

if __name__ == '__main__':
    unittest.main()
