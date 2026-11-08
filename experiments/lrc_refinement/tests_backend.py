"""Structural tests for refinement, without loading a learned model or data."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import unittest

import numpy as np

from experiments.lrc_route_repair.index import BoundedKeyIndex
from .backend import RefinedIndex, choose_probes, prefix_bits


class PrefixTests(unittest.TestCase):
    def test_packing_threshold_equality_and_nested_depth(self):
        projected = np.array([[0.0, -1.0, 2.0, -3.0], [-1.0, 0.0, -1.0, 0.0]])
        cuts = np.zeros(4)
        np.testing.assert_array_equal(prefix_bits(projected, cuts, 4), [5, 10])
        np.testing.assert_array_equal(prefix_bits(projected, cuts, 0), [0, 0])
        deepest = prefix_bits(projected, cuts, 4)
        for depth in range(5):
            np.testing.assert_array_equal(
                prefix_bits(projected, cuts, depth), deepest & np.uint64((1 << depth) - 1))

    def test_uint64_last_bit_and_leading_broadcast(self):
        self.assertEqual(int(prefix_bits(np.ones(64), np.zeros(64), 64)), (1 << 64) - 1)
        projected = np.array([[1.0, -1.0], [1.0, -1.0]])
        cuts = np.array([[0.0, 0.0], [2.0, -2.0]])
        np.testing.assert_array_equal(prefix_bits(projected, cuts, 2), [1, 2])
        self.assertEqual(prefix_bits(np.ones((0, 4)), np.zeros(4), 2).shape, (0,))

    def test_invalid_descriptors_fail_before_routing(self):
        for depth in (-1, 65):
            with self.subTest(depth=depth), self.assertRaises(ValueError):
                prefix_bits(np.zeros(64), np.zeros(64), depth)
        with self.assertRaises(TypeError):
            prefix_bits([0.0], [0.0], True)
        with self.assertRaises(ValueError):
            prefix_bits([0.0], [0.0, 0.0], 1)
        with self.assertRaises(ValueError):
            prefix_bits([float("nan")], [0.0], 1)
        with self.assertRaises(ValueError):
            prefix_bits([0.0], [0.0], 2)


class ProbeTests(unittest.TestCase):
    def test_two_coarse_preserves_reader_order_and_deduplicates(self):
        query = np.array([0.2, -0.1, 1.2])
        cuts = np.zeros((16, 3))
        cuts[2] = [0.3, -0.2, 1.3]
        self.assertEqual(choose_probes((7, 2), query, cuts, 3, "two_coarse"), ((7, 5), (2, 2)))
        self.assertEqual(choose_probes((7, 7), query, cuts, 3, "two_coarse"), ((7, 5),))
        self.assertEqual(choose_probes((7, 2), query, cuts, 0, "two_coarse"), ((7, 0), (2, 0)))

    def test_sibling_uses_least_margin_and_lowest_index_tie(self):
        cuts = np.zeros((16, 3))
        query = np.array([0.2, -0.2, 1.0])
        # The equal margin in bits 0 and 1 must select bit 0, not a record ID.
        self.assertEqual(choose_probes((9, 3), query, cuts, 3, "sibling"), ((9, 5), (9, 4)))
        query = np.array([0.3, -0.1, 1.0])
        self.assertEqual(choose_probes((9, 3), query, cuts, 3, "sibling"), ((9, 5), (9, 7)))
        self.assertEqual(choose_probes((9, 3), query, cuts, 0, "sibling"), ((9, 0),))

    def test_no_occupancy_filter_or_replacement_probe(self):
        cuts = np.zeros((16, 2))
        requested = choose_probes((6, 2), np.ones(2), cuts, 2, "two_coarse")
        index = RefinedIndex([2, 2, 9, 9], [3, 3, 3, 3])
        hit = index.gather(requested)
        self.assertEqual(requested, ((6, 3), (2, 3)))
        self.assertEqual(hit.candidates, (0, 1))
        self.assertEqual((hit.probes, hit.empty, hit.overflow), (2, 1, 0))
        self.assertNotIn((9, 3), requested)

    def test_policy_rejects_hidden_extra_coarse_probes(self):
        cuts = np.zeros((16, 2))
        with self.assertRaises(ValueError):
            choose_probes((1, 2, 3), np.ones(2), cuts, 2)
        with self.assertRaises(ValueError):
            choose_probes((), np.ones(2), cuts, 2)
        with self.assertRaises(ValueError):
            choose_probes((16, 1), np.ones(2), cuts, 2)
        with self.assertRaises(ValueError):
            choose_probes((1, 2), np.ones(2), cuts, 2, "occupancy")


class IndexTests(unittest.TestCase):
    def test_depth_zero_matches_native_index_including_overflow(self):
        generator = np.random.default_rng(72011)
        for size in (0, 1, 16, 64):
            codes = generator.integers(0, 8, size=size)
            for capacity in (0, 1, 2):
                native = BoundedKeyIndex(codes, max_candidates=capacity)
                refined = RefinedIndex(codes, np.zeros(size, dtype=np.uint64), capacity=capacity)
                for code in range(10):
                    with self.subTest(size=size, capacity=capacity, code=code):
                        before, after = native.lookup(code), refined.lookup(code, 0)
                        self.assertEqual(before.candidates, after.candidates)
                        self.assertEqual(before.bucket_size, after.bucket_size)
                        self.assertEqual(before.overflow, after.overflow)
                        self.assertEqual(before.empty, after.empty)

    def test_overflow_is_whole_bucket_miss_never_an_ordinal_prefix(self):
        index = RefinedIndex([4, 4, 4, 2, 2], [0, 0, 0, 1, 1])
        hit = index.gather(((4, 0), (2, 1)))
        self.assertEqual(index.buckets[(4, 0)], (0, 1, 2))
        self.assertEqual(hit.lookups[0].candidates, ())
        self.assertEqual(hit.candidates, (3, 4))
        self.assertEqual((hit.probes, hit.empty, hit.overflow, hit.bucketrefs, hit.returned),
                         (2, 0, 1, 5, 2))

    def test_arbitrary_permutation_preserves_eligible_occurrences(self):
        codes = np.array([4, 4, 4, 2, 2, 7, 9])
        prefixes = np.array([0, 0, 0, 1, 1, 3, 0])
        permutation = np.array([2, 5, 4, 0, 6, 1, 3])
        inverse = np.argsort(permutation)
        self.assertFalse(np.array_equal(permutation, inverse))
        probes = ((4, 0), (2, 1))
        original = RefinedIndex(codes, prefixes).gather(probes)
        permuted = RefinedIndex(codes[permutation], prefixes[permutation]).gather(probes)
        self.assertEqual(set(original.candidates), {int(permutation[i]) for i in permuted.candidates})
        self.assertEqual({int(inverse[i]) for i in original.candidates}, set(permuted.candidates))
        self.assertEqual((original.probes, original.empty, original.overflow, original.bucketrefs),
                         (permuted.probes, permuted.empty, permuted.overflow, permuted.bucketrefs))

    def test_budget_deduplication_and_immutable_audit_view(self):
        index = RefinedIndex([1, 1, 2, 2], [0, 0, 1, 1])
        hit = index.gather(((1, 0), (2, 1)))
        self.assertEqual((hit.returned, hit.probes), (4, 2))
        repeated = index.gather(((1, 0), (1, 0)))
        self.assertEqual((repeated.candidates, repeated.probes), ((0, 1), 1))
        self.assertEqual(index.gather(()).candidates, ())
        with self.assertRaises(ValueError):
            index.gather(((1, 0), (2, 1), (3, 2)))
        with self.assertRaises(ValueError):
            RefinedIndex([1], [0], capacity=3)
        with self.assertRaises(TypeError):
            index.buckets[(1, 0)] = (100,)
        with self.assertRaises(FrozenInstanceError):
            index.capacity = 9
        with self.assertRaises(ValueError):
            RefinedIndex([1, 2], [0])


if __name__ == "__main__":
    unittest.main()
