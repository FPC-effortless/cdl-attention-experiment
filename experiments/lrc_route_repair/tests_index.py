"""Contract tests for bounded, collision-aware occurrence lookup.

Run directly with ``python experiments/lrc_route_repair/tests_index.py``.
The stdlib suite does not require PyTorch; tensor integration is optional.
"""

from dataclasses import FrozenInstanceError
from itertools import permutations
import unittest

if __package__:
    from .index import BoundedKeyIndex
else:
    from index import BoundedKeyIndex


class BoundedKeyIndexTests(unittest.TestCase):
    def test_empty_index_and_missing_address(self):
        index = BoundedKeyIndex([], max_candidates=3)
        for code in (0, 42):
            result = index.lookup(code)
            self.assertEqual(result.candidates, ())
            self.assertEqual(result.bucket_size, 0)
            self.assertEqual(result.candidate_reads, 0)
            self.assertEqual(result.dropped_count, 0)
            self.assertTrue(result.empty)
            self.assertTrue(result.miss)
            self.assertFalse(result.overflow)
        self.assertEqual(len(index), 0)
        self.assertEqual(index.stats.empty_lookups, 2)

    def test_zero_is_a_valid_code(self):
        index = BoundedKeyIndex([0, 9, 0], max_candidates=2)
        self.assertEqual(index.lookup(0).candidates, (0, 2))
        self.assertEqual(index.lookup(9).candidates, (1,))
        self.assertTrue(index.lookup(1).empty)

    def test_duplicate_codes_at_boundary_return_every_occurrence(self):
        index = BoundedKeyIndex([5, 7, 5, 7, 5], max_candidates=3)
        result = index.lookup(5)
        self.assertEqual(result.candidates, (0, 2, 4))
        self.assertEqual(result.bucket_size, 3)
        self.assertEqual(result.candidate_reads, 3)
        self.assertFalse(result.overflow)
        self.assertFalse(result.miss)
        self.assertEqual(result.max_candidates, 3)
        self.assertEqual(result.probes, 1)

    def test_overflow_is_an_explicit_miss_and_retains_all_records_for_audit(self):
        index = BoundedKeyIndex([5, 7, 5, 5, 5], max_candidates=3)
        result = index.lookup(5)
        self.assertEqual(result.candidates, ())
        self.assertEqual(result.bucket_size, 4)
        self.assertTrue(result.overflow)
        self.assertTrue(result.miss)
        self.assertFalse(result.empty)
        self.assertEqual(result.candidate_reads, 0)
        self.assertEqual(result.dropped_count, 4)
        before_audit = index.stats
        self.assertEqual(index.audit_bucket(5), (0, 2, 3, 4))
        self.assertEqual(index.audit_bucket(8), ())
        self.assertEqual(index.stats, before_audit)
        self.assertEqual(sum(map(len, index.audit_buckets().values())), len(index))

    def test_zero_cap_disables_candidate_reads_without_losing_audit(self):
        index = BoundedKeyIndex([0, 0, 3], max_candidates=0)
        self.assertTrue(index.lookup(0).overflow)
        self.assertTrue(index.lookup(3).overflow)
        self.assertFalse(index.lookup(1).overflow)
        self.assertEqual(index.stats.candidate_reads, 0)
        self.assertEqual(index.audit_bucket(0), (0, 1))

    def test_invalid_codes_and_caps_fail_explicitly(self):
        for bad_value, error in (
            (-1, ValueError), (1.0, TypeError), (True, TypeError),
            ("1", TypeError), (None, TypeError), ([1], TypeError),
        ):
            with self.subTest(value=bad_value):
                with self.assertRaises(error):
                    BoundedKeyIndex([bad_value], max_candidates=2)
                with self.assertRaises(error):
                    BoundedKeyIndex([], max_candidates=bad_value)
                index = BoundedKeyIndex([1], max_candidates=2)
                with self.assertRaises(error):
                    index.lookup(bad_value)
                self.assertEqual(index.stats.lookups, 0)

    def test_input_is_snapshotted(self):
        codes = [2, 3, 2]
        index = BoundedKeyIndex(codes, max_candidates=3)
        codes[:] = [9, 9, 9, 9]
        self.assertEqual(index.lookup(2).candidates, (0, 2))
        self.assertTrue(index.lookup(9).empty)
        self.assertEqual(len(index), 3)

    def test_candidate_set_and_unique_content_winner_are_permutation_equivariant(self):
        codes = [5, 2, 5, 9, 2]
        # The resolver's unique scores stand for content attached to occurrences.
        scores = [0.2, -0.5, 0.9, 0.3, 0.7]
        original = BoundedKeyIndex(codes, max_candidates=2)
        for permutation in permutations(range(len(codes))):
            reordered = BoundedKeyIndex(
                [codes[position] for position in permutation], max_candidates=2
            )
            for code in (2, 5, 9, 10):
                baseline = original.lookup(code).candidates
                candidates = reordered.lookup(code).candidates
                mapped = {permutation[position] for position in candidates}
                self.assertEqual(mapped, set(baseline))
                if candidates:
                    winner = max(candidates, key=lambda p: scores[permutation[p]])
                    expected = max(baseline, key=lambda p: scores[p])
                    self.assertEqual(permutation[winner], expected)

    def test_overflow_does_not_choose_a_prefix_after_permutation(self):
        codes = [8, 8, 1, 8]
        for permutation in permutations(range(len(codes))):
            index = BoundedKeyIndex(
                [codes[position] for position in permutation], max_candidates=2
            )
            result = index.lookup(8)
            self.assertTrue(result.overflow)
            self.assertEqual(result.candidates, ())
            self.assertEqual(result.dropped_count, 3)

    def test_lookup_never_revisits_input(self):
        class ReadOnce:
            def __init__(self):
                self.used = False

            def __iter__(self):
                if self.used:
                    raise AssertionError("query revisited the input history")
                self.used = True
                yield from (0, 4, 0, 8)

        source = ReadOnce()
        index = BoundedKeyIndex(source, max_candidates=2)
        for _ in range(10):
            self.assertEqual(index.lookup(0).candidates, (0, 2))
            self.assertEqual(index.lookup(8).candidates, (3,))

    def test_lookup_uses_one_get_without_history_iteration(self):
        class NoScanDict(dict):
            gets = 0

            def get(self, key, default=None):
                self.gets += 1
                return super().get(key, default)

            def __iter__(self):
                raise AssertionError("query scanned the address table")

            def items(self):
                raise AssertionError("query scanned the address table")

            def values(self):
                raise AssertionError("query scanned the address table")

            def keys(self):
                raise AssertionError("query scanned the address table")

        index = BoundedKeyIndex(range(100_000), max_candidates=2)
        table = NoScanDict(index._buckets)
        index._buckets = table
        for code in (0, 50_000, 99_999, 100_001):
            before = table.gets
            result = index.lookup(code)
            self.assertEqual(table.gets - before, 1)
            self.assertLessEqual(result.candidate_reads, 2)
        self.assertEqual(index.stats.candidate_reads, 3)

    def test_overflow_path_reads_length_without_traversal_or_slicing(self):
        class LengthOnlyBucket:
            def __len__(self):
                return 1_000_000

            def __iter__(self):
                raise AssertionError("overflow traversed a bucket")

            def __getitem__(self, key):
                raise AssertionError("overflow accessed or sliced a bucket")

        index = BoundedKeyIndex([], max_candidates=4)
        index._buckets = {1: LengthOnlyBucket()}
        result = index.lookup(1)
        self.assertEqual(result.candidates, ())
        self.assertEqual(result.bucket_size, 1_000_000)
        self.assertEqual(result.dropped_count, 1_000_000)
        self.assertEqual(result.candidate_reads, 0)

    def test_stats_count_all_outcomes_and_reset_only_query_counters(self):
        index = BoundedKeyIndex([1, 1, 2, 2, 2, 3], max_candidates=2)
        snapshot = index.stats
        for code in (1, 2, 9, 3):
            index.lookup(code)
        stats = index.stats
        self.assertEqual(snapshot.lookups, 0)
        self.assertEqual(stats.stored_records, 6)
        self.assertEqual(stats.occupied_buckets, 3)
        self.assertEqual(stats.lookups, 4)
        self.assertEqual(stats.probes, 4)
        self.assertEqual(stats.candidate_reads, 3)
        self.assertEqual(stats.empty_lookups, 1)
        self.assertEqual(stats.overflow_lookups, 1)
        self.assertEqual(stats.dropped_candidates, 3)
        self.assertEqual(stats.max_returned, 2)
        index.reset_stats()
        self.assertEqual(index.stats, snapshot)
        self.assertEqual(index.lookup(1).candidates, (0, 1))

    def test_results_and_audit_view_are_immutable(self):
        index = BoundedKeyIndex([1, 1], max_candidates=2)
        result = index.lookup(1)
        with self.assertRaises(FrozenInstanceError):
            result.overflow = True
        with self.assertRaises(FrozenInstanceError):
            index.stats.lookups = 99
        with self.assertRaises(TypeError):
            index.audit_buckets()[1] = ()
        self.assertEqual(index.lookup(1).candidates, (0, 1))

    def test_array_conversion_occurs_once_and_requires_one_dimension(self):
        class ArrayRow:
            def __init__(self, ndim, values):
                self.ndim = ndim
                self.values = values
                self.conversions = 0

            def tolist(self):
                self.conversions += 1
                return self.values

        row = ArrayRow(1, [4, 6, 4])
        index = BoundedKeyIndex(row, max_candidates=2)
        self.assertEqual(index.lookup(4).candidates, (0, 2))
        self.assertEqual(row.conversions, 1)
        for rank in (0, 2):
            with self.assertRaises(ValueError):
                BoundedKeyIndex(ArrayRow(rank, [4]), max_candidates=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
