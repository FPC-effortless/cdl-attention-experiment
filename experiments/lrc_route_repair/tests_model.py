"""Mechanism tests of bounded content resolution, without fitting or task labels.

The fixtures set learned reader outputs to explicit vectors and use known
continuous candidate keys. They exercise the real index, gather, dot-product
resolver, dependent second read and output path. Passing these tests proves
those structural properties, not that training learns useful keys or routes.

Run with the original adapter and PyTorch available on PYTHONPATH:
    python -m unittest experiments.lrc_route_repair.tests_model -v
"""

from itertools import chain
import types
import unittest

import torch
from torch import nn

from .index import BoundedKeyIndex, LookupResult
from .model import ContentAddressedLRC, PreparedMemory


FIRST = torch.tensor([1.0, 0.0, 0.0, 0.0])
SECOND = torch.tensor([0.0, 0.0, 1.0, 0.0])
CODEBOOK = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],
    [0.8, 0.6, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.8, 0.6],
])


def controlled_model(*, second_key=SECOND, probes=2, capacity=2):
    """Use exact query vectors, retaining the production resolution code."""
    model = ContentAddressedLRC(width=4, codes=4, probes=probes, capacity=capacity)
    model.router.codebook.copy_(CODEBOOK)
    model.answer = nn.Identity()
    model.eval()
    bridge_values = []

    def query_first(router, query):
        return torch.zeros(len(query), 4), FIRST.expand(len(query), -1)

    def query_second(router, state, value):
        bridge_values.append(value.detach().clone())
        return value if second_key is None else second_key

    model.router.query_first = types.MethodType(query_first, model.router)
    model.router.query_second = types.MethodType(query_second, model.router)
    return model, bridge_values


def prepared(keys, codes, *, values=None, capacity=2, divisor=1):
    keys = torch.as_tensor(keys, dtype=torch.float32)
    codes = torch.as_tensor(codes, dtype=torch.long)
    if keys.ndim == 2:
        keys = keys.unsqueeze(0)
    if codes.ndim == 1:
        codes = codes.unsqueeze(0)
    if values is None:
        values = torch.arange(keys.numel(), dtype=torch.float32).reshape(keys.shape)
    else:
        values = torch.as_tensor(values, dtype=torch.float32)
        if values.ndim == 2:
            values = values.unsqueeze(0)
    return PreparedMemory(
        values, keys, codes,
        [BoundedKeyIndex(row, max_candidates=capacity) for row in codes],
        divisor,
    )


def collision_fixture():
    """Two eligible buckets per hop, two distinct fine keys per bucket."""
    keys = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.8, -0.6, 0.0, 0.0],
        [0.8, 0.6, 0.0, 0.0],
        [0.6, 0.8, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.8, -0.6],
        [0.0, 0.0, 0.8, 0.6],
        [0.0, 0.0, 0.6, 0.8],
    ])
    codes = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    # A successful first read supplies SECOND as its bridge, not an oracle label.
    values = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 32
    values[0] = SECOND
    return keys, codes, values


class CountedKeys:
    """Allow only bounded occurrence gathers and count actual dot products.

    This wrapper independently records the rows presented to matrix multiply;
    trusting RouteOutput.inspected alone would miss a hidden dense score path.
    """

    def __init__(self, tensor, *, cap=4, allowed=None):
        self.tensor = tensor
        self.cap = cap
        self.allowed = allowed
        self.gathers = []
        self.scores = []

    def __getitem__(self, item):
        if not isinstance(item, tuple) or len(item) != 2:
            raise AssertionError("resolver attempted an unbounded key access")
        row, positions = item
        if not isinstance(row, int) or not isinstance(positions, list):
            raise AssertionError("resolver must gather explicit occurrence IDs")
        if not 0 < len(positions) <= self.cap:
            raise AssertionError("resolver exceeded the candidate bound")
        if self.allowed is not None and not set(positions) <= self.allowed:
            raise AssertionError("resolver inspected a noncandidate occurrence")
        self.gathers.append((row, tuple(positions)))
        return CountedCandidateMatrix(self.tensor[row, positions], row, positions, self.scores)


class CountedCandidateMatrix:
    def __init__(self, tensor, row, positions, scores):
        self.tensor = tensor
        self.row = row
        self.positions = tuple(positions)
        self.scores = scores

    def __matmul__(self, read_key):
        result = self.tensor @ read_key
        self.scores.append((self.row, self.positions, result.detach().clone()))
        return result


class ContentAddressedLRCTests(unittest.TestCase):
    @staticmethod
    def query(batch=1):
        return torch.zeros(batch, 12, dtype=torch.long)

    def assert_null_second(self, output):
        self.assertEqual(output.selected[0].tolist(), [-1, -1])
        self.assertEqual(output.read_codes[0, 1].tolist(), [-1, -1])
        for tensor in (output.probes, output.inspected, output.returned,
                       output.overflow, output.bucket_size, output.ties):
            self.assertEqual(int(tensor[0, 1]), 0)
        self.assertTrue(torch.equal(output.logits[0], torch.zeros(4)))

    def test_candidate_content_changes_real_score_and_winner_at_fixed_query(self):
        model, _ = controlled_model(second_key=FIRST)
        keys = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.8, -0.6, 0.0, 0.0]])
        values = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
        memory = prepared(keys, [0, 0], values=values)
        original_keys = CountedKeys(memory.keys)
        memory.keys = original_keys
        first = model.query_memory(memory, self.query())
        changed = prepared(keys.flip(0), [0, 0], values=values)
        changed_keys = CountedKeys(changed.keys)
        changed.keys = changed_keys
        second = model.query_memory(changed, self.query())

        self.assertEqual(first.selected.tolist(), [[0, 0]])
        self.assertEqual(second.selected.tolist(), [[1, 1]])
        self.assertTrue(torch.equal(first.logits[0], values[0]))
        self.assertTrue(torch.equal(second.logits[0], values[1]))
        torch.testing.assert_close(original_keys.scores[0][2], torch.tensor([1.0, 0.8]))
        torch.testing.assert_close(changed_keys.scores[0][2], torch.tensor([0.8, 1.0]))
        self.assertTrue(torch.equal(first.read_codes, second.read_codes))
        self.assertTrue(torch.equal(first.writer_codes, second.writer_codes))

    def test_two_hop_occurrences_follow_permutation_and_bridge_content(self):
        keys, codes, values = collision_fixture()
        model, bridges = controlled_model(second_key=None)
        baseline = model.query_memory(prepared(keys, codes, values=values), self.query())
        self.assertEqual(baseline.selected.tolist(), [[0, 4]])
        rng = torch.Generator().manual_seed(1701)
        orders = [torch.arange(7, -1, -1)]
        orders += [torch.arange(8).roll(shift) for shift in range(1, 8)]
        orders += [torch.randperm(8, generator=rng) for _ in range(24)]
        for order in orders:
            with self.subTest(order=order.tolist()):
                result = model.query_memory(
                    prepared(keys[order], codes[order], values=values[order]), self.query()
                )
                self.assertEqual(order[result.selected[0]].tolist(), [0, 4])
                self.assertTrue(torch.equal(result.logits, baseline.logits))
                self.assertEqual(result.inspected.tolist(), [[4, 4]])
                self.assertEqual(result.ties.tolist(), [[0, 0]])
        self.assertTrue(all(torch.equal(bridge, SECOND) for bridge in bridges))

    def test_collapsed_raw_probes_are_deduplicated_before_index_lookup(self):
        model, _ = controlled_model()
        memory = prepared(
            [[1, 0, 0, 0], [0.8, -0.6, 0, 0],
             [0, 0, 1, 0], [0, 0, 0.8, -0.6]],
            [0, 0, 1, 1], divisor=2,
        )
        keys = CountedKeys(memory.keys)
        memory.keys = keys
        result = model.query_memory(memory, self.query())
        self.assertEqual(result.selected.tolist(), [[0, 2]])
        self.assertEqual(result.read_codes.tolist(), [[[0, -1], [1, -1]]])
        self.assertEqual(result.probes.tolist(), [[1, 1]])
        self.assertEqual(result.inspected.tolist(), [[2, 2]])
        self.assertEqual(memory.indexes[0].stats.lookups, 2)
        self.assertEqual([len(score[1]) for score in keys.scores], [2, 2])

    def test_overlapping_candidate_ids_are_scored_once(self):
        class AliasedIndex:
            """Adversarial adapter: distinct addresses alias occurrence IDs."""
            def __init__(self):
                self.calls = []

            def lookup(self, code):
                self.calls.append(code)
                positions = {0: (0, 1), 1: (1, 2), 2: (3, 4), 3: (4, 5)}[code]
                return LookupResult(code, positions, 2, False, 2)

        model, _ = controlled_model()
        memory = prepared(
            [[1, 0, 0, 0], [0.8, 0.6, 0, 0], [0.6, 0.8, 0, 0],
             [0, 0, 1, 0], [0, 0, 0.8, 0.6], [0, 0, 0.6, 0.8]],
            [0, 0, 1, 2, 2, 3],
        )
        index = AliasedIndex()
        memory.indexes = [index]
        keys = CountedKeys(memory.keys)
        memory.keys = keys
        result = model.query_memory(memory, self.query())
        self.assertEqual(result.selected.tolist(), [[0, 3]])
        self.assertEqual(index.calls, [0, 1, 2, 3])
        self.assertEqual(result.returned.tolist(), [[3, 3]])
        self.assertEqual(result.inspected.tolist(), [[3, 3]])
        self.assertEqual([entry[1] for entry in keys.scores], [(0, 1, 2), (3, 4, 5)])

    def test_actual_dot_products_respect_four_candidate_cap_with_large_history(self):
        keys, codes, values = collision_fixture()
        # The extra occurrences are retained but their address is never probed.
        keys = torch.cat((keys, torch.zeros(10_000, 4)))
        codes = torch.cat((codes, torch.full((10_000,), 99, dtype=torch.long)))
        values = torch.cat((values, torch.ones(10_000, 4)))
        memory = prepared(keys, codes, values=values)
        counted = CountedKeys(memory.keys, cap=4, allowed=set(range(8)))
        memory.keys = counted
        model, _ = controlled_model(second_key=None)
        result = model.query_memory(memory, self.query())
        self.assertEqual(result.selected.tolist(), [[0, 4]])
        self.assertEqual(result.inspected.tolist(), [[4, 4]])
        self.assertEqual([len(entry[1]) for entry in counted.scores], [4, 4])
        self.assertEqual(result.probes.tolist(), [[2, 2]])
        self.assertEqual(memory.indexes[0].stats.lookups, 4)
        self.assertEqual(memory.indexes[0].stats.candidate_reads, 8)
        self.assertEqual(set(chain.from_iterable(entry[1] for entry in counted.scores)), set(range(8)))

    def test_probe_override_reduces_actual_reads(self):
        keys, codes, values = collision_fixture()
        memory = prepared(keys, codes, values=values)
        counted = CountedKeys(memory.keys, cap=2)
        memory.keys = counted
        model, _ = controlled_model(second_key=None)
        result = model.query_memory(memory, self.query(), probe_count=1)
        self.assertEqual(result.selected.tolist(), [[0, 4]])
        self.assertEqual(result.read_codes.tolist(), [[[0], [2]]])
        self.assertEqual(result.inspected.tolist(), [[2, 2]])
        self.assertEqual([len(entry[1]) for entry in counted.scores], [2, 2])
        self.assertEqual(memory.indexes[0].stats.lookups, 2)

    def test_tied_first_scores_abort_without_order_tiebreak_or_second_query(self):
        model, bridges = controlled_model()
        memory = prepared([FIRST.tolist(), FIRST.tolist()], [0, 0])
        result = model.query_memory(memory, self.query())
        self.assert_null_second(result)
        self.assertEqual(result.ties.tolist(), [[1, 0]])
        self.assertEqual(result.inspected.tolist(), [[2, 0]])
        self.assertEqual(bridges, [])

    def test_tied_second_scores_return_zero_final_value(self):
        model, bridges = controlled_model(second_key=None)
        memory = prepared(
            [FIRST.tolist(), SECOND.tolist(), SECOND.tolist()],
            [0, 2, 2], values=[SECOND.tolist(), [1, 2, 3, 4], [4, 3, 2, 1]],
        )
        result = model.query_memory(memory, self.query())
        self.assertEqual(result.selected.tolist(), [[0, -1]])
        self.assertEqual(result.ties.tolist(), [[0, 1]])
        self.assertEqual(result.inspected.tolist(), [[1, 2]])
        self.assertEqual(len(bridges), 1)
        self.assertTrue(torch.equal(bridges[0], SECOND))
        self.assertTrue(torch.equal(result.logits, torch.zeros(1, 4)))

    def test_empty_first_read_aborts_without_global_fallback(self):
        model, bridges = controlled_model()
        # A strong second-hop candidate exists, but the bridge was not found.
        memory = prepared([SECOND.tolist()], [2])
        counted = CountedKeys(memory.keys)
        memory.keys = counted
        result = model.query_memory(memory, self.query())
        self.assert_null_second(result)
        self.assertEqual(result.probes.tolist(), [[2, 0]])
        self.assertEqual(counted.gathers, [])
        self.assertEqual(bridges, [])

    def test_overflow_first_read_aborts_and_counts_omitted_occurrences(self):
        model, bridges = controlled_model()
        memory = prepared(
            [FIRST.tolist(), FIRST.tolist(), FIRST.tolist(), SECOND.tolist()],
            [0, 0, 0, 2],
        )
        counted = CountedKeys(memory.keys)
        memory.keys = counted
        result = model.query_memory(memory, self.query())
        self.assert_null_second(result)
        self.assertEqual(result.overflow.tolist(), [[1, 0]])
        self.assertEqual(result.bucket_size.tolist(), [[3, 0]])
        self.assertEqual(result.returned.tolist(), [[0, 0]])
        self.assertEqual(memory.indexes[0].stats.dropped_candidates, 3)
        self.assertEqual(memory.indexes[0].audit_bucket(0), (0, 1, 2))
        self.assertEqual(counted.scores, [])
        self.assertEqual(bridges, [])

    def test_one_overflow_does_not_discard_valid_candidate_in_other_bucket(self):
        model, bridges = controlled_model(second_key=None)
        memory = prepared(
            [FIRST.tolist(), FIRST.tolist(), FIRST.tolist(),
             [0.8, 0.6, 0, 0], SECOND.tolist(), [0, 0, 0.8, 0.6]],
            [0, 0, 0, 1, 2, 3],
            values=[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3],
                    SECOND.tolist(), [4, 3, 2, 1], [1, 2, 3, 4]],
        )
        counted = CountedKeys(memory.keys, allowed={3, 4, 5})
        memory.keys = counted
        result = model.query_memory(memory, self.query())
        self.assertEqual(result.selected.tolist(), [[3, 4]])
        self.assertEqual(result.overflow.tolist(), [[1, 0]])
        self.assertEqual(result.bucket_size.tolist(), [[4, 2]])
        self.assertEqual(result.inspected.tolist(), [[1, 2]])
        self.assertEqual(result.probes.tolist(), [[2, 2]])
        self.assertEqual(len(bridges), 1)
        self.assertTrue(torch.equal(result.logits[0], torch.tensor([4.0, 3.0, 2.0, 1.0])))

    def test_missing_second_read_cannot_reuse_first_value_as_answer(self):
        model, bridges = controlled_model(second_key=None)
        memory = prepared([FIRST.tolist()], [0], values=[SECOND.tolist()])
        result = model.query_memory(memory, self.query())
        self.assertEqual(result.selected.tolist(), [[0, -1]])
        self.assertEqual(result.inspected.tolist(), [[1, 0]])
        self.assertEqual(result.probes.tolist(), [[2, 2]])
        self.assertEqual(len(bridges), 1)
        self.assertTrue(torch.equal(result.logits, torch.zeros(1, 4)))

    def test_null_chain_in_one_batch_row_does_not_abort_another_row(self):
        keys, codes, values = collision_fixture()
        model, bridges = controlled_model(second_key=None)
        memory = prepared(
            torch.stack((keys, keys)),
            torch.stack((torch.full_like(codes, 99), codes)),
            values=torch.stack((values, values)),
        )
        result = model.query_memory(memory, self.query(batch=2))
        self.assertEqual(result.selected.tolist(), [[-1, -1], [0, 4]])
        self.assertEqual(result.inspected.tolist(), [[0, 0], [4, 4]])
        self.assertEqual(len(bridges), 1)
        self.assertTrue(torch.equal(result.logits[0], torch.zeros(4)))
        self.assertTrue(torch.equal(result.logits[1], values[4]))

    def test_ordinary_forward_constructs_index_from_vectors_without_support_labels(self):
        keys, codes, values = collision_fixture()
        model, bridges = controlled_model(second_key=None)

        def write(router, records):
            self.assertEqual(tuple(records.shape), (1, 8, 12))
            return values.unsqueeze(0), keys.unsqueeze(0)

        model.router.write = types.MethodType(write, model.router)
        output = model(torch.zeros(1, 8, 12, dtype=torch.long), self.query())
        self.assertEqual(output.writer_codes[0].tolist(), codes.tolist())
        self.assertEqual(output.selected.tolist(), [[0, 4]])
        self.assertEqual(output.inspected.tolist(), [[4, 4]])
        self.assertTrue(torch.equal(output.logits[0], values[4]))
        self.assertEqual(len(bridges), 1)
        with self.assertRaises(TypeError):
            model.query_memory(prepared(keys, codes), self.query(), support=torch.tensor([[0, 4]]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
