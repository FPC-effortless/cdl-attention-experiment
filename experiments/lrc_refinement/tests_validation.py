"""Audit checks using development fixtures only; no checkpoints are loaded."""
from __future__ import annotations

import unittest

import torch

from lrc_vb001_comparison.source import study as vb
from .validation import input_signatures, split_overlap, validate_location_quartets


def copy_data(data):
    return {key: value.clone() for key, value in data.items()}


class LocationValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # These are the explicitly designated DEVELOPMENT seeds. Confirmation
        # generator seeds must not be added to this test module.
        cls.iid = vb.dataset(94901, 8, False)
        cls.withheld = vb.dataset(94902, 8, True)
        cls.other = vb.dataset(94903, 8, False)

    def test_valid_development_panels_and_no_mutation(self):
        for data, heldout in ((self.iid, False), (self.withheld, True), (self.other, False)):
            with self.subTest(heldout=heldout):
                before = vb.digest_data(data)
                receipt = validate_location_quartets(data, heldout)
                self.assertEqual(receipt["status"], "passed")
                self.assertEqual(receipt["examples"], 32)
                self.assertEqual(receipt["quartets"], 8)
                self.assertEqual(receipt["changed_tokens_vs_quartet_reference"], 24)
                self.assertEqual(receipt["data_sha256"], before)
                self.assertEqual(vb.digest_data(data), before)

    def test_schema_dtype_and_nonquartet_shapes_are_rejected(self):
        bad_cases = {}
        missing = copy_data(self.iid)
        del missing["person"]
        bad_cases["missing_key"] = missing
        extra = copy_data(self.iid)
        extra["extra"] = torch.zeros(32, dtype=torch.long)
        bad_cases["extra_key"] = extra
        dtype = copy_data(self.iid)
        dtype["records"] = dtype["records"].float()
        bad_cases["wrong_dtype"] = dtype
        scalar = copy_data(self.iid)
        scalar["answer"] = torch.tensor(0, dtype=torch.long)
        bad_cases["scalar_answer"] = scalar
        record_shape = copy_data(self.iid)
        record_shape["records"] = record_shape["records"][:, :, :-1]
        bad_cases["record_shape"] = record_shape
        bad_cases["nonquartet_count"] = vb.take(self.iid, slice(0, 31))
        bad_cases["empty"] = vb.take(self.iid, slice(0, 0))
        for name, bad in bad_cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_location_quartets(bad, False)
        with self.assertRaises(ValueError):
            validate_location_quartets(self.iid, 0)

    def test_wrong_answer_is_rejected(self):
        bad = copy_data(self.iid)
        bad["answer"][1] = 3
        with self.assertRaises(ValueError):
            validate_location_quartets(bad, False)

    def test_consistent_but_wrong_support_labels_are_rejected(self):
        for hop in (0, 1):
            bad = copy_data(self.iid)
            wrong = next(i for i in range(vb.N_RECORDS) if i not in bad["support"][0].tolist())
            bad["support"][:4, hop] = wrong
            with self.subTest(hop=hop), self.assertRaises(ValueError):
                validate_location_quartets(bad, False)

    def test_changed_or_mislabeled_query_is_rejected(self):
        for rows in (slice(1, 2), slice(0, 4)):
            bad = copy_data(self.iid)
            entity = int(bad["entity"][0])
            bad["query"][rows, 6] = vb.VOCAB[vb.NAMES[(entity + 1) % 4]]
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                validate_location_quartets(bad, False)

    def test_changed_distractor_is_rejected(self):
        bad = copy_data(self.iid)
        position = next(i for i in range(vb.N_RECORDS) if i not in bad["support"][0].tolist())
        labels = vb.NAMES if int(bad["records"][1, position, 5]) == vb.VOCAB["partnered"] else vb.PLACES
        ids = [vb.VOCAB[label] for label in labels]
        old = ids.index(int(bad["records"][1, position, 7]))
        bad["records"][1, position, 7] = ids[(old + 1) % 4]
        with self.assertRaises(ValueError):
            validate_location_quartets(bad, False)

    def test_invalid_typed_payload_is_rejected_even_if_quartet_stable(self):
        bad = copy_data(self.iid)
        first = int(bad["support"][0, 0])
        bad["records"][:4, first, 7] = vb.VOCAB[vb.PLACES[0]]
        with self.assertRaisesRegex(ValueError, "invalid relation value"):
            validate_location_quartets(bad, False)

    def test_wrong_split_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "requested split"):
            validate_location_quartets(self.iid, True)
        with self.assertRaisesRegex(ValueError, "requested split"):
            validate_location_quartets(self.withheld, False)

    def test_duplicate_logical_facts_are_rejected(self):
        bad = copy_data(self.iid)
        positions = [i for i in range(8) if i not in bad["support"][0].tolist()]
        destination, source = positions[:2]
        bad["records"][:4, destination] = bad["records"][:4, source]
        with self.assertRaisesRegex(ValueError, "duplicate logical fact"):
            validate_location_quartets(bad, False)

    def test_era_blocks_are_checked_even_with_correct_support_remapping(self):
        bad = copy_data(self.iid)
        permutation = torch.arange(vb.N_RECORDS)
        permutation[0], permutation[8] = 8, 0
        inverse = torch.argsort(permutation)
        bad["records"] = bad["records"][:, permutation]
        bad["support"] = inverse[bad["support"]]
        with self.assertRaisesRegex(ValueError, "two era blocks"):
            validate_location_quartets(bad, False)

    def test_wrong_person_metadata_is_rejected(self):
        bad = copy_data(self.iid)
        bad["person"][:4] = (bad["person"][0] + 2) % 4
        with self.assertRaises(ValueError):
            validate_location_quartets(bad, False)

    def test_signatures_ignore_labels_and_include_both_inputs(self):
        original = input_signatures(self.iid)
        self.assertEqual(len(original), 32)
        self.assertTrue(all(isinstance(value, bytes) and len(value) == 32 for value in original))
        labels_only = copy_data(self.iid)
        labels_only["answer"] = (labels_only["answer"] + 1) % 4
        labels_only["support"] = (labels_only["support"] + 1) % vb.N_RECORDS
        self.assertEqual(original, input_signatures(labels_only))
        for field, index in (("records", (0, 0, 7)), ("query", (0, 6))):
            changed = copy_data(self.iid)
            changed[field][index] += 1
            replacement = input_signatures(changed)
            with self.subTest(field=field):
                self.assertEqual(len(original & replacement), 31)
                self.assertEqual(len(original - replacement), 1)
                self.assertEqual(len(replacement - original), 1)

    def test_signature_input_shapes_and_dtypes_are_checked(self):
        bad_cases = [
            {"records": self.iid["records"]},
            {"records": self.iid["records"].float(), "query": self.iid["query"]},
            {"records": self.iid["records"], "query": self.iid["query"][:-1]},
            {"records": self.iid["records"][:, :-1], "query": self.iid["query"]},
        ]
        for index, bad in enumerate(bad_cases):
            with self.subTest(index=index), self.assertRaises(ValueError):
                input_signatures(bad)
        self.assertEqual(input_signatures(vb.take(self.iid, slice(0, 0))), set())

    def test_overlap_counts_and_within_panel_duplicates(self):
        train = vb.take(self.iid, slice(0, 8))
        fresh_a = vb.take(self.iid, slice(4, 12))
        fresh_a = {key: torch.cat((value, value[:1])) for key, value in fresh_a.items()}
        fresh_b = vb.take(self.iid, slice(10, 16))
        result = split_overlap(train, {"a": fresh_a, "b": fresh_b, "c": self.other})
        self.assertEqual(result["with_train"], {"a": 4, "b": 0, "c": 0})
        self.assertEqual(result["between_fresh"], {"a": {"b": 2, "c": 0}, "b": {"c": 0}, "c": {}})
        self.assertEqual(result["within_dataset_duplicate_inputs"], {"a": 1, "b": 0, "c": 0})
        self.assertEqual(result["dataset_unique_inputs"], {"a": 8, "b": 6, "c": 32})
        self.assertFalse(result["all_disjoint"])
        disjoint = split_overlap(train, {"different_inputs": self.other})
        self.assertTrue(disjoint["all_disjoint"])
        self.assertEqual(disjoint["with_train"], {"different_inputs": 0})


if __name__ == "__main__":
    unittest.main()
