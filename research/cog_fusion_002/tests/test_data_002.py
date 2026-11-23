"""Data-contract checks without running any model on held examples."""
import copy
import json

import pytest

from data import (OLD_FRESH, TRAIN_TEMPLATES, FRESH_TEMPLATES, build_corpus,
                  family, semantic_signature, text_signature, validate_family)


def test_all_eight_kinds_have_correct_interleaved_truth_and_nonidentical_pairs():
    for index in range(8):
        generated = family(index, "train")
        validate_family(generated)
        for world in generated["worlds"]:
            observed, boundaries = 0, []
            for step in world["steps"]:
                if step["type"] == "observe":
                    observed += 1
                else:
                    boundaries.append(observed)
            assert observed == 8 and boundaries == [3, 5, 8]
        assert text_signature(generated["worlds"][0]) != text_signature(generated["worlds"][1])


def test_long_neutral_observations_follow_both_answers():
    generated = family(6, "long", extra_fillers=64)
    validate_family(generated)
    count, boundaries = 0, []
    for step in generated["worlds"][0]["steps"]:
        if step["type"] == "observe":
            count += 1
        else:
            boundaries.append(count)
    assert count == 72 and boundaries == [3, 37, 72]


def test_invalid_labels_and_identical_input_pairs_are_rejected():
    wrong_label = family(0, "train")
    wrong_label["worlds"][0]["steps"][3]["answer_text"] += "X"
    with pytest.raises(ValueError, match="answer label"):
        validate_family(wrong_label)
    identical = family(1, "train")
    identical["worlds"][1] = copy.deepcopy(identical["worlds"][0])
    with pytest.raises(ValueError, match="different input"):
        validate_family(identical)


def test_split_semantics_and_wording_banks_are_disjoint(tmp_path):
    path = tmp_path / "corpus.json"
    manifest = build_corpus(path, train_families=64, held_families=8, dev_families=8, long_families=8)
    corpus = json.loads(path.read_text())
    observed = set()
    allowed_characters = set("\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    for split, families in corpus.items():
        for generated in families:
            signatures = {semantic_signature(world) for world in generated["worlds"]}
            assert not signatures & observed
            observed.update(signatures)
            for world in generated["worlds"]:
                assert all(set(step["text"]) <= allowed_characters for step in world["steps"])
    patterns = lambda bank: {pattern for values in bank.values() for pattern in values}
    assert not patterns(TRAIN_TEMPLATES) & patterns(OLD_FRESH)
    assert not patterns(TRAIN_TEMPLATES) & patterns(FRESH_TEMPLATES)
    assert not patterns(FRESH_TEMPLATES) & patterns(OLD_FRESH)
    train_only = json.loads((tmp_path / "train.json").read_text())
    assert set(train_only) == {"schema", "split", "families"}
    assert train_only["families"] == corpus["train"]
    assert manifest["identical_input_pairs"] == 0


def test_generation_is_reproducible_and_cancellation_is_an_invariant_control(tmp_path):
    first, second = tmp_path / "a/corpus.json", tmp_path / "b/corpus.json"
    build_corpus(first, train_families=16, held_families=8, dev_families=8, long_families=8)
    build_corpus(second, train_families=16, held_families=8, dev_families=8, long_families=8)
    assert first.read_bytes() == second.read_bytes()
    cancelled = family(7, "train")
    assert cancelled["worlds"][0]["answers"] == cancelled["worlds"][1]["answers"]
    assert semantic_signature(cancelled["worlds"][0]) == semantic_signature(cancelled["worlds"][1])
    assert text_signature(cancelled["worlds"][0]) != text_signature(cancelled["worlds"][1])
