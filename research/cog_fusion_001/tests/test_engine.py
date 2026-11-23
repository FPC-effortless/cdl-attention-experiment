"""Behavioral engine regressions: exact continuation and bounded real retrieval."""
from pathlib import Path
import os

import pytest
import torch

from fusion.engine import Resolution, bucket, greedy_generate, hybrid_read
from fusion.model import FusionModel

torch.set_num_threads(1)
CHARS = tuple("\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
PREFIX = "\nAnswer: "


def state_map(value, function):
    if isinstance(value, torch.Tensor):
        return function(value)
    if isinstance(value, dict):
        return {key: state_map(item, function) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(state_map(item, function) for item in value)
    return [state_map(item, function) for item in value]


def assert_state_equal(a, b, *, exact=True):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0 if exact else 3e-5, atol=0 if exact else 3e-6)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_state_equal(a[key], b[key], exact=exact)
    else:
        assert len(a) == len(b)
        for left, right in zip(a, b):
            assert_state_equal(left, right, exact=exact)


class RecordingDecoder:
    """Fixture records consumed bytes and deliberately varies answer lengths."""
    chars = CHARS
    answers = {0: "A.", 1: "Bela.", 2: "QRSTUVWXYZab"}

    def initial(self, identities):
        return {"identity": torch.tensor(identities)[:, None],
                "count": torch.zeros(len(identities), 1, dtype=torch.long),
                "history": torch.full((len(identities), 64), -1, dtype=torch.long),
                "nested": ({"accumulator": torch.zeros(len(identities), 2)},)}

    def decode(self, tokens, state, values, masks, *, memory_enabled=None):
        state = state_map(state, lambda value: value.clone())
        logits = torch.full((*tokens.shape, len(self.chars)), -100.)
        for row in range(tokens.size(0)):
            for time in range(tokens.size(1)):
                count = int(state["count"][row, 0])
                state["history"][row, count] = tokens[row, time]
                state["count"][row, 0] += 1
                state["nested"][0]["accumulator"][row] += torch.tensor([1, int(tokens[row, time])])
                index = int(state["count"][row, 0]) - len(PREFIX)
                text = self.answers[int(state["identity"][row, 0])]
                # Extra X bytes expose the old silent-finished-row continuation bug.
                next_char = text[index] if 0 <= index < len(text) else "X"
                logits[row, time, self.chars.index(next_char)] = 100.
        return logits, state


def test_finished_rows_consume_no_invisible_tokens_and_match_single_requests():
    assert len(CHARS) == 65
    model = RecordingDecoder()
    resolutions = [Resolution([index], [], "ok") for index in range(3)]
    original = model.initial([0, 1, 2])
    originals_copy = state_map(original, lambda value: value.clone())
    outputs, final = greedy_generate(model, original, resolutions)
    assert outputs == ["A.", "Bela.", "QRSTUVWXYZab"]
    for row, output in enumerate(outputs):
        count = int(final["count"][row, 0])
        assert count == len(PREFIX + output)
        consumed = "".join(CHARS[index] for index in final["history"][row, :count].tolist())
        assert consumed == PREFIX + output
        assert bool((final["history"][row, count:] == -1).all())
        single_output, single_state = greedy_generate(model, model.initial([row]), [resolutions[row]])
        assert single_output == [output]
        assert_state_equal(state_map(final, lambda value: value[row:row+1]), single_state)
    assert_state_equal(original, originals_copy)


def test_fixed_vocabulary_and_twelve_character_cap():
    model = RecordingDecoder()
    # No legal-answer choices are supplied: all 65 output coordinates compete.
    outputs, state = greedy_generate(model, model.initial([2]), [Resolution([16], [], "missing")])
    assert outputs == ["QRSTUVWXYZab"]
    assert state["count"].item() == len(PREFIX) + 12
    with pytest.raises(ValueError, match="twelve"):
        greedy_generate(model, model.initial([0]), [Resolution([0], [], "ok")], max_chars=13)


def test_real_native_model_batched_generation_matches_single_continuations():
    path = Path(os.environ.get("COG_NATIVE_CHECKPOINT",
        "/workspace/scratch/2a5ba884844e/deliverables/COG_CONNECTED_1M.pt"))
    if not path.exists():
        pytest.skip("selected historical native checkpoint unavailable")
    model = FusionModel.from_native(path, 17).eval()
    query, lengths = model.encode_texts(["Who owns the chest?", "Who owns the container holding the coin?"])
    with torch.inference_mode():
        _, _, state = model.encode_event(query, lengths)
        resolutions = [Resolution([0], [(8, 0)], "ok"), Resolution([8, 1], [(12, 1), (8, 0)], "ok")]
        outputs, final = greedy_generate(model, state, resolutions)
        for row in range(2):
            single_output, single_state = greedy_generate(model,
                state_map(state, lambda value: value[row:row+1]), [resolutions[row]])
            assert single_output == [outputs[row]]
            assert_state_equal(state_map(final, lambda value: value[row:row+1]), single_state, exact=False)


class RoutingSpy:
    """Only candidate identities and actual previous results enter this fixture."""
    def __init__(self, tied=False):
        self.calls, self.scored = [], []
        self.tied = tied

    def route_logits(self, feature, previous_value=None, hop=0):
        previous = None if previous_value is None else int(previous_value[0])
        self.calls.append((hop, previous))
        target = (0, 1) if hop == 0 else (previous, 0)
        address = bucket(*target)
        logits = torch.full((1, 16), -100.)
        logits[0, address] = 10
        logits[0, (address + 1) % 16] = 5
        return logits, torch.tensor([target])

    def candidate_scores(self, vector, candidate_entities, candidate_relations):
        candidates = list(zip(candidate_entities[0].tolist(), candidate_relations[0].tolist()))
        self.scored.append(candidates)
        assert len(candidates) <= 4
        if self.tied:
            return torch.zeros(1, len(candidates))
        target = tuple(vector[0].tolist())
        return torch.tensor([[10. if candidate == target else -float(index + 1)
                              for index, candidate in enumerate(candidates)]])


def test_hybrid_fetches_at_most_four_real_candidates_per_hop_and_follows_actual_first_value():
    ledger = {(entity, relation): 16 for entity in range(16) for relation in range(2)}
    ledger[(0, 1)] = 8
    ledger[(8, 0)] = 2
    ledger[(9, 0)] = 3
    model = RoutingSpy()
    original = hybrid_read(model, torch.zeros(1, 64), ledger, plan=2)
    assert original.keys == [(0, 1), (8, 0)]
    assert original.values == [8, 2]
    assert original.status == "ok"
    assert model.calls == [(0, None), (1, 8)]
    assert [len(candidates) for candidates in model.scored] == [4, 4]
    assert original.candidates_fetched == 8
    assert original.codebook_comparisons == 32
    assert original.index_records == 32
    assert all(key in ledger for candidates in model.scored for key in candidates)
    # The same feature and router must traverse a different second key when
    # the actual first record changes. No support IDs or answers are provided.
    ledger[(0, 1)] = 9
    changed_model = RoutingSpy()
    changed = hybrid_read(changed_model, torch.zeros(1, 64), ledger, plan=2)
    assert changed_model.calls == [(0, None), (1, 9)]
    assert changed.keys == [(0, 1), (9, 0)]
    assert changed.values == [9, 3]


def test_hybrid_ties_and_empty_candidates_are_explicit():
    ledger = {(0, 1): 8, (7, 0): 2}
    tied = hybrid_read(RoutingSpy(tied=True), torch.zeros(1, 64), ledger, plan=2)
    assert tied.status == "tie" and tied.values == [16]
    assert tied.candidates_fetched == 2
    missing = hybrid_read(RoutingSpy(), torch.zeros(1, 64), {}, plan=2)
    assert missing.status == "missing" and missing.values == [16]
    assert missing.candidates_fetched == 0
