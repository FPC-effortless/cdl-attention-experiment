"""Behavioral checks for causal continuation and the learned evidence path."""
import os
from pathlib import Path

import pytest
import torch

from fusion.model import FusionConfig, FusionModel, clone_state, detach_state, state_bytes
from fusion.native.model import ConnectedCOGTinyLM, ConnectedConfig

torch.set_num_threads(1)


@pytest.fixture
def model():
    torch.manual_seed(73)
    chars = tuple("\n .?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    native = ConnectedCOGTinyLM(ConnectedConfig(vocab_size=len(chars), d_model=24, n_layers=2,
                memory_dim=8, controller_dim=8, dropout=0))
    model = FusionModel(native, chars, config=FusionConfig(semantic_dim=16, controller_dim=12,
                                                         evidence_dim=8, key_dim=8))
    return model.eval()


def assert_states_close(a, b, atol=2e-6):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=atol)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_states_close(a[key], b[key], atol)
    else:
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_states_close(x, y, atol)


def slice_state(state, index):
    if isinstance(state, torch.Tensor):
        return state[index:index+1]
    if isinstance(state, dict):
        return {key: slice_state(value, index) for key, value in state.items()}
    return tuple(slice_state(value, index) for value in state)


def test_original_local_language_function_is_preserved(model):
    tokens, lengths = model.encode_texts(["Amara owns the chest."])
    logits, _ = model.forward_lm(tokens)
    features, _ = model.native.backbone_features(tokens)
    expected = model.native.lm_head(model.native.final_norm(features)).float()
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)


def test_padding_cannot_change_events_or_continuation(model):
    texts = ["Bela.", "The coin is in the chest."]
    tokens, lengths = model.encode_texts(texts)
    features, labels, state = model.encode_event(tokens, lengths)
    for row, text in enumerate(texts):
        single_tokens, single_lengths = model.encode_texts([text])
        single_features, single_labels, single_state = model.encode_event(single_tokens, single_lengths)
        torch.testing.assert_close(features[row:row+1], single_features, atol=2e-6, rtol=2e-5)
        for name in labels:
            torch.testing.assert_close(labels[name][row:row+1], single_labels[name], atol=2e-6, rtol=2e-5)
        assert_states_close(slice_state(state, row), single_state)
        # Change every padding token, then verify no valid-prefix difference.
    poisoned = tokens.clone()
    poisoned[0, lengths[0]:] = model.stoi["Z"]
    pfeature, _, pstate = model.encode_event(poisoned, lengths)
    torch.testing.assert_close(features, pfeature, rtol=0, atol=0)
    assert_states_close(state, pstate, atol=0)


def test_decoder_incremental_equals_batched_and_state_is_not_mutated(model):
    with torch.no_grad():
        model.evidence_output.weight.normal_(0, .05)
        model.controller_output.weight.normal_(0, .05)
    query, length = model.encode_texts(["Who owns the chest?"])
    _, _, state = model.encode_event(query, length)
    before = clone_state(state)
    answer, _ = model.encode_texts([" Amara.\n"])
    values, mask = torch.tensor([[8, 0]]), torch.tensor([[True, True]])
    full_logits, full_state = model.decode(answer, state, values, mask)
    current = state
    pieces = []
    for position in range(answer.size(1)):
        logits, current = model.decode(answer[:, position:position+1], current, values, mask)
        pieces.append(logits)
    torch.testing.assert_close(torch.cat(pieces, 1), full_logits, rtol=2e-5, atol=2e-6)
    assert_states_close(current, full_state)
    assert_states_close(state, before, atol=0)


def test_evidence_changes_original_character_logits_and_acute_removal_is_exact(model):
    with torch.no_grad():
        model.evidence_output.weight.normal_(0, .1)
        model.controller_output.weight.normal_(0, .1)
    query, length = model.encode_texts(["Who owns the chest?"])
    _, _, state = model.encode_event(query, length)
    tokens, _ = model.encode_texts([" "])
    values_a, values_b = torch.tensor([[0, 16]]), torch.tensor([[1, 16]])
    mask = torch.tensor([[True, False]])
    output_a, _ = model.decode(tokens, state, values_a, mask)
    output_b, _ = model.decode(tokens, state, values_b, mask)
    assert (output_a - output_b).abs().max() > 1e-5
    off_a, offstate_a = model.decode(tokens, state, values_a, mask, memory_enabled=False)
    off_b, offstate_b = model.decode(tokens, state, values_b, mask, memory_enabled=False)
    torch.testing.assert_close(off_a, off_b, rtol=0, atol=0)
    assert_states_close(offstate_a, offstate_b, atol=0)
    output_a.square().mean().backward()
    assert model.native.embedding.weight.grad.abs().sum() > 0
    assert model.value_embedding.weight.grad.abs().sum() > 0
    assert model.evidence_output.weight.grad.abs().sum() > 0


def test_second_hop_uses_actual_first_value_and_only_scores_supplied_candidates(model):
    tokens, lengths = model.encode_texts(["Who owns the container holding the coin?"])
    features, _, _ = model.encode_event(tokens, lengths)
    _, first = model.route_logits(features, torch.tensor([8]), hop=1)
    _, second = model.route_logits(features, torch.tensor([9]), hop=1)
    assert (first - second).abs().max() > 1e-5
    candidates = torch.tensor([[8, 9, 10, 11]])
    relations = torch.tensor([[0, 0, 1, 1]])
    scores = model.candidate_scores(first, candidates, relations)
    assert scores.shape == (1, 4)
    # Adding an unrelated candidate cannot alter scores of supplied candidates.
    shorter = model.candidate_scores(first, candidates[:, :2], relations[:, :2])
    torch.testing.assert_close(shorter, scores[:, :2], rtol=0, atol=0)


def test_state_size_does_not_grow_with_history(model):
    one, lengths = model.encode_texts(["A"])
    many, longlengths = model.encode_texts(["Amara owns the chest. " * 20])
    _, _, first = model.encode_event(one, lengths)
    _, _, second = model.encode_event(many, longlengths)
    assert state_bytes(first) == state_bytes(second) == state_bytes(model.init_state(1))
    detached = detach_state(second)
    assert not detached["event"].requires_grad


def test_padded_decode_preserves_correct_final_controller(model):
    query, qlengths = model.encode_texts(["Who owns the chest?", "Who owns the coin?"])
    _, _, state = model.encode_event(query, qlengths)
    tokens, lengths = model.encode_texts([" Bela.\n", " Unknown.\n"])
    values = torch.tensor([[1, 16], [16, 16]])
    mask = torch.tensor([[True, False], [True, False]])
    logits, nextstate = model.decode(tokens, state, values, mask, lengths=lengths)
    for row, length in enumerate(lengths.tolist()):
        single, ss = model.decode(tokens[row:row+1, :length], slice_state(state, row),
                                  values[row:row+1], mask[row:row+1])
        torch.testing.assert_close(single, logits[row:row+1, :length], atol=2e-6, rtol=2e-5)
        assert_states_close(ss, slice_state(nextstate, row))


def test_checkpoint_reconstruction_keeps_logits_and_identity(model, tmp_path):
    tokens, lengths = model.encode_texts(["The coin is in the chest."])
    _, logits, _ = model.encode_event(tokens, lengths)
    path = tmp_path / "model.pt"
    torch.save(model.checkpoint_payload(protocol="test"), path)
    reconstructed = FusionModel.from_checkpoint(path).eval()
    _, expected, _ = reconstructed.encode_event(tokens, lengths)
    for key in logits:
        torch.testing.assert_close(logits[key], expected[key], rtol=0, atol=0)
    assert model.model_identity() == reconstructed.model_identity()


def test_malformed_state_and_evidence_are_rejected(model):
    tokens, lengths = model.encode_texts(["Amara."])
    with pytest.raises(ValueError, match="lengths"):
        model.encode_event(tokens, torch.tensor([0]))
    state = model.init_state(1)
    state["event"] = torch.zeros(1, 17)
    with pytest.raises(ValueError, match="event"):
        model.encode_event(tokens, lengths, state)
    with pytest.raises(ValueError, match="vocabulary"):
        model.decode(tokens, None, torch.tensor([[18, 0]]), torch.tensor([[True, False]]))


def test_selected_native_checkpoint_identity_if_available():
    path = Path(os.environ.get("COG_NATIVE_CHECKPOINT",
        "/workspace/scratch/2a5ba884844e/deliverables/COG_CONNECTED_1M.pt"))
    if not path.exists():
        pytest.skip("selected historical native checkpoint is not available")
    model = FusionModel.from_native(path, 17).eval()
    assert model.parameter_summary() == {"total": 1077834, "native": 993574,
        "new_interfaces": 84260, "inactive_native_board": 80160, "active": 997674}
