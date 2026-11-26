"""Optional v2 path tests; not a claim of joint language/composition learning."""
import hashlib
import io
import json
from pathlib import Path

import pytest
import torch

from fusion.full_model import FullFusionModel
from fusion.composition import RecordRuntime, StaleDerivedError
from fusion.model import FusionConfig, FusionModel, state_bytes
from fusion.native.model import ConnectedCOGTinyLM, ConnectedConfig


CHARS = tuple("\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
torch.set_num_threads(1)


def assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    else:
        assert len(left) == len(right)
        for x, y in zip(left, right):
            assert_equal(x, y)


@pytest.fixture
def full():
    torch.manual_seed(6061)
    native = ConnectedCOGTinyLM(ConnectedConfig(vocab_size=len(CHARS), d_model=24, n_layers=2,
                                                memory_dim=8, controller_dim=8, dropout=0))
    base = FusionModel(native, CHARS, config=FusionConfig(semantic_dim=16, controller_dim=12,
                                                        evidence_dim=8, key_dim=8))
    return FullFusionModel(base, seed=1701).eval()


def test_zero_controller_exactly_preserves_all_v1_token_interfaces(full):
    tokens, lengths = full.encode_texts(["Who owns the chest?", "Where is the coin?"])
    original_state = full.base.init_state(2)
    full_state = full.init_state(2)
    old_features, old_heads, old_next = full.base.encode_event(tokens, lengths, original_state)
    features, heads, next_state = full.encode_event(tokens, lengths, full_state)
    assert_equal(features, old_features)
    assert_equal(heads, old_heads)
    assert_equal(next_state["base"], old_next)
    assert not torch.count_nonzero(next_state["record_controller"])

    old_buckets, old_query = full.base.route_logits(features)
    buckets, query = full.route_logits(features, state=next_state)
    assert_equal(buckets, old_buckets)
    assert_equal(query, old_query)

    decode_tokens, decode_lengths = full.encode_texts(["\nAnswer: ", "\nAnswer: "])
    evidence = torch.tensor([[8, 0], [9, 1]])
    mask = torch.tensor([[True, True], [True, True]])
    old_logits, old_state = full.base.decode(decode_tokens, old_next, evidence, mask, lengths=decode_lengths)
    logits, state = full.decode(decode_tokens, next_state, evidence, mask, lengths=decode_lengths)
    assert_equal(logits, old_logits)
    assert_equal(state["base"], old_state)
    old_lm, old_lm_state = full.base.forward_lm(tokens, original_state, lengths=lengths)
    lm, lm_state = full.forward_lm(tokens, full_state, lengths=lengths)
    assert_equal(lm, old_lm)
    assert_equal(lm_state["base"], old_lm_state)


def test_first_selected_value_has_real_paths_to_write_query_and_character_logits(full):
    first = torch.tensor([[0.10, -0.21, 0.32, -0.43, 0.54, -0.65, 0.23, 0.04]], requires_grad=True)
    second = torch.tensor([[-0.32, 0.41, -0.12, 0.23, 0.24, -0.15, 0.06, 0.37]])
    start = full.init_state(1)
    one = full.consume_selected(first, start)
    two = full.consume_selected(second, one)
    changed_first = first.detach().clone()
    changed_first[0, 2] += 0.4
    changed = full.consume_selected(second, full.consume_selected(changed_first, start))

    numeric = full.propose_numeric_write(two)
    changed_numeric = full.propose_numeric_write(changed)
    assert (numeric - changed_numeric).abs().max() > 1e-6
    features = torch.zeros(1, full.base.config.semantic_dim)
    heads = full.semantic_write_logits(features, state=two)
    changed_heads = full.semantic_write_logits(features, state=changed)
    assert (heads["value"] - changed_heads["value"]).abs().max() > 1e-6
    buckets, query = full.route_logits(features, hop=1, previous_value=torch.tensor([8]), state=two)
    other_buckets, _ = full.route_logits(features, hop=1, previous_value=torch.tensor([8]), state=changed)
    assert (buckets - other_buckets).abs().max() > 1e-6
    token, _ = full.encode_texts([" "])
    evidence, mask = torch.tensor([[16, 16]]), torch.tensor([[False, False]])
    logits, _ = full.decode(token, two, evidence, mask)
    other_logits, _ = full.decode(token, changed, evidence, mask)
    assert (logits - other_logits).abs().max() > 1e-6

    # Each gradient ends at the first actual retrieved payload, through the
    # second read. Inspect the real writer/route/character outputs separately.
    for output in (numeric, heads["value"], buckets, query, logits):
        gradient, = torch.autograd.grad(output.square().sum(), first, retain_graph=True)
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 1e-10


def test_clearing_controller_removes_first_read_and_preserves_base_state(full):
    first = torch.full((1, 8), 0.6)
    second = torch.full((1, 8), -0.3)
    initial = full.init_state(1)
    one = full.consume_selected(first, initial)
    cleared = full.clear_controller(one)
    assert_equal(cleared["base"], initial["base"])
    assert not torch.count_nonzero(cleared["record_controller"])
    cleared_then_second = full.consume_selected(second, cleared)
    second_only = full.consume_selected(second, initial)
    assert_equal(cleared_then_second, second_only)
    assert_equal(full.propose_numeric_write(cleared_then_second), full.propose_numeric_write(second_only))
    intact = full.consume_selected(second, one)
    assert (full.propose_numeric_write(intact) - full.propose_numeric_write(second_only)).abs().max() > 1e-6


def test_numeric_training_mode_freezes_base_and_feedback(full):
    identity = full.base.model_identity()
    full.numeric_training_only()
    expected_trainable = {id(parameter) for parameter in full.computer.parameters()}
    observed_trainable = {id(parameter) for parameter in full.parameters() if parameter.requires_grad}
    assert observed_trainable == expected_trainable
    state = full.consume_selected(torch.full((2, 8), 0.3), full.init_state(2))
    state = full.consume_selected(torch.full((2, 8), -0.1), state)
    full.propose_numeric_write(state).square().sum().backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in full.computer.parameters())
    assert all(parameter.grad is None for parameter in full.base.parameters())
    for module in (full.semantic_feedback, full.query_feedback, full.character_feedback):
        assert all(parameter.grad is None for parameter in module.parameters())
    assert full.base.model_identity() == identity


def test_wrapper_state_checkpoint_and_source_checkpoint_remain_separate(full, tmp_path):
    base_path = tmp_path / "v1.pt"
    torch.save(full.base.checkpoint_payload(purpose="review fixture"), base_path)
    source_hash = hashlib.sha256(base_path.read_bytes()).hexdigest()
    model = FullFusionModel.from_v1_checkpoint(base_path, seed=1701).eval().numeric_training_only()
    state = model.init_state(1)
    event_tokens, event_lengths = model.encode_texts(["Amara owns the chest."])
    _, _, state = model.encode_event(event_tokens, event_lengths, state)
    initial_bytes = state_bytes(state)
    for index in range(32):
        state = model.consume_selected(torch.full((1, 8), (index - 16) / 40), state)
        assert state_bytes(state) == initial_bytes
    assert initial_bytes == state_bytes(model.base.init_state(1)) + 32 * 4
    checkpoint = tmp_path / "v2.pt"
    torch.save(model.checkpoint_payload(qualification="separate numeric component"), checkpoint)
    restored = FullFusionModel.from_checkpoint(checkpoint).eval()
    assert restored.model_identity() == model.model_identity()
    assert restored.parameter_summary() == model.parameter_summary()
    assert restored.admission["base_checkpoint_sha256"] == source_hash
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    recovered_state = torch.load(buffer, weights_only=True)
    token, _ = model.encode_texts([" "])
    restored.validate_state(recovered_state, token, finite=True)
    assert_equal(restored.propose_numeric_write(recovered_state), model.propose_numeric_write(state))
    evidence, mask = torch.tensor([[16, 16]]), torch.tensor([[False, False]])
    original_logits, original_state = model.decode(token, state, evidence, mask)
    restored_logits, restored_state = restored.decode(token, recovered_state, evidence, mask)
    assert_equal(restored_logits, original_logits)
    assert_equal(restored_state, original_state)
    assert hashlib.sha256(base_path.read_bytes()).hexdigest() == source_hash


def test_full_wrapper_durable_numeric_write_is_actual_output_and_tracks_both_sources(full, tmp_path):
    first = ("world", "left", "vector")
    second = ("world", "right", "vector")
    destination = ("world", "computed", "vector")
    a = torch.tensor([[0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.1]])
    b = torch.tensor([[-0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.1]])
    with torch.inference_mode():
        expected_state = full.consume_selected(b, full.consume_selected(a, full.init_state(1)))
        expected = full.propose_numeric_write(expected_state)[0].tolist()
    path = tmp_path / "full-derived.sqlite"
    with RecordRuntime(full, path) as runtime:
        runtime.put(first, a[0].tolist(), "source-a")
        runtime.put(second, b[0].tolist(), "source-b")
        receipt = runtime.compute(first, second, destination, "compute")
        assert receipt["reply"]["values"] == expected
        assert runtime.read_derived(destination) == expected
        assert_equal(runtime.state, expected_state)
    with RecordRuntime(full, path) as resumed:
        assert_equal(resumed.state, expected_state)
        assert resumed.read_derived(destination) == expected
        assert resumed.compute(first, second, destination, "compute")["replayed"]
        # Replacing either supporting record invalidates the derived value.
        resumed.put(second, (b + 0.1)[0].tolist(), "new-source-b")
        with pytest.raises(StaleDerivedError):
            resumed.read_derived(destination)
        resumed.compute(first, second, destination, "recompute")
        resumed.put(first, (a - 0.1)[0].tolist(), "new-source-a")
        with pytest.raises(StaleDerivedError):
            resumed.read_derived(destination)


def test_invalid_payload_or_missing_explicit_route_state_is_rejected(full):
    state = full.init_state(1)
    with pytest.raises(ValueError, match="shape"):
        full.consume_selected(torch.zeros(1, 9), state)
    with pytest.raises(ValueError, match="finite"):
        full.consume_selected(torch.full((1, 8), float("nan")), state)
    bad = {"base": state["base"], "record_controller": torch.zeros(1, 31)}
    with pytest.raises(ValueError, match="geometry"):
        full.consume_selected(torch.zeros(1, 8), bad)
    with pytest.raises(TypeError):
        full.route_logits(torch.zeros(1, full.base.config.semantic_dim))


def test_optional_extension_does_not_change_any_locked_v1_training_source():
    root = Path(__file__).resolve().parents[1]
    lock_path = root / "evidence/TRAINING_LOCK.json"
    if not lock_path.is_file():
        pytest.skip("historical training lock is not included in this checkout")
    lock = json.loads(lock_path.read_text())
    for relative, expected in lock["source_sha256"].items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected, relative

