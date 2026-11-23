"""Independent state-boundary checks for the actual 002 trainer and 001 runtime.

The small random model and forced-termination wrapper are engineering fixtures,
not measurements of learned semantic or language capability.
"""
import copy
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import PARENT
from train import batch_loss
import train as trainer
from fusion.engine import Resolution, evidence_tensors, greedy_generate
from fusion.model import FusionModel, FusionConfig
from fusion.native.model import ConnectedCOGTinyLM, ConnectedConfig


CHARS = tuple("\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")


def leaves(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        return [leaf for child in value.values() for leaf in leaves(child)]
    return [leaf for child in value for leaf in leaves(child)]


def same(a, b, tolerance=0):
    assert len(leaves(a)) == len(leaves(b))
    for x, y in zip(leaves(a), leaves(b)):
        assert torch.allclose(x, y, rtol=0, atol=tolerance)


def row(value, index):
    if isinstance(value, torch.Tensor):
        return value[index:index + 1]
    if isinstance(value, dict):
        return {key: row(child, index) for key, child in value.items()}
    return type(value)(row(child, index) for child in value)


@pytest.fixture
def model():
    torch.set_num_threads(1)
    torch.manual_seed(72019)
    native = ConnectedCOGTinyLM(ConnectedConfig(vocab_size=len(CHARS), d_model=24,
        n_layers=2, memory_dim=8, controller_dim=8, dropout=0))
    return FusionModel(native, CHARS, config=FusionConfig(
        semantic_dim=16, controller_dim=12, evidence_dim=8, key_dim=8)).train()


def families():
    worlds = []
    for owner, new, chest, item, oi, ni, ci, ii in (
            ("Amara", "Bela", "chest", "coin", 0, 1, 8, 12),
            ("Cora", "Davi", "crate", "gem", 2, 3, 9, 13)):
        def ask(value, identifier):
            return {"type": "ask", "text": f"Who owns the container holding the {item}?",
                    "query_entity": ii, "query_plan": 2,
                    "answer_text": value + ".", "evidence_values": [ci, identifier]}
        worlds.append({"steps": [
            {"type": "observe", "text": f"{owner} owns the {chest}.", "operation": 1,
             "entity": ci, "relation": 0, "value": oi},
            {"type": "observe", "text": f"The {item} is in the {chest}.", "operation": 1,
             "entity": ii, "relation": 1, "value": ci},
            ask(owner, oi),
            {"type": "observe", "text": f"The {chest} belongs to {new}.", "operation": 1,
             "entity": ci, "relation": 0, "value": ni},
            ask(new, ni)]})
    return [{"worlds": worlds}]


def test_teacher_loss_branch_cannot_change_carried_generated_state(model):
    original = families()
    changed_labels = copy.deepcopy(original)
    for world in changed_labels[0]["worlds"]:
        for step in world["steps"]:
            if step["type"] == "ask":
                step["answer_text"] = "unknown."
    loss_a, metrics_a, states_a = batch_loss(model, original, arm="continuous", collect_states=True)
    loss_b, metrics_b, states_b = batch_loss(model, changed_labels, arm="continuous", collect_states=True)
    assert float(loss_a.detach()) != float(loss_b.detach())
    assert metrics_a["teacher_decode_chars"] != metrics_b["teacher_decode_chars"]
    for a, b in zip(states_a, states_b):
        assert a["generated"] == b["generated"]
        same(a["carried_state"], b["carried_state"])


def test_later_losses_reach_actual_generated_state_only_in_continuous_arm(model):
    loss, _, states = batch_loss(model, families(), arm="continuous", collect_states=True)
    first_hidden = states[0]["carried_state"]["native"]["layers"][0]["hidden"]
    gradient, = torch.autograd.grad(loss, first_hidden, allow_unused=True)
    assert gradient is not None
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    reset_loss, _, reset_states = batch_loss(model, families(), arm="reset_training", collect_states=True)
    reset_hidden = reset_states[0]["carried_state"]["native"]["layers"][0]["hidden"]
    reset_gradient, = torch.autograd.grad(reset_loss, reset_hidden, allow_unused=True)
    assert reset_gradient is None or not torch.count_nonzero(reset_gradient)


def test_reset_control_changes_only_declared_answer_boundaries(model, monkeypatch):
    observed_inputs = []
    original_encode = model.encode_event
    def track(tokens, lengths=None, state=None):
        observed_inputs.append(state)
        return original_encode(tokens, lengths, state)
    monkeypatch.setattr(model, "encode_event", track)
    _, metrics, generated = batch_loss(model, families(), arm="reset_training", collect_states=True)
    # Steps: observation, observation, answer, observation, answer.
    assert torch.count_nonzero(observed_inputs[1]["native"]["layers"][0]["hidden"])
    same(observed_inputs[3], model.init_state(2))
    assert metrics["answer_boundaries_reset"] == 4
    observed_inputs.clear()
    _, metrics, generated = batch_loss(model, families(), arm="continuous", collect_states=True)
    same(observed_inputs[3], generated[0]["carried_state"])
    assert metrics["answer_boundaries_reset"] == 0


def test_variable_answer_lengths_consume_period_and_never_hidden_padding(model):
    class TerminationFixture:
        """Steer token selection while retaining the actual model state updates."""
        def __init__(self, native):
            self.native, self.chars = native, native.chars
            self.outputs = {0: "A.", 1: "Bela."}
            self.positions = {}
            self.consumed = {0: "", 1: ""}
        def decode(self, tokens, state, values, masks, **kwargs):
            logits, next_state = self.native.decode(tokens, state, values, masks, **kwargs)
            steering = torch.zeros_like(logits)
            for index, tag in enumerate(values[:, 0].tolist()):
                text = "".join(self.chars[token] for token in tokens[index].tolist())
                self.consumed[tag] += text
                if tokens.size(1) > 1:
                    self.positions[tag] = 0
                else:
                    self.positions[tag] += 1
                wanted = self.outputs[tag][min(self.positions[tag], len(self.outputs[tag]) - 1)]
                steering[index, -1, self.native.stoi[wanted]] = 1000
            return logits + steering, next_state
    wrapper = TerminationFixture(model)
    resolutions = [Resolution([0], [], "ok"), Resolution([1], [], "ok")]
    output, actual = greedy_generate(wrapper, model.init_state(2), resolutions)
    assert output == ["A.", "Bela."]
    for index, text in enumerate(output):
        assert wrapper.consumed[index] == "\nAnswer: " + text
        values, masks = evidence_tensors([resolutions[index]])
        tokens, lengths = model.encode_texts(["\nAnswer: " + text])
        _, expected = model.decode(tokens, model.init_state(1), values, masks, lengths=lengths)
        same(row(actual, index), expected, tolerance=2e-6)
        incomplete, incomplete_lengths = model.encode_texts(["\nAnswer: " + text[:-1]])
        _, before_period = model.decode(incomplete, model.init_state(1), values, masks,
                                       lengths=incomplete_lengths)
        assert any(not torch.allclose(a, b, rtol=0, atol=1e-6)
                   for a, b in zip(leaves(row(actual, index)), leaves(before_period)))


def test_batch_counter_includes_actual_rollouts_and_separate_teacher_branch(model):
    sample = families()
    _, metrics, states = batch_loss(model, sample, arm="continuous", collect_states=True)
    actual_output = sum(len(text) for state in states for text in state["generated"])
    assert metrics["generated_chars"] == actual_output
    assert metrics["generation_prefix_chars"] == 4 * len("\nAnswer: ")
    expected_inputs = sum(len(step["text"]) + 1 for world in sample[0]["worlds"] for step in world["steps"])
    assert metrics["input_chars"] == expected_inputs
    assert metrics["physical_token_positions"] == (
        metrics["input_physical_positions"] + metrics["teacher_decode_physical_positions"]
        + metrics["generation_prefix_chars"] + metrics["generated_chars"])


def test_reset_training_preserves_ledger_across_declared_boundaries(model, monkeypatch):
    observed = []
    original_read = trainer.strict_read
    def inspect_ledger(ledger, entity, plan):
        observed.append(dict(ledger))
        return original_read(ledger, entity, plan)
    monkeypatch.setattr(trainer, "strict_read", inspect_ledger)
    # Supplied warmup operations isolate this executor invariant; this fixture
    # is not evaluated or described as learned interpretation performance.
    batch_loss(model, families(), arm="reset_training", teacher=True)
    assert len(observed) == 4
    assert observed[0] == {(8, 0): 0, (12, 1): 8}
    assert observed[1] == {(9, 0): 2, (13, 1): 9}
    assert observed[2] == {(8, 0): 1, (12, 1): 8}
    assert observed[3] == {(9, 0): 3, (13, 1): 9}
