"""Independent runtime review using the full-size discarded calibration model.

These checks exercise actual neural state and sampled output. They make no
semantic-quality claim and never read held evaluation families. The calibration
checkpoint is deliberately kept separate from the six scored final fits.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from fusion.model import FusionModel
from fusion.session import PREFIX, Session
from fusion.store import ConflictError


ROOT = Path(__file__).resolve().parents[1]


def tensor_leaves(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from tensor_leaves(value[key])
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tensor_leaves(item)
    else:
        raise TypeError(type(value))


def states_equal(left, right):
    a, b = list(tensor_leaves(left)), list(tensor_leaves(right))
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert torch.equal(x, y)


@pytest.fixture(scope="module")
def checkpoint():
    value = Path(os.environ.get("COG_FUSION_REVIEW_CHECKPOINT", ROOT / "evidence/calibration/final.pt"))
    if not value.is_file():
        pytest.skip("full-size discarded calibration checkpoint not included in this checkout")
    payload = torch.load(value, map_location="cpu", weights_only=True)
    assert payload["metadata"]["step"] == 5
    return value


@pytest.fixture
def model(checkpoint):
    torch.set_num_threads(1)
    model = FusionModel.from_checkpoint(checkpoint).eval()
    assert model.parameter_summary()["total"] == 1_077_834
    return model


def test_actual_embedding_calls_account_for_every_new_character_once(model, tmp_path):
    consumed = []
    def observe_embedding(_module, args):
        tokens = args[0]
        assert tokens.shape[0] == 1
        consumed.extend(tokens[0].tolist())
    hook = model.native.embedding.register_forward_pre_hook(observe_embedding)
    path = tmp_path / "accounting.sqlite"
    try:
        with Session(model, path, seed=417) as session:
            observations = ["Amara owns the chest.", "The coin is in the chest."]
            for index, text in enumerate(observations):
                session.observe(text, f"observation-{index}")
            question = "Who owns the container holding the coin?"
            result = session.ask(question, "question-1", temperature=0.73)
            assert result["generation_mode"] == "neural"
            expected = "".join(text + "\n" for text in observations) + question + "\n" + PREFIX + result["text"]
            assert "".join(model.chars[index] for index in consumed) == expected
            count = len(consumed)
            saved_counters = dict(session.counters)
            replay = session.ask(question, "question-1", temperature=0.73)
            assert replay == dict(result, replayed=True)
            assert len(consumed) == count
            assert session.counters == saved_counters
        with Session(model, path, seed=99999) as resumed:
            assert len(consumed) == count  # restoring tensors never embeds old input
            assert resumed.counters == saved_counters
            next_question = "Who owns the chest?"
            answer = resumed.ask(next_question, "question-2", temperature=0.73)
            expected_suffix = next_question + "\n" + PREFIX + answer["text"]
            assert "".join(model.chars[index] for index in consumed[count:]) == expected_suffix
            assert len(consumed) == sum(resumed.counters[key] for key in (
                "input_tokens", "control_tokens", "emitted_tokens"))
            assert resumed.counters["historical_tokens_replayed"] == 0
    finally:
        hook.remove()


@pytest.mark.parametrize("stage", ["before_commit", "after_commit"])
def test_full_size_sampled_process_crash_matches_uninterrupted_continuation(model, checkpoint, tmp_path, stage):
    reference_path = tmp_path / "uninterrupted.sqlite"
    recovered_path = tmp_path / "recovered.sqlite"
    with Session(model, reference_path, seed=907) as reference:
        with Session(model, recovered_path, seed=907) as candidate:
            for session in (reference, candidate):
                session.observe("Amara owns the chest.", "event-1")
                session.ask("Who owns the chest?", "question-1", temperature=0.69)
            states_equal(reference.state, candidate.state)
        expected = reference.ask("Who is the owner of the chest?", "question-2", temperature=0.81)
        script = """
import os, sys, torch
from fusion.session import Session
torch.set_num_threads(1)
def crash(stage):
    if stage == sys.argv[3]:
        os._exit(71)
with Session.from_checkpoint(sys.argv[1], sys.argv[2], seed=5) as session:
    session.ask('Who is the owner of the chest?', 'question-2', temperature=0.81, failpoint=crash)
"""
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        proc = subprocess.run([sys.executable, "-c", script, str(checkpoint), str(recovered_path), stage],
                              env=env, capture_output=True, text=True, timeout=60)
        assert proc.returncode == 71, proc.stderr
        with Session(model, recovered_path, seed=1) as resumed:
            actual = resumed.ask("Who is the owner of the chest?", "question-2", temperature=0.81)
            assert actual["text"] == expected["text"]
            assert actual["replayed"] == (stage == "after_commit")
            assert actual["query_entity"] == expected["query_entity"]
            assert actual["query_plan"] == expected["query_plan"]
            assert actual["selected_values"] == expected["selected_values"]
            states_equal(resumed.state, reference.state)
            assert torch.equal(resumed.generator.get_state(), reference.generator.get_state())
            assert resumed.counters == reference.counters
            assert resumed.store.db.execute(
                "SELECT COUNT(*) FROM receipts WHERE request_id='question-2'").fetchone()[0] == 1
            # A later answer checks continuation, not merely replayed text.
            continuation = resumed.ask("Who owns the chest?", "question-3", temperature=0.67)
            expected_continuation = reference.ask("Who owns the chest?", "question-3", temperature=0.67)
            assert continuation["text"] == expected_continuation["text"]
            states_equal(resumed.state, reference.state)
            assert torch.equal(resumed.generator.get_state(), reference.generator.get_state())


def test_pending_policy_does_not_emit_or_sample_and_survives_restart(model, tmp_path):
    path = tmp_path / "pending.sqlite"
    with Session(model, path, seed=29) as session:
        session.observe("Amara owns the chest.", "observation")
        session.store.mark_pending("unresolved", payload="scope not yet interpreted")
        random_before = session.generator.get_state().clone()
        answer = session.ask("Who owns the chest?", "blocked-1", temperature=0.8)
        assert answer["text"] == "pending" and answer["generation_mode"] == "runtime_policy"
        assert session.counters["emitted_tokens"] == 0
        assert session.counters["control_tokens"] == 0
        assert torch.equal(session.generator.get_state(), random_before)
    with Session(model, path, seed=100) as resumed:
        answer = resumed.ask("Who owns the chest?", "blocked-2", temperature=0.8)
        assert answer["generation_mode"] == "runtime_policy"
        assert torch.equal(resumed.generator.get_state(), random_before)
        counters = dict(resumed.counters)
        with pytest.raises(ConflictError):
            resumed.ask("Who owns the crate?", "blocked-2", temperature=0.8)
        assert resumed.counters == counters

