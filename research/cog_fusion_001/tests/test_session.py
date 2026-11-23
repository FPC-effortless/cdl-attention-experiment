"""Executable neural/runtime integration checks, not learned-capability scores.

The deliberately tiny random fixture makes discrete head choices predictable
so faults target the runtime. It still executes the real recurrent model and
unrestricted character decoder. Scientific evaluation uses separately trained
million-parameter checkpoints and unmodified predicted interfaces.
"""
from pathlib import Path
import json
import os
import string
import subprocess
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fusion.model import FusionModel, FusionConfig
from fusion.native.model import ConnectedCOGTinyLM, ConnectedConfig
from fusion.session import Session, _serialize
from fusion.store import CapacityError, ConflictError, InjectedFailure, IntegrityError, Operation


def tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        return [tensor for v in value.values() for tensor in tensors(v)]
    return [tensor for v in value for tensor in tensors(v)]


def assert_same_state(a, b):
    assert len(tensors(a)) == len(tensors(b))
    for left, right in zip(tensors(a), tensors(b)):
        assert torch.equal(left, right)


@pytest.fixture
def model():
    torch.set_num_threads(1)
    torch.manual_seed(1207)
    chars = tuple(sorted(set(string.ascii_letters + " .?\n:'!,-;$3&")))
    native = ConnectedCOGTinyLM(ConnectedConfig(
        vocab_size=len(chars), d_model=16, n_layers=2, ffn_mult=2,
        n_entities=4, n_relations=4, memory_dim=8, controller_dim=8, dropout=0.0))
    model = FusionModel(native, chars, config=FusionConfig(
        semantic_dim=8, controller_dim=8, evidence_dim=8, key_dim=8)).eval()
    # Predict SET chest.owner=Amara, and query chest.owner. These are explicit
    # runtime fixtures, not parser successes or label-supplied inference results.
    choices = {"operation": 1, "entity": 8, "relation": 0, "value": 0,
               "query_plan": 0, "query_entity": 8}
    with torch.no_grad():
        for name, head in model.semantic_heads.items():
            head.weight.zero_(); head.bias.fill_(-1); head.bias[choices[name]] = 1
    return model


def test_exact_restart_continuation_and_new_input_accounting(tmp_path, model):
    full = Session(model, tmp_path / "full.db", seed=31)
    split_path = tmp_path / "split.db"
    split = Session(model, split_path, seed=31)
    events = ["Amara owns the chest.", "The coin is in the chest."]
    for i, event in enumerate(events):
        full.observe(event, f"obs-{i}")
        split.observe(event, f"obs-{i}")
    first = full.ask("Who owns the chest?", "q1")
    second = split.ask("Who owns the chest?", "q1")
    assert first["text"] == second["text"]
    before = dict(split.counters)
    split.close()
    split = Session(model, split_path, seed=999)  # saved RNG overrides new seed
    assert split.counters == before
    assert_same_state(full.state, split.state)
    for i in range(2):
        q = "Who is the owner of the chest?"
        a = full.ask(q, f"next-{i}")
        b = split.ask(q, f"next-{i}")
        assert a["text"] == b["text"]
    assert_same_state(full.state, split.state)
    assert full.counters == split.counters
    expected_input = sum(len(s) + 1 for s in events) + len("Who owns the chest?") + 1
    expected_input += 2 * (len("Who is the owner of the chest?") + 1)
    assert split.counters["input_tokens"] == expected_input
    assert split.counters["historical_tokens_replayed"] == 0
    assert split.counters["control_tokens"] == 3 * len("\nAnswer: ")
    full.close(); split.close()


def test_observation_uses_actual_predicted_operation(model, tmp_path):
    with Session(model, tmp_path / "obs.db") as session:
        result = session.observe("Amara owns the chest.", "observe")
        assert result["predicted_operation"] == [1, 8, 0, 0]
        assert session.store.resolve(("world", "chest", "owner")).value == "Amara"
        assert result["record_updated"]
        assert session.counters["record_updates"] == 1


def test_replay_does_not_feed_tokens_sample_or_restore_old_continuation(model, tmp_path):
    with Session(model, tmp_path / "replay.db", seed=123) as session:
        original = session.ask("Who owns the chest?", "q", temperature=0.8)
        old_counters = dict(session.counters)
        old_rng = session.generator.get_state().clone()
        replay = session.ask("Who owns the chest?", "q", temperature=0.8)
        assert replay == dict(original, replayed=True)
        assert session.counters == old_counters
        assert torch.equal(session.generator.get_state(), old_rng)
        session.observe("Amara owns the chest.", "obs")
        latest = session.store.snapshot_state()
        later_replay = session.ask("Who owns the chest?", "q", temperature=0.8)
        assert later_replay["text"] == original["text"]
        assert session.store.snapshot_state() == latest
        assert session.counters["observations"] == 1
        with pytest.raises(ConflictError):
            session.ask("Who owns the chest?", "q", temperature=1.1)


@pytest.mark.parametrize("stage", ["after_operations", "before_commit", "after_commit"])
def test_atomic_model_store_and_rng_when_publication_fails(model, tmp_path, stage):
    candidate = Session(model, tmp_path / f"{stage}.db", seed=99)
    reference = Session(model, tmp_path / f"{stage}-reference.db", seed=99)
    expected = reference.ask("Who owns the chest?", "q", temperature=0.9)
    with pytest.raises(InjectedFailure):
        candidate.ask("Who owns the chest?", "q", temperature=0.9, failpoint=stage)
    if stage == "after_commit":
        assert candidate.counters == reference.counters
    else:
        assert candidate.counters["input_tokens"] == 0
    actual = candidate.ask("Who owns the chest?", "q", temperature=0.9)
    assert actual["text"] == expected["text"]
    assert actual["replayed"] == (stage == "after_commit")
    assert_same_state(candidate.state, reference.state)
    assert torch.equal(candidate.generator.get_state(), reference.generator.get_state())
    assert candidate.counters == reference.counters
    candidate.close(); reference.close()


def test_actual_new_process_restores_sampling_and_continuation(model, tmp_path):
    checkpoint = tmp_path / "random-fixture.pt"
    torch.save(model.checkpoint_payload(purpose="runtime test only"), checkpoint)
    full = Session(model, tmp_path / "full.db", seed=73)
    resumed_path = tmp_path / "resumed.db"
    split = Session(model, resumed_path, seed=73)
    for session in (full, split):
        session.observe("Amara owns the chest.", "obs")
        session.ask("Who owns the chest?", "q1", temperature=0.7)
    split.close()
    expected = full.ask("Who is the owner of the chest?", "q2", temperature=0.7)
    script = """
import json,sys,torch
from fusion.session import Session
torch.set_num_threads(1)
with Session.from_checkpoint(sys.argv[1], sys.argv[2],seed=999) as s:
    result=s.ask('Who is the owner of the chest?','q2',temperature=0.7)
    print(json.dumps({'text':result['text'],'counters':s.counters}))
"""
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    proc = subprocess.run([sys.executable, "-c", script, str(checkpoint), str(resumed_path)],
                          env=environment, check=True, capture_output=True, text=True)
    actual = json.loads(proc.stdout)
    assert actual["text"] == expected["text"]
    assert actual["counters"] == full.counters
    with Session(model, resumed_path) as resumed:
        assert_same_state(resumed.state, full.state)
        assert torch.equal(resumed.generator.get_state(), full.generator.get_state())
    full.close()


def test_pending_abstention_is_policy_and_never_runs_decoder(model, tmp_path):
    with Session(model, tmp_path / "pending.db") as session:
        session.observe("Amara owns the chest.", "obs")
        session.store.mark_pending("correction", key=("world", "chest", "owner"), payload="Bela")
        result = session.ask("Who owns the chest?", "blocked")
        assert result["status"] == "pending" and result["text"] == "pending"
        assert result["generation_mode"] == "runtime_policy"
        assert session.counters["emitted_tokens"] == 0
        assert session.counters["control_tokens"] == 0
        session.observe("Amara owns the chest.", "retry", clear_pending=["correction"])
        resumed = session.ask("Who owns the chest?", "resumed")
        assert resumed["generation_mode"] == "neural"
        assert resumed["evidence_status"] == "ok"


def test_unknown_or_retracted_facts_use_real_neural_unknown_path(model, tmp_path):
    with Session(model, tmp_path / "unknown.db") as session:
        result = session.ask("Who owns the chest?", "missing")
        assert result["evidence_status"] == "missing"
        assert result["generation_mode"] == "neural"
        assert result["status"] == "generated"
        assert len(result["text"]) >= 1
        session.observe("Amara owns the chest.", "obs")
        session.store.invalidate_source("obs", request_id="erase")
        result = session.ask("Who owns the chest?", "erased")
        assert result["evidence_status"] == "retracted"
        assert result["generation_mode"] == "neural"


def test_unregistered_value_refuses_instead_of_guessing(model, tmp_path):
    with Session(model, tmp_path / "vocab.db") as session:
        session.store.commit_request("external", "external", operations=[
            Operation("set", ("world", "chest", "owner"), "UnregisteredPerson")])
        before = session.store.snapshot_state()
        with pytest.raises(IntegrityError, match="admitted entity vocabulary"):
            session.ask("Who owns the chest?", "query")
        assert session.store.snapshot_state() == before
        assert session.counters["input_tokens"] == 0


def test_invalid_input_cannot_advance_state_or_create_receipt(model, tmp_path):
    with Session(model, tmp_path / "input.db") as session:
        with pytest.raises(ValueError):
            session.observe("hello🙂", "bad")
        assert session.counters["requests"] == 0
        assert session.store.snapshot_state()["revision"] == 0
        with pytest.raises(ValueError):
            session.ask("Who owns the chest?", "bad-temp", temperature=float("nan"))
        assert session.store.snapshot_state()["revision"] == 0


def test_model_or_scope_change_refused(model, tmp_path):
    path = tmp_path / "identity.db"
    with Session(model, path) as session:
        session.observe("Amara owns the chest.", "obs")
        with torch.no_grad():
            model.native.lm_head.weight[0, 0] += 0.01
        with pytest.raises(IntegrityError):
            session.ask("Who owns the chest?", "q")
    with pytest.raises(IntegrityError):
        Session(model, path)


def test_scope_change_refused_after_committed_continuation(model, tmp_path):
    path = tmp_path / "scope.db"
    with Session(model, path, scope="original") as session:
        session.observe("Amara owns the chest.", "obs")
    with pytest.raises(IntegrityError):
        Session(model, path, scope="other")


def test_capacity_failure_does_not_advance_neural_process(model, tmp_path):
    with Session(model, tmp_path / "full.db", capacity=1) as session:
        session.store.commit_request("preload", "preload", operations=[
            Operation("set", ("world", "crate", "owner"), "Bela")])
        before = session.store.snapshot_state()
        with pytest.raises(CapacityError):
            session.observe("Amara owns the chest.", "overflow")
        assert session.store.snapshot_state() == before
        assert session.counters["requests"] == 0
        assert session.store.resolve(("world", "chest", "owner")).status == "missing"


def test_external_neural_advance_requires_explicit_refresh(model, tmp_path):
    path = tmp_path / "shared.db"
    one, two = Session(model, path), Session(model, path)
    one.observe("Amara owns the chest.", "obs")
    with pytest.raises(ConflictError):
        two.ask("Who owns the chest?", "q")
    assert two.counters["requests"] == 0
    two.refresh()
    assert two.counters["observations"] == 1
    two.ask("Who owns the chest?", "q")
    one.close(); two.close()


def test_finite_saved_state_is_checked_before_resumption(model, tmp_path):
    path = tmp_path / "bad-state.db"
    with Session(model, path) as session:
        state = model.init_state(1)
        state["event"].fill_(float("nan"))
        session.store.commit_request("corrupt", "corrupt",
                                     neural_state=session._state_blob(state, session.counters),
                                     rng_state=session._rng_blob())
    with pytest.raises(IntegrityError, match="geometry"):
        Session(model, path)


def test_reentrant_request_cannot_publish_against_a_newer_mutable_revision(model, tmp_path):
    with Session(model, tmp_path / "single-flight.db") as session:
        attempted = []
        def nested(_module, _inputs, _output):
            with pytest.raises(ConflictError, match="already processing"):
                session.observe("Amara owns the chest.", "nested")
            attempted.append(True)
        handle = model.event_encoder.register_forward_hook(nested)
        try:
            session.observe("Amara owns the chest.", "outer")
        finally:
            handle.remove()
        assert attempted == [True]
        assert session.counters["requests"] == 1
        assert session.store.lookup_receipt("nested", {"unused": True}) is None


def test_normal_weight_mutation_during_inference_is_rejected_before_publish(model, tmp_path):
    with Session(model, tmp_path / "mutation.db") as session:
        def mutate(_module, _inputs, _output):
            with torch.no_grad():
                model.native.lm_head.weight[0, 0] += 0.01
        handle = model.native.lm_head.register_forward_hook(mutate)
        try:
            with pytest.raises(IntegrityError, match="changed during"):
                session.ask("Who owns the chest?", "mutated")
        finally:
            handle.remove()
        assert session.store.snapshot_state()["revision"] == 0
        assert session.counters["requests"] == 0
