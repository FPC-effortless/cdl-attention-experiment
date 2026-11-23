"""Typed numeric executor tests, separate from the prospective learned fit."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fusion.composition import RecordComputer, RecordRuntime, StaleDerivedError
from fusion.store import ConflictError, InjectedFailure, IntegrityError


A, B, OUT, OTHER = [("numeric", name, "vector") for name in ("first", "second", "result", "other")]
AV, BV = [0.6, -0.4, 0.1, 0.7, -0.2, 0.3, -0.5, 0.2], [-0.2, 0.1, -0.4, 0.2, 0.6, -0.1, 0.3, 0.5]


@pytest.fixture
def computer():
    torch.set_num_threads(1)
    torch.manual_seed(1701)
    return RecordComputer().eval()


def populated(model, path):
    runtime = RecordRuntime(model, path)
    runtime.put(A, AV, "source-a")
    runtime.put(B, BV, "source-b")
    runtime.put(OTHER, [0.25] * 8, "source-other")
    return runtime


def test_controller_budget_first_input_gradient_and_reset_ablation(computer):
    assert sum(p.numel() for p in computer.parameters()) == 4296
    assert computer.init_state(1).numel() * computer.init_state(1).element_size() == 128
    first = torch.tensor([AV], requires_grad=True)
    second = torch.tensor([BV])
    result = computer(first, second)
    gradient, = torch.autograd.grad(result.sum(), first)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    reset = computer(first, second, reset_between=True)
    altered_reset = computer(first * -1, second, reset_between=True)
    assert torch.equal(reset, altered_reset)
    assert not torch.equal(result, computer(first * -1, second))


def test_real_record_values_compute_write_and_untouched_preservation(computer, tmp_path):
    with populated(computer, tmp_path / "actual.db") as runtime:
        other = runtime.store.resolve(OTHER)
        expected = computer(torch.tensor([AV]), torch.tensor([BV]))[0].detach().tolist()
        receipt = runtime.compute(A, B, OUT, "compute")
        assert receipt["reply"]["values"] == expected
        assert runtime.read_derived(OUT) == expected
        assert runtime.store.validate_packet(other)
        envelope = runtime.store.resolve(OUT).value
        assert len(envelope["parents"]) == 2
        assert [p["key"] for p in envelope["parents"]] == [list(A), list(B)]
        assert runtime.operations == 1
        replay = runtime.compute(A, B, OUT, "compute")
        assert replay["replayed"] and runtime.operations == 1


@pytest.mark.parametrize("changed", [A, B])
def test_either_source_mutation_invalidates_derived(computer, tmp_path, changed):
    with populated(computer, tmp_path / "changed.db") as runtime:
        runtime.compute(A, B, OUT, "compute")
        runtime.put(OTHER, [-0.25] * 8, "unrelated-change")
        assert len(runtime.read_derived(OUT)) == 8
        runtime.put(changed, [-0.5] * 8, "relevant-change")
        with pytest.raises(StaleDerivedError):
            runtime.read_derived(OUT)


@pytest.mark.parametrize("source", ["source-a", "source-b"])
def test_either_source_erasure_invalidates_derived(computer, tmp_path, source):
    with populated(computer, tmp_path / "erased.db") as runtime:
        runtime.compute(A, B, OUT, "compute")
        runtime.store.invalidate_source(source, request_id="erase")
        with pytest.raises(StaleDerivedError):
            runtime.read_derived(OUT)


def test_restart_between_actual_first_and_second_reads(computer, tmp_path):
    path = tmp_path / "staged.db"
    with populated(computer, path) as runtime:
        runtime.begin_read(A, "begin")
        controller = runtime.state["record_controller"].clone()
        assert torch.count_nonzero(controller)
    with RecordRuntime(computer, path) as resumed:
        assert torch.equal(controller, resumed.state["record_controller"])
        completed = resumed.finish_read(B, OUT, "finish")
        expected = computer(torch.tensor([AV]), torch.tensor([BV]))[0].detach().tolist()
        assert completed["reply"]["values"] == expected
        assert resumed.read_derived(OUT) == expected
        assert resumed.pending_read is None
        # Retrying the first step cannot resurrect a completed staged operation.
        assert resumed.begin_read(A, "begin")["replayed"]
        assert resumed.pending_read is None


def test_staged_first_source_change_refuses_and_explicit_reset_recovers(computer, tmp_path):
    with populated(computer, tmp_path / "stale-stage.db") as runtime:
        runtime.begin_read(A, "begin")
        with pytest.raises(ConflictError):
            runtime.compute(A, B, OUT, "new-unfinished")
        runtime.put(A, [0.1] * 8, "corrected")
        with pytest.raises(StaleDerivedError):
            runtime.finish_read(B, OUT, "finish")
        runtime.reset_controller("cancel")
        assert runtime.pending_read is None
        assert not torch.count_nonzero(runtime.state["record_controller"])
        runtime.compute(A, B, OUT, "new")
        assert len(runtime.read_derived(OUT)) == 8


@pytest.mark.parametrize("stage", ["after_operations", "after_commit"])
def test_computation_state_and_derived_publication_are_atomic(computer, tmp_path, stage):
    with populated(computer, tmp_path / "fault.db") as runtime:
        original = runtime.store.snapshot_state()
        with pytest.raises(InjectedFailure):
            runtime.compute(A, B, OUT, "compute", failpoint=stage)
        if stage == "after_operations":
            assert runtime.store.snapshot_state() == original
            assert runtime.store.resolve(OUT).status == "missing"
            assert runtime.operations == 0
        else:
            assert runtime.operations == 1
            assert len(runtime.read_derived(OUT)) == 8
        retry = runtime.compute(A, B, OUT, "compute")
        assert retry["replayed"] == (stage == "after_commit")
        assert runtime.operations == 1


def test_missing_pending_alias_and_invalid_vectors_refused(computer, tmp_path):
    with populated(computer, tmp_path / "invalid.db") as runtime:
        with pytest.raises(ValueError):
            runtime.compute(A, B, A, "alias")
        with pytest.raises(IntegrityError):
            runtime.put(OTHER, [float("nan")] * 8, "nan")
        with pytest.raises(IntegrityError):
            runtime.put(OTHER, [1] * 7, "short")
        runtime.store.mark_pending("pending", key=B)
        with pytest.raises(ConflictError):
            runtime.compute(A, B, OUT, "blocked")


def test_checkpoint_identity_and_exact_numeric_continuation(computer, tmp_path):
    checkpoint = tmp_path / "computer.pt"
    torch.save(computer.checkpoint_payload(purpose="structural test"), checkpoint)
    restored = RecordComputer.from_checkpoint(checkpoint).eval()
    assert restored.model_identity() == computer.model_identity()
    assert torch.equal(restored(torch.tensor([AV]), torch.tensor([BV])),
                       computer(torch.tensor([AV]), torch.tensor([BV])))


def test_same_revision_snapshot_guards_second_source_before_commit(computer, tmp_path, monkeypatch):
    with populated(computer, tmp_path / "interleaving.db") as runtime:
        original_resolve = runtime.store.resolve
        changed = False
        def read_then_change(key, *args, **kwargs):
            nonlocal changed
            packet = original_resolve(key, *args, **kwargs)
            if key == A and not changed:
                changed = True
                runtime.put(B, [-0.3] * 8, "concurrent-change")
            return packet
        monkeypatch.setattr(runtime.store, "resolve", read_then_change)
        with pytest.raises(ConflictError, match="share"):
            runtime.compute(A, B, OUT, "stale")
        assert original_resolve(OUT).status == "missing"
