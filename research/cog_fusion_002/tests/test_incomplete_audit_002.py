"""Incomplete-campaign integrity fixtures; no training or model inference."""
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

from test_training_audit_002 import audit_campaign, completed_campaign, fixture_family


PATH = Path(__file__).resolve().parents[1] / "review/audit_incomplete_campaign.py"
SPEC = importlib.util.spec_from_file_location("incomplete_audit_002", PATH)
partial = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(partial)


@pytest.fixture
def partial_campaign(audit_campaign, monkeypatch):
    root = audit_campaign
    for module in (partial, partial.audit):
        monkeypatch.setattr(module, "HERE", root)
        monkeypatch.setattr(module, "PARENT", root.with_name("cog_fusion_001"))
    monkeypatch.setattr(partial.audit, "parent_checkpoint", lambda seed: partial.PARENT / "evidence/runs" / f"memory_seed{seed}" / "final.pt")
    evidence = root / "evidence"
    lock_path = evidence / "TRAINING_LOCK.json"
    lock = json.loads(lock_path.read_text())
    for index in range(22 - len(lock["source_sha256"])):
        source = root / f"fixture_locked_{index}.txt"
        source.write_text("locked fixture")
        lock["source_sha256"][partial.audit.relative(source)] = partial.audit.sha256(source)
    lock_path.write_text(json.dumps(lock))
    exits_path = evidence / "CAMPAIGN_EXIT.json"
    exits = json.loads(exits_path.read_text()); exits[0]["returncode"] = -9
    exits_path.write_text(json.dumps(exits))
    (evidence / "CAMPAIGN_JOBS.json").write_text("[]")
    (root / "recover_interrupted.py").write_text("# fixture recovery source")
    protocol = {"source_sha256": partial.audit.sha256(root / "recover_interrupted.py"),
                "training_lock_sha256": partial.audit.sha256(lock_path),
                "campaign_jobs_sha256": partial.audit.sha256(evidence / "CAMPAIGN_JOBS.json"),
                "original_campaign_exit_present_at_declaration": False}
    (evidence / "RECOVERY_PROTOCOL.json").write_text(json.dumps(protocol))
    unresolved = evidence / "runs/continuous_seed17"
    for leaf in ("final.pt", "optimizer.pt", "RESULT.json"):
        (unresolved / leaf).unlink()
    archive = evidence / "failed_attempts/continuous_seed17"
    archive.mkdir(parents=True)
    shutil.copyfile(unresolved / "IDENTITY.json", archive / "IDENTITY.json")
    lines = (unresolved / "train.jsonl").read_text().splitlines()
    (archive / "train.jsonl").write_text("\n".join(lines[:13]) + "\n")
    (unresolved / "train.jsonl").write_text("\n".join(lines[:107]) + "\n")
    for leaf in ("CAMPAIGN_EXIT.json", "CAMPAIGN_JOBS.json", "TRAINING_LOCK.json"):
        shutil.copyfile(evidence / leaf, archive / ("ORIGINAL_" + leaf))
    (archive / "ORIGINAL_STDOUT.log").write_text("original interrupted output")
    job = {"arm": "continuous", "seed": 17, "original_returncode": -9,
           "failed_attempt": str(archive.relative_to(root)),
           "original_run_manifest": {leaf: {"sha256": partial.audit.sha256(archive / leaf)} for leaf in ("IDENTITY.json", "train.jsonl")},
           "original_log_sha256": partial.audit.sha256(archive / "ORIGINAL_STDOUT.log")}
    (archive / "ARCHIVE_MANIFEST.json").write_text(json.dumps(job))
    (evidence / "RECOVERY_STARTED.json").write_text(json.dumps({"jobs": [job],
        "original_campaign_exit_sha256": partial.audit.sha256(exits_path),
        "recovery_protocol_sha256": partial.audit.sha256(evidence / "RECOVERY_PROTOCOL.json")}))
    return root


def test_inventory_is_five_original_zero_exit_fits_and_unresolved_retry(partial_campaign):
    admitted = partial.preflight()
    assert len(admitted["runs"]) == 5 and len(admitted["checks"]) == 22
    assert {entry["name"] for entry in admitted["runs"]} == {
        "reset_training_seed17", "continuous_seed29", "reset_training_seed29", "continuous_seed41", "reset_training_seed41"}
    assert all(admitted["missing"].values()) and not any(admitted["held"].values())
    assert not (partial_campaign / "evidence/RECOVERY_RESULT.json").exists()


@pytest.mark.parametrize("problem", ["changed_locked", "positive_exit", "new_endpoint", "fake_recovery", "held_artifact", "infinite_clock"])
def test_partial_admission_does_not_select_or_complete_a_campaign(partial_campaign, problem):
    evidence = partial_campaign / "evidence"
    if problem == "changed_locked":
        (partial_campaign / "fixture_locked_0.txt").write_text("changed")
    elif problem == "positive_exit":
        path = evidence / "CAMPAIGN_EXIT.json"
        value = json.loads(path.read_text()); value[1]["returncode"] = 1
        path.write_text(json.dumps(value))
    elif problem == "new_endpoint":
        (evidence / "runs/continuous_seed17/final.pt").write_bytes(b"endpoint appeared")
    elif problem == "fake_recovery":
        (evidence / "RECOVERY_RESULT.json").write_text('{"status":"completed"}')
    elif problem == "held_artifact":
        (evidence / "evaluation").mkdir()
    else:
        path = evidence / "runs/reset_training_seed41/RESULT.json"
        value = json.loads(path.read_text()); value["cpu_seconds"] = float("inf")
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        partial.preflight()


def test_stable_snapshot_preserves_original_and_publishes_exact_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(partial.audit, "HERE", tmp_path / "study")
    source, destination = tmp_path / "trace.jsonl", tmp_path / "snapshots/trace.jsonl"
    source.write_bytes(b'{"step":1}\n{"step":2')
    before = source.read_bytes()
    result = partial.stable_snapshot(source, destination)
    assert source.read_bytes() == destination.read_bytes() == before
    assert result["sha256"] == partial.audit.sha256(source)
    assert "no process-liveness conclusion" in result["stability_scope"]
    with pytest.raises(FileExistsError):
        partial.stable_snapshot(source, destination)


def test_mutating_snapshot_source_never_publishes_an_accepted_copy(tmp_path, monkeypatch):
    monkeypatch.setattr(partial.audit, "HERE", tmp_path / "study")
    source, destination = tmp_path / "trace.jsonl", tmp_path / "snapshots/trace.jsonl"
    source.write_bytes(b"first")
    original = Path.read_bytes
    calls = 0

    def mutate(path):
        nonlocal calls
        if path == source:
            calls += 1
            if calls == 2:
                source.write_bytes(b"new logged data")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", mutate)
    with pytest.raises(ValueError, match="changed across copy"):
        partial.stable_snapshot(source, destination)
    assert not destination.exists() and not list(destination.parent.iterdir())


def test_retry_snapshot_must_match_the_preflight_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(partial, "HERE", tmp_path / "study")
    path = tmp_path / "snapshot.json"
    path.write_text('{"seed":17}')
    partial.verify_snapshot_identity({"seed": 17}, {"snapshot": "snapshot.json"})
    path.write_text('{"seed":29}')
    with pytest.raises(ValueError, match="identity changed"):
        partial.verify_snapshot_identity({"seed": 17}, {"snapshot": "snapshot.json"})


def prefix_rows():
    family = fixture_family()
    expected = partial.audit.expected_batch_counts([family] * 4)
    rows = []
    for step in range(1, 4):
        generated = 84
        physical = sum(expected[key] for key in ("input_physical_positions", "teacher_decode_physical_positions",
                       "generation_prefix_chars", "rehearsal_physical_positions")) + generated
        rows.append({**expected, "step": step, "teacher_typed_execution": True, "answer_boundaries_reset": 0,
                     "generated_chars": generated, "generation_decoder_calls": 18, "physical_token_positions": physical})
    return rows, family


def test_prefix_accounting_is_only_observed_work_and_exposure_not_probe_scoring():
    rows, family = prefix_rows()
    replay, counters = partial.observed_prefix(rows, {"seed": 17}, [family], 1000)
    assert replay["updates"] == 3 and counters["generated_chars"] == 252
    public = partial.public_replay(replay)
    assert not any(key.startswith("train_probe_") for key in public)
    assert public["total_family_draws"] == 12
    rows[0]["input_chars"] += 1
    with pytest.raises(ValueError, match="supplied-count"):
        partial.observed_prefix(rows, {"seed": 17}, [family], 1000)


def test_existing_output_refuses_before_any_partial_admission(tmp_path, monkeypatch):
    output = tmp_path / "audit.json"
    output.write_text("original")
    monkeypatch.setattr(partial, "preflight", lambda: pytest.fail("must refuse existing output first"))
    with pytest.raises(FileExistsError):
        partial.run(output, tmp_path / "snapshot")
    assert output.read_text() == "original"
