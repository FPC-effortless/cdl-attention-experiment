"""Pure-Python audit fixtures; no Torch import, checkpoint load, or inference."""
from collections import Counter
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import math
import shutil

import pytest

from test_run_evaluations_002 import completed_campaign


PATH = Path(__file__).resolve().parents[1] / "review/audit_training.py"
SPEC = importlib.util.spec_from_file_location("training_audit_002", PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


@pytest.fixture
def audit_campaign(completed_campaign, monkeypatch):
    monkeypatch.setattr(audit, "HERE", completed_campaign)
    monkeypatch.setattr(audit, "PARENT", completed_campaign.with_name("cog_fusion_001"))
    monkeypatch.setattr(audit, "parent_checkpoint", lambda seed: audit.PARENT / "evidence/runs" / f"memory_seed{seed}" / "final.pt")
    for folder in (completed_campaign / "evidence/runs").iterdir():
        path = folder / "train.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row.update(cpu_seconds=row["step"] * .1, wall_seconds=row["step"] * .2)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        result_path = folder / "RESULT.json"
        result = json.loads(result_path.read_text())
        result.update(cpu_seconds=121.0, wall_seconds=241.0)
        result_path.write_text(json.dumps(result))
    return completed_campaign


def test_complete_endpoint_admission_and_missing_optimizer_refusal(audit_campaign):
    lock, runs, recovery, bound = audit.admit_completed_endpoints()
    assert len(runs) == 6 and recovery is None
    assert all("optimizer.pt" in name for name in bound if name.endswith("optimizer.pt"))
    (audit_campaign / "evidence/runs/reset_training_seed41/optimizer.pt").unlink()
    with pytest.raises(ValueError, match="artifact missing"):
        audit.admit_completed_endpoints()


@pytest.mark.parametrize("corruption", ["truncated", "nonfinite"])
def test_every_trace_is_checked_in_the_lightweight_completion_gate(audit_campaign, corruption):
    path = audit_campaign / "evidence/runs/reset_training_seed41/train.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if corruption == "truncated":
        rows.pop()
    else:
        rows[-1]["loss"] = float("nan")
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        audit.admit_completed_endpoints()


def test_completed_receipt_requires_finite_clocks(audit_campaign):
    path = audit_campaign / "evidence/runs/reset_training_seed41/RESULT.json"
    receipt = json.loads(path.read_text()); receipt["cpu_seconds"] = float("inf")
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="invalid clocks"):
        audit.admit_completed_endpoints()


@pytest.fixture
def recovered_audit_campaign(audit_campaign):
    evidence = audit_campaign / "evidence"
    path = evidence / "CAMPAIGN_EXIT.json"
    exits = json.loads(path.read_text()); exits[0]["returncode"] = -9
    path.write_text(json.dumps(exits))
    source = audit_campaign / "recover_interrupted.py"
    source.write_text("# frozen fixture recovery source\n")
    (evidence / "CAMPAIGN_JOBS.json").write_text("[]")
    protocol = {"source_sha256": audit.sha256(source), "training_lock_sha256": audit.sha256(evidence / "TRAINING_LOCK.json"),
                "campaign_jobs_sha256": audit.sha256(evidence / "CAMPAIGN_JOBS.json"),
                "original_campaign_exit_present_at_declaration": False}
    (evidence / "RECOVERY_PROTOCOL.json").write_text(json.dumps(protocol))
    run = evidence / "runs/continuous_seed17"
    archive = evidence / "failed_attempts/continuous_seed17"
    archive.mkdir(parents=True)
    shutil.copyfile(run / "IDENTITY.json", archive / "IDENTITY.json")
    rows = [json.loads(line) for line in (run / "train.jsonl").read_text().splitlines()[:2]]
    (archive / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    for leaf in ("CAMPAIGN_EXIT.json", "TRAINING_LOCK.json", "CAMPAIGN_JOBS.json"):
        shutil.copyfile(evidence / leaf, archive / ("ORIGINAL_" + leaf))
    (archive / "ORIGINAL_STDOUT.log").write_bytes(b"original interrupted stdout")
    summary = {"completed_updates": 2, "trailing_partial_line_bytes": 0,
               "first_reported": {key: rows[0][key] for key in ("step", "cpu_seconds", "wall_seconds")},
               "last_reported": {key: rows[-1][key] for key in ("step", "cpu_seconds", "wall_seconds")},
               "training_loop_cpu_seconds_lower_bound": rows[-1]["cpu_seconds"], "reported_counters_lower_bound": {}}
    manifest = {"original_returncode": -9, "original_trace": summary,
                "original_run_manifest": {leaf: {"sha256": audit.sha256(archive / leaf)} for leaf in ("IDENTITY.json", "train.jsonl")},
                "original_log_sha256": audit.sha256(archive / "ORIGINAL_STDOUT.log")}
    (archive / "ARCHIVE_MANIFEST.json").write_text(json.dumps(manifest))
    recovery = {"status": "completed", "held_model_scores_inspected": False,
                "original_campaign_exit_sha256": audit.sha256(evidence / "CAMPAIGN_EXIT.json"),
                "recovery_protocol_sha256": audit.sha256(evidence / "RECOVERY_PROTOCOL.json"),
                "recovered": [{"arm": "continuous", "seed": 17, "status": "completed", "original_returncode": -9,
                               "original_signal": "SIGKILL", "original_trace": summary,
                               "failed_attempt": str(archive.relative_to(audit_campaign)),
                               "archive_manifest_sha256": audit.sha256(archive / "ARCHIVE_MANIFEST.json"),
                               "checkpoint_sha256": audit.sha256(run / "final.pt"),
                               "process": {"child_cpu_seconds_including_initialization": 123.0}}]}
    (evidence / "RECOVERY_RESULT.json").write_text(json.dumps(recovery))
    return audit_campaign


def test_recovery_receipts_and_archived_costs_are_bound(recovered_audit_campaign):
    _, runs, recovery, bound = audit.admit_completed_endpoints()
    assert len(runs) == 6
    archived = recovered_audit_campaign / "evidence/failed_attempts/continuous_seed17/train.jsonl"
    assert audit.relative(archived) in bound
    assert audit.interrupted_costs(recovery)[0]["training_loop_cpu_seconds_lower_bound"] == .2


@pytest.mark.parametrize("field", ["original_campaign_exit_sha256", "recovery_protocol_sha256", "archive_manifest_sha256", "original_returncode"])
def test_recovery_hash_or_returncode_mismatch_is_rejected(recovered_audit_campaign, field):
    path = recovered_audit_campaign / "evidence/RECOVERY_RESULT.json"
    receipt = json.loads(path.read_text())
    container = receipt if field in ("original_campaign_exit_sha256", "recovery_protocol_sha256") else receipt["recovered"][0]
    container[field] = -24 if field == "original_returncode" else "0" * 64
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        audit.admit_completed_endpoints()


@pytest.mark.parametrize("cpu_values", [[-1.0], [2.0, 1.0]])
def test_negative_or_nonmonotonic_archived_clocks_are_rejected(tmp_path, cpu_values):
    path = tmp_path / "train.jsonl"
    path.write_text("".join(json.dumps({"step": index, "loss": .1, "gradient_norm": .2,
                                       "cpu_seconds": cpu, "wall_seconds": float(index)}) + "\n"
                            for index, cpu in enumerate(cpu_values, 1)))
    with pytest.raises(ValueError):
        audit.checked_trace(path, allow_partial=True)


def test_incomplete_campaign_is_rejected_before_importing_torch(tmp_path, monkeypatch):
    import builtins
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch":
            pytest.fail("Torch must not be imported before completion admission")
        return original_import(name, *args, **kwargs)

    def refuse():
        raise ValueError("all six endpoints are not complete")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(audit, "admit_completed_endpoints", refuse)
    with pytest.raises(ValueError, match="not complete"):
        audit.run(tmp_path / "audit.json")
    assert not (tmp_path / "audit.json").exists()


def test_existing_audit_is_preserved_before_admission(tmp_path, monkeypatch):
    path = tmp_path / "audit.json"
    path.write_text("original audit")
    monkeypatch.setattr(audit, "admit_completed_endpoints", lambda: pytest.fail("must refuse existing output first"))
    with pytest.raises(FileExistsError):
        audit.run(path)
    assert path.read_text() == "original audit"


def test_rng_replay_exact_call_order_hash_and_probe_coverage():
    report, state = audit.reconstruct_draws(17, 6, 1000, updates=3, batch_families=2, teacher_updates=1)
    assert report["draws"] == [
        {"step": 1, "family_indices": [2, 4], "lm_window_starts": [16, 771, 457, 424]},
        {"step": 2, "family_indices": [2, 5], "lm_window_starts": [93, 816, 607, 454]},
        {"step": 3, "family_indices": [5, 0], "lm_window_starts": [119, 487, 593, 689]},
    ]
    assert report["batch_trace_sha256"] == "c3b41a1a9a13f7191ede7db70f4768083176be1241944f0f37d3ee4531c59bfb"
    assert report["train_probe_families_sampled"] == 4
    assert report["train_probe_families_sampled_during_predicted_execution"] == 3
    assert report["train_probe_unsampled_indices"] == [1, 3]
    assert report["full_family_sample_counts"] == [1, 0, 2, 0, 1, 2]
    assert state[0] == 3


def fixture_family():
    worlds = []
    for text, question, answer in (("x", "q", "A."), ("zz", "qqqq", "BBBB.")):
        steps = []
        for observation in range(1, 9):
            steps.append({"type": "observe", "text": text})
            if observation in (3, 5, 8):
                steps.append({"type": "ask", "text": question, "answer_text": answer})
        worlds.append({"steps": steps})
    return {"worlds": worlds}


def test_counts_distinguish_real_characters_padding_and_teacher_targets():
    counts = audit.expected_batch_counts([fixture_family()])
    assert counts == {
        "input_chars": 61, "input_physical_positions": 78,
        "teacher_decode_chars": 69, "teacher_decode_physical_positions": 78,
        "answer_targets": 21, "generation_prefix_chars": 54,
        "rehearsal_targets": 512, "rehearsal_physical_positions": 512,
        "write_total": 16, "post_answer_write_total": 10, "first_post_answer_write_total": 4,
        "query_total": 6, "generated_answer_total": 6, "teacher_decoder_calls": 3,
        "semantic_encoder_calls": 11, "rehearsal_calls": 1,
    }


def trace_fixture(arm="continuous"):
    replay, _ = audit.reconstruct_draws(17, 1, 1000, updates=3, batch_families=1, teacher_updates=1)
    expected = [audit.expected_batch_counts([fixture_family()]) for _ in range(3)]
    rows = [{**counts, "step": index, "loss": .1, "gradient_norm": .2,
             "answer_boundaries_reset": 6 if arm == "reset_training" else 0,
             "teacher_typed_execution": index <= 1, "generated_chars": 21,
             "generation_decoder_calls": 18, "physical_token_positions": 743,
             "write_exact": 16, "post_answer_write_exact": 10, "first_post_answer_write_exact": 4,
             "query_exact": 6, "generated_answer_exact": 6,
             "cpu_seconds": index * .2, "wall_seconds": index * .3}
            for index, counts in enumerate(expected, 1)]
    totals = Counter()
    for row in rows:
        totals.update({key: value for key, value in row.items() if type(value) is int and key != "step"})
    result = {**totals, **{key: replay[key] for key in ("batch_trace_sha256", "distinct_training_families_seen", "training_families_available")},
              "arm": arm, "teacher_steps": 1, "cpu_cap": 3600, "cpu_seconds": .7, "wall_seconds": 1.0}
    return rows, result, replay, expected


@pytest.mark.parametrize("arm", ["continuous", "reset_training"])
def test_trace_audit_matches_counts_and_declared_policy(arm):
    observed = audit.audit_trace(*trace_fixture(arm))
    assert observed["pass"] and observed["mismatch_count"] == 0
    assert observed["reported_integer_counter_sums"]["generated_chars"] == 63
    assert observed["reported_integer_counter_sums"]["answer_boundaries_reset"] == (18 if arm == "reset_training" else 0)


def test_forged_counts_trace_digest_and_coverage_are_reported():
    rows, receipt, replay, expected = trace_fixture()
    rows[1]["input_chars"] += 1
    receipt["batch_trace_sha256"] = "0" * 64
    receipt["distinct_training_families_seen"] = 0
    result = audit.audit_trace(rows, receipt, replay, expected)
    assert not result["pass"] and result["mismatch_count"] == 4
    assert {row["field"] for row in result["first_mismatches"]} == {"input_chars", "batch_trace_sha256", "distinct_training_families_seen"}


@pytest.mark.parametrize("field,value", [("physical_token_positions", 742), ("generated_chars", 1),
                                         ("generation_decoder_calls", 40), ("answer_boundaries_reset", 1)])
def test_inconsistent_dynamic_or_state_policy_accounting_is_rejected(field, value):
    rows, receipt, replay, expected = trace_fixture()
    rows[0][field] = value
    with pytest.raises(ValueError):
        audit.audit_trace(rows, receipt, replay, expected)


def test_interrupted_attempt_cost_remains_separate_and_is_a_lower_bound(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "HERE", tmp_path)
    archived = tmp_path / "evidence/failed_attempts/continuous_seed17/train.jsonl"
    archived.parent.mkdir(parents=True)
    archived.write_text(json.dumps({"step": 1, "loss": .1, "gradient_norm": .2, "cpu_seconds": 1.25,
                                    "wall_seconds": 2.0, "physical_token_positions": 500}) + '\n{"step":2')
    receipt = {"recovered": [{"arm": "continuous", "seed": 17, "failed_attempt": str(archived.parent.relative_to(tmp_path)),
                              "original_returncode": -9, "process": {"child_cpu_seconds_including_initialization": 125.5},
                              "original_trace": {"completed_updates": 1, "trailing_partial_line_bytes": len('{"step":2'),
                                  "first_reported": {"step": 1, "cpu_seconds": 1.25, "wall_seconds": 2.0},
                                  "last_reported": {"step": 1, "cpu_seconds": 1.25, "wall_seconds": 2.0},
                                  "training_loop_cpu_seconds_lower_bound": 1.25,
                                  "reported_counters_lower_bound": {"physical_token_positions": 500}}}]}
    result = audit.interrupted_costs(receipt)[0]
    assert result["completed_updates"] == 1
    assert result["training_loop_cpu_seconds_lower_bound"] == 1.25
    assert result["replacement_child_process_cpu_seconds_including_initialization"] == 125.5
    assert result["trailing_partial_line_bytes"] > 0
    assert result["physical_token_positions_lower_bound"] == 500
    receipt["recovered"][0]["original_trace"]["training_loop_cpu_seconds_lower_bound"] = 0
    with pytest.raises(ValueError, match="disagree"):
        audit.interrupted_costs(receipt)


class TensorFixture:
    """Tiny metadata/byte stand-in; never allocates a parameter-sized array."""
    dtype = "float32"

    def __init__(self, raw, count):
        self.raw, self.count, self.shape = raw, count, (count,)

    def detach(self):
        return self

    cpu = contiguous = numpy = detach

    def tobytes(self):
        return self.raw

    def numel(self):
        return self.count

    def __ne__(self, other):
        return self.raw != other.raw


def checkpoint_fixture():
    parent = {"schema": "cog-fusion-001-v1", "config": {"native_board_enabled": False},
              "native_config": {}, "chars": [chr(index) for index in range(65)],
              "memory_enabled": True, "admission": {}, "model": {
                  "route_scale": TensorFixture(b"old interface", 84260),
                  "native.local.weight": TensorFixture(b"old backbone", 913414),
                  "native.record_board.weight": TensorFixture(b"unchanged board", 80160)}}
    final = copy.deepcopy(parent)
    final["model"]["route_scale"].raw = b"new interface"
    final["model"]["native.local.weight"].raw = b"new backbone"
    result = {"seed": 17, "arm": "continuous", "parent_sha256": "p" * 64,
              "initial_tensor_sha256": audit.state_digest(parent["model"]),
              "final_model_identity": audit.checkpoint_identity(final), "native_lr": .000075, "interface_lr": .0005,
              "cpu_seconds": 100.0, "wall_seconds": 110.0}
    replay = {"batch_trace_sha256": "b" * 64}
    final["metadata"] = {"study": "COG-FUSION-002", "step": 1200, "seed": 17, "arm": "continuous",
                         "parent_sha256": result["parent_sha256"], "batch_trace_sha256": replay["batch_trace_sha256"],
                         "generated_chars": 42, "cpu_seconds": 100.0, "wall_seconds": 110.0}
    factor = .2 + .8 * (1 + math.cos(math.pi * 1199 / 1200)) / 2
    optimizer = {"step": 1200, "python_rng": ("fixed",), "torch_rng": TensorFixture(b"rng", 3), "optimizer": {
        "param_groups": [{"params": [0, 1], "lr": result["native_lr"] * factor, "weight_decay": .01,
                          "betas": (.9, .999), "eps": 1e-8, "amsgrad": False, "maximize": False},
                         {"params": [2], "lr": result["interface_lr"] * factor, "weight_decay": .01,
                          "betas": (.9, .999), "eps": 1e-8, "amsgrad": False, "maximize": False}],
        "state": {index: {"step": 1200.0, "exp_avg": TensorFixture(b"moment", count),
                          "exp_avg_sq": TensorFixture(b"square moment", count)}
                  for index, count in ((0, 913414), (2, 84260))}}}
    backend = SimpleNamespace(isfinite=lambda _: SimpleNamespace(all=lambda: True),
                              count_nonzero=lambda value: int(value), is_tensor=lambda value: isinstance(value, TensorFixture))
    return backend, parent, final, optimizer, result, replay, ("fixed",), {"generated_chars": 42}


def test_raw_payload_audit_counts_changes_and_verifies_inactive_board_without_torch():
    report = audit.audit_checkpoint_payloads(*checkpoint_fixture())
    assert report["pass"] and report["changed_active_tensors"] == 2
    assert report["inactive_board_tensors"] == 1 and report["inactive_board_bitwise_unchanged"]
    assert report["inactive_board_parent_sha256"] == report["inactive_board_final_sha256"]
    assert report["inactive_board_optimizer_states"] == 0
    assert report["optimizer_parameter_step_range"] == [1200, 1200]


@pytest.mark.parametrize("corruption", ["board", "board_optimizer", "rng", "final_step", "optimizer_step", "missing_active",
                                         "old_moment_step", "missing_moment", "moment_shape", "moment_dtype", "betas", "eps", "amsgrad", "maximize"])
def test_raw_payload_audit_rejects_inactive_changes_and_endpoint_mismatches(corruption):
    args = checkpoint_fixture()
    _, parent, final, optimizer, result, *_ = args
    if corruption == "board":
        final["model"]["native.record_board.weight"].raw = b"changed board"
        result["final_model_identity"] = audit.checkpoint_identity(final)
    elif corruption == "board_optimizer":
        optimizer["optimizer"]["state"][1] = {"step": 1200.0}
    elif corruption == "rng":
        optimizer["python_rng"] = ("changed",)
    elif corruption == "final_step":
        final["metadata"]["step"] = 1199
    elif corruption == "optimizer_step":
        optimizer["step"] = 1199
    elif corruption == "missing_active":
        del optimizer["optimizer"]["state"][0]
    elif corruption == "old_moment_step":
        optimizer["optimizer"]["state"][0]["step"] = 1199.0
    elif corruption == "missing_moment":
        del optimizer["optimizer"]["state"][0]["exp_avg"]
    elif corruption == "moment_shape":
        optimizer["optimizer"]["state"][0]["exp_avg"].shape = (1,)
    elif corruption == "moment_dtype":
        optimizer["optimizer"]["state"][0]["exp_avg"].dtype = "float64"
    else:
        optimizer["optimizer"]["param_groups"][0][corruption] = {
            "betas": (.8, .999), "eps": 1e-7, "amsgrad": True, "maximize": True}[corruption]
    with pytest.raises(ValueError):
        audit.audit_checkpoint_payloads(*args)
