"""Integrity evidence for the incomplete 002 campaign, never qualification.

The five original zero-exit fits are checked individually. The original -9
attempt and the unresolved permitted retry remain distinct incomplete attempts.
No complete-campaign admission is called or bypassed, no model is instantiated,
and no inference, probe, retry, or held-data evaluation is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_training as audit

HERE, PARENT = audit.HERE, audit.PARENT
INCOMPLETE = ("continuous", 17)
HELD_PATHS = ("EVALUATION_LOCK.json", "EVALUATION_JOBS.json", "EVALUATION_EXIT.json",
              "TRAINING_AUDIT.json", "LANGUAGE_PRESERVATION.json", "RUNTIME_RESULTS.json",
              "RESULTS.json", "SUMMARY.json", "QUALIFICATION.json", "evaluation", "regression",
              "runtime", "state_interventions")


def presence():
    return {name: (HERE / "evidence" / name).exists() for name in HELD_PATHS}


def fingerprint(path):
    stat = Path(path).stat()
    return stat.st_ino, stat.st_size, stat.st_mtime_ns


def stable_snapshot(source, destination):
    """Publish a copy only after byte/stat checks spanning the copy operation."""
    source, destination = Path(source), Path(destination)
    audit.require(source.is_file() and not source.is_symlink(), "snapshot source must be a regular file")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("snapshot already exists")
    first = fingerprint(source)
    raw = source.read_bytes()
    audit.require(fingerprint(source) == first and len(raw) == first[1], "snapshot source changed during read")
    checksum = hashlib.sha256(raw).hexdigest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix="snapshot_", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw); handle.flush(); os.fsync(handle.fileno())
        second = source.read_bytes()
        audit.require(fingerprint(source) == first and hashlib.sha256(second).hexdigest() == checksum,
                      "snapshot source changed across copy")
        # Hard-link publication is exclusive and never replaces an existing target.
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {"snapshot": audit.relative(destination), "sha256": checksum, "bytes": len(raw),
            "source_mtime_ns": first[2], "stability_scope": "matching bytes and file metadata during this bounded copy window; no process-liveness conclusion"}


def public_replay(replay):
    # Exposure concerns the TRAIN pool. No adequacy probe is scored or filtered.
    return {key: value for key, value in replay.items() if not key.startswith("train_probe_")}


def verify_snapshot_identity(identity, snapshot):
    audit.require(audit.read(HERE.parent / snapshot["snapshot"]) == identity,
                  "retry identity changed between admission and stable snapshot")


def coverage(replay):
    return {key: replay[key] for key in ("updates", "total_family_draws", "distinct_training_families_seen",
            "training_families_available", "full_family_sample_counts", "family_draw_sha256", "lm_start_sha256")}


def observed_prefix(rows, identity, families, lm_length):
    replay, _ = audit.reconstruct_draws(identity["seed"], len(families), lm_length, updates=len(rows))
    totals = Counter()
    for row, draw in zip(rows, replay["draws"]):
        expected = audit.expected_batch_counts([families[index] for index in draw["family_indices"]])
        for key, wanted in expected.items():
            audit.require(type(row.get(key)) is int and row[key] == wanted, "incomplete trace supplied-count mismatch: " + key)
        audit.require(row.get("teacher_typed_execution") is (row["step"] <= 100), "incomplete typed warmup differs")
        audit.require(row.get("answer_boundaries_reset") == 0, "incomplete continuous attempt recorded a reset")
        chars, calls = row["generated_chars"], row["generation_decoder_calls"]
        questions = expected["teacher_decoder_calls"]
        batch = expected["generated_answer_total"] // questions
        maxima = calls - questions
        audit.require(type(chars) is int and type(calls) is int and questions <= maxima <= 12 * questions
                      and maxima + questions * (batch - 1) <= chars <= batch * maxima,
                      "incomplete generated counts violate rollout bounds")
        physical = sum(expected[key] for key in ("input_physical_positions", "teacher_decode_physical_positions",
                       "generation_prefix_chars", "rehearsal_physical_positions")) + chars
        audit.require(row["physical_token_positions"] == physical, "incomplete physical position accounting differs")
        totals.update({key: value for key, value in row.items() if type(value) is int and key != "step"})
    return replay, dict(totals)


def preflight():
    evidence = HERE / "evidence"
    observed_held = presence()
    audit.require(not any(observed_held.values()), "held evaluation or complete-campaign audit artifacts already exist")
    lock_path = evidence / "TRAINING_LOCK.json"
    lock = audit.read(lock_path)
    audit.require(lock.get("updates") == 1200 and lock.get("seeds") == [17, 29, 41]
                  and lock.get("arms") == ["continuous", "reset_training"]
                  and lock.get("selected_checkpoint") == "fixed_final_update", "unexpected original campaign declaration")
    checks = []
    for name, expected in lock["source_sha256"].items():
        observed = audit.sha256(HERE.parent / name)
        checks.append({"path": name, "expected": expected, "observed": observed, "match": observed == expected})
    audit.require(len(checks) == 22 and all(row["match"] for row in checks), "one of the 22 locked inputs differs")
    paths = [lock_path, evidence / "CAMPAIGN_EXIT.json", evidence / "CAMPAIGN_JOBS.json",
             evidence / "RECOVERY_PROTOCOL.json", evidence / "RECOVERY_STARTED.json", HERE / "recover_interrupted.py"]
    exits = audit.read(evidence / "CAMPAIGN_EXIT.json")
    expected_jobs = {(arm, seed) for seed in audit.SEEDS for arm in audit.ARMS}
    audit.require(len(exits) == 6 and {(row["arm"], row["seed"]) for row in exits} == expected_jobs, "original exit inventory differs")
    zero_exits = [row for row in exits if row["returncode"] == 0]
    interrupted = [row for row in exits if row["returncode"] != 0]
    audit.require(len(zero_exits) == 5 and interrupted == [{"arm": "continuous", "seed": 17, "returncode": -9}],
                  "this audit requires the five normal original exits and the recorded -9 interruption")
    protocol = audit.read(evidence / "RECOVERY_PROTOCOL.json")
    started = audit.read(evidence / "RECOVERY_STARTED.json")
    audit.require(not (evidence / "RECOVERY_RESULT.json").exists(), "a recovery completion receipt now exists; do not reuse this incomplete disposition")
    audit.require(protocol["source_sha256"] == audit.sha256(HERE / "recover_interrupted.py")
                  and protocol["training_lock_sha256"] == audit.sha256(lock_path)
                  and protocol["campaign_jobs_sha256"] == audit.sha256(evidence / "CAMPAIGN_JOBS.json")
                  and protocol["original_campaign_exit_present_at_declaration"] is False, "recovery protocol binding differs")
    audit.require(started["original_campaign_exit_sha256"] == audit.sha256(evidence / "CAMPAIGN_EXIT.json")
                  and started["recovery_protocol_sha256"] == audit.sha256(evidence / "RECOVERY_PROTOCOL.json")
                  and len(started["jobs"]) == 1, "recovery-start receipt binding differs")
    job = started["jobs"][0]
    audit.require((job["arm"], job["seed"]) == INCOMPLETE and job["original_returncode"] == -9,
                  "the recorded retry is not the interrupted endpoint")
    archive = HERE / job["failed_attempt"]
    audit.require(audit.read(archive / "ARCHIVE_MANIFEST.json") == job, "archived original manifest differs from recovery-start receipt")
    for name, item in job["original_run_manifest"].items():
        audit.require(audit.sha256(archive / name) == item["sha256"], "original archived bytes differ")
    for leaf, expected in (("ORIGINAL_CAMPAIGN_EXIT.json", started["original_campaign_exit_sha256"]),
                           ("ORIGINAL_TRAINING_LOCK.json", protocol["training_lock_sha256"]),
                           ("ORIGINAL_CAMPAIGN_JOBS.json", protocol["campaign_jobs_sha256"]),
                           ("ORIGINAL_STDOUT.log", job["original_log_sha256"])):
        audit.require(audit.sha256(archive / leaf) == expected, "preserved original receipt differs: " + leaf)
    paths.extend(sorted(path for path in archive.iterdir() if path.is_file()))
    unresolved = evidence / "runs/continuous_seed17"
    missing = {leaf: not (unresolved / leaf).exists() for leaf in ("final.pt", "incomplete.pt", "optimizer.pt", "RESULT.json")}
    audit.require(all(missing.values()), "the unresolved endpoint now has a terminal artifact; inspect it separately")
    identity = audit.read(unresolved / "IDENTITY.json")
    audit.require(identity == audit.read(archive / "IDENTITY.json"), "retry identity differs from original attempt")
    runs = []
    for row in zero_exits:
        name = f"{row['arm']}_seed{row['seed']}"
        folder = evidence / "runs" / name
        result = audit.read(folder / "RESULT.json")
        initial = audit.read(folder / "IDENTITY.json")
        required = {"status": "completed", "study": "COG-FUSION-002", "arm": row["arm"], "seed": row["seed"],
                    "steps": 1200, "teacher_steps": 100, "batch_families": 4, "threads": 1, "cpu_cap": 3600,
                    "parent_sha256": lock["parent_sha256"][str(row["seed"])],
                    "native_lr": lock["native_lr"], "interface_lr": lock["interface_lr"]}
        audit.require(all(result.get(key) == value for key, value in required.items())
                      and all(result.get(key) == value for key, value in initial.items()), "completed fit identity differs: " + name)
        audit.require(all(lock["source_sha256"].get(key) == value for key, value in initial["source_sha256"].items()), "fit source differs")
        audit.require(audit.sha256(folder / "final.pt") == result["checkpoint_sha256"], "completed checkpoint hash differs")
        trace, _ = audit.checked_trace(folder / "train.jsonl", updates=1200)
        audit.require(all(type(result.get(key)) in (int, float) and math.isfinite(result[key])
                          and result[key] >= trace[-1][key] for key in ("cpu_seconds", "wall_seconds")),
                      "completed receipt has invalid or inconsistent clocks")
        paths.extend(folder / leaf for leaf in ("IDENTITY.json", "RESULT.json", "final.pt", "optimizer.pt", "train.jsonl"))
        runs.append({"name": name, "folder": folder, "result": result})
    for seed in audit.SEEDS:
        path = audit.parent_checkpoint(seed)
        audit.require(audit.sha256(path) == lock["parent_sha256"][str(seed)], "parent checkpoint differs")
        paths.append(path)
    for key, value in identity["source_sha256"].items():
        audit.require(lock["source_sha256"].get(key) == value, "incomplete attempt source differs")
    audit.require(identity["parent_sha256"] == lock["parent_sha256"]["17"] and identity["steps"] == 1200,
                  "incomplete attempt parent/budget differs")
    return {"lock": lock, "checks": checks, "held": observed_held, "runs": runs, "job": job,
            "archive": archive, "unresolved": unresolved, "identity": identity, "missing": missing,
            "input_sha256": {audit.relative(path): audit.sha256(path) for path in paths}}


def run(output, snapshot_dir):
    output, snapshot_dir = Path(output), Path(snapshot_dir)
    if output.exists() or output.is_symlink() or snapshot_dir.exists() or snapshot_dir.is_symlink():
        raise FileExistsError("partial audit and snapshot directory must both be new")
    admitted = preflight()
    families_path = HERE / "evidence/data/train.json"
    lm_path = PARENT / "evidence/reproduction_inputs/train.bin"
    families = audit.read(families_path)["families"]
    audit.require(len(families) == 2048 and lm_path.stat().st_size % 2 == 0, "TRAIN geometry differs")
    data_hash, family_hash, lm_hash = audit.sha256(families_path), audit.digest(families), audit.sha256(lm_path)
    identities = [entry["result"] for entry in admitted["runs"]] + [admitted["identity"]]
    audit.require(all(row["train_data_sha256"] == data_hash and row["train_families_sha256"] == family_hash
                      and row["shakespeare_train_sha256"] == lm_hash for row in identities), "TRAIN input receipt differs")
    inputs = dict(admitted["input_sha256"])
    inputs.update({audit.relative(families_path): data_hash, audit.relative(lm_path): lm_hash})
    protected = [Path(__file__), Path(audit.__file__), HERE / "common.py", HERE / "run_evaluations.py"]
    protected_hashes = {audit.relative(path): audit.sha256(path) for path in protected}
    inputs.update(protected_hashes)
    snapshots = {}
    for leaf, destination in (("IDENTITY.json", "retry_IDENTITY.json"), ("train.jsonl", "retry_train.jsonl")):
        source = admitted["unresolved"] / leaf
        snapshots[audit.relative(source)] = stable_snapshot(source, snapshot_dir / destination)
    verify_snapshot_identity(admitted["identity"], snapshots[audit.relative(admitted["unresolved"] / "IDENTITY.json")])
    log_source = HERE / "evidence/logs/continuous_seed17.recovery.log"
    if log_source.is_file():
        snapshots[audit.relative(log_source)] = stable_snapshot(log_source, snapshot_dir / "retry_stdout.log")
    for source, record in snapshots.items():
        inputs[source] = record["sha256"]
        inputs[record["snapshot"]] = record["sha256"]
    record = {"schema": "cog-fusion-002-incomplete-campaign-audit-v1", "campaign_status": "incomplete",
              "campaign_complete": False, "audit_status": "running", "individual_completed_fit_integrity_pass": False,
              "qualification_status": "not_evaluated_incomplete_campaign", "model_inference_calls": 0,
              "probe_evaluations": 0, "held_data_files_read": [], "source_sha256": protected_hashes,
              "input_sha256": inputs, "locked_hashes": admitted["checks"], "locked_hashes_verified": len(admitted["checks"]),
              "completed_fit_names": [entry["name"] for entry in admitted["runs"]], "held_artifacts_before": admitted["held"],
              "stable_snapshots": snapshots, "runs": [], "replays": {}, "incomplete_attempts": [],
              "unresolved_terminal_artifacts_missing": admitted["missing"], "threads": 1,
              "observation_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "scope": "individual completed-fit artifact integrity and observed incomplete-attempt accounting; no subset qualification or liveness/death determination"}
    wall, cpu = time.perf_counter(), time.process_time()
    try:
        retry_trace_source = audit.relative(admitted["unresolved"] / "train.jsonl")
        retry_trace = HERE.parent / snapshots[retry_trace_source]["snapshot"]
        for label, path, process_status, exit_code in (
                ("original", admitted["archive"] / "train.jsonl", "terminated_by_recorded_signal", -9),
                ("permitted_retry", retry_trace, "unknown", None)):
            rows, partial = audit.checked_trace(path, allow_partial=True)
            audit.require(0 < len(rows) < 1200, "incomplete attempt no longer has a proper unfinished prefix")
            replay, totals = observed_prefix(rows, admitted["identity"], families, lm_path.stat().st_size // 2)
            if label == "original":
                original = admitted["job"]["original_trace"]
                audit.require(original["completed_updates"] == len(rows) and original["reported_counters_lower_bound"] == totals
                              and original["training_loop_cpu_seconds_lower_bound"] == rows[-1]["cpu_seconds"],
                              "original interrupted trace disagrees with preserved receipt")
            record["incomplete_attempts"].append({"name": "continuous_seed17", "arm": "continuous", "seed": 17,
                "attempt": label, "observed_updates": len(rows), "process_status": process_status, "exit_code": exit_code,
                "signal": "SIGKILL" if label == "original" else None, "os_termination_confirmed": label == "original",
                "cpu_seconds_lower_bound": rows[-1]["cpu_seconds"], "wall_seconds_lower_bound": rows[-1]["wall_seconds"],
                "counters_lower_bound": totals, "coverage": coverage(replay), "reconstructed_draws": public_replay(replay),
                "trace_source": audit.relative(admitted["archive"] / "train.jsonl") if label == "original" else retry_trace_source,
                "trace_used": audit.relative(path), "trace_sha256": audit.sha256(path), "trailing_partial_line_bytes": partial,
                "prefix_accounting_matches_reconstructed_inputs": True,
                "draw_digest_status": "expected prefix reconstructed from declared seed and logged update count; no endpoint draw digest exists",
                "cost_scope": "observed logged training-loop lower bounds; excludes initialization, unlogged work and any subsequent activity",
                "process_status_basis": "original CAMPAIGN_EXIT returncode -9" if label == "original" else
                    "no final/optimizer/RESULT or recovery exit receipt; stable snapshot does not determine process status"})
        os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
        import torch
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        record["torch_version"] = torch.__version__
        for seed in audit.SEEDS:
            replay, rng = audit.reconstruct_draws(seed, len(families), lm_path.stat().st_size // 2)
            record["replays"][str(seed)] = public_replay(replay)
            expected_rows = [audit.expected_batch_counts([families[index] for index in draw["family_indices"]]) for draw in replay["draws"]]
            parent = torch.load(audit.parent_checkpoint(seed), map_location="cpu", weights_only=True)
            for entry in (entry for entry in admitted["runs"] if entry["result"]["seed"] == seed):
                folder, result = entry["folder"], entry["result"]
                rows, _ = audit.checked_trace(folder / "train.jsonl", updates=1200)
                accounting = audit.audit_trace(rows, result, replay, expected_rows)
                final = torch.load(folder / "final.pt", map_location="cpu", weights_only=True)
                optimizer = torch.load(folder / "optimizer.pt", map_location="cpu", weights_only=True)
                tensors = audit.audit_checkpoint_payloads(torch, parent, final, optimizer, result, replay, rng,
                                                          accounting["reported_integer_counter_sums"])
                record["runs"].append({"name": entry["name"], "arm": result["arm"], "seed": seed, "steps": 1200,
                    "cpu_seconds": result["cpu_seconds"], "wall_seconds": result["wall_seconds"],
                    "counters": accounting["reported_integer_counter_sums"], "coverage": coverage(replay),
                    "checkpoint_path": audit.relative(folder / "final.pt"), "checkpoint_sha256": audit.sha256(folder / "final.pt"),
                    "optimizer_path": audit.relative(folder / "optimizer.pt"), "optimizer_sha256": audit.sha256(folder / "optimizer.pt"),
                    "result_sha256": audit.sha256(folder / "RESULT.json"), "individual_integrity_pass": accounting["pass"] and tensors["pass"],
                    "accounting": accounting, "checkpoints": tensors,
                    "interpretation": "completed-fit integrity only; no campaign, TRAIN-distribution or held-test qualification"})
                del final, optimizer
                gc.collect()
            del parent
            gc.collect()
        for name, wanted in inputs.items():
            audit.require(audit.sha256(HERE.parent / name) == wanted, "audited input changed during inspection: " + name)
        for row in admitted["checks"]:
            audit.require(audit.sha256(HERE.parent / row["path"]) == row["expected"], "locked input changed during audit")
        record["held_artifacts_after"] = presence()
        audit.require(record["held_artifacts_after"] == admitted["held"], "held artifact presence changed during audit")
        audit.require(not (HERE / "evidence/RECOVERY_RESULT.json").exists()
                      and all(not (admitted["unresolved"] / leaf).exists() for leaf in admitted["missing"]),
                      "unresolved endpoint acquired a terminal artifact during audit")
        record["individual_completed_fit_integrity_pass"] = len(record["runs"]) == 5 and all(row["individual_integrity_pass"] for row in record["runs"])
        record["audit_status"] = "completed" if record["individual_completed_fit_integrity_pass"] else "failed"
    except Exception as exc:
        record["audit_status"] = "failed"
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        completed_cpu = sum(row["cpu_seconds"] for row in record["runs"])
        incomplete_cpu = sum(row["cpu_seconds_lower_bound"] for row in record["incomplete_attempts"])
        record["costs"] = {"completed_training_loop_cpu_seconds": completed_cpu,
            "incomplete_attempt_training_loop_cpu_seconds_lower_bound": incomplete_cpu,
            "observed_training_loop_cpu_seconds_lower_bound": completed_cpu + incomplete_cpu,
            "scope": "sum of recorded per-process training-loop CPU; incomplete prefixes are lower bounds; overlapping wall times are not added as elapsed campaign time"}
        totals = Counter()
        for row in record["runs"]: totals.update(row["counters"])
        for row in record["incomplete_attempts"]: totals.update(row["counters_lower_bound"])
        record["observed_counters_lower_bound"] = dict(totals)
        record["audit_cpu_seconds"] = time.process_time() - cpu
        record["audit_wall_seconds"] = time.perf_counter() - wall
        record["held_artifacts_after"] = presence()
        audit.create(output, record)
    audit.require(record["audit_status"] == "completed", "some individual completed-fit checks failed; no qualification permitted")
    print(json.dumps({"output": str(output), "campaign_status": record["campaign_status"],
                      "individual_completed_fit_integrity_pass": record["individual_completed_fit_integrity_pass"],
                      "completed_fits": len(record["runs"]), "incomplete_attempts":
                      [{key: row[key] for key in ("attempt", "observed_updates", "process_status", "cpu_seconds_lower_bound")}
                       for row in record["incomplete_attempts"]]}), flush=True)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "evidence/INCOMPLETE_CAMPAIGN_AUDIT.json")
    parser.add_argument("--snapshot-dir", type=Path, default=HERE / "evidence/incomplete_snapshots")
    args = parser.parse_args()
    run(args.output, args.snapshot_dir)
