"""One prospectively declared rerun of an externally interrupted campaign job.

This file never changes the locked trainer or its command. A stalled log is not
evidence of process death: execution requires the original parent's exit receipt.
No checkpoint is loaded and no held performance is read by this program.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
SEEDS = (17, 29, 41)
ARMS = ("continuous", "reset_training")


class RecoveryRefused(RuntimeError):
    pass


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def create(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


@dataclass(frozen=True)
class Context:
    root: Path = HERE

    @property
    def evidence(self):
        return self.root / "evidence"

    @property
    def parent(self):
        return self.root.with_name("cog_fusion_001")

    def name(self, row):
        return f"{row['arm']}_seed{row['seed']}"

    def run(self, row):
        return self.evidence / "runs" / self.name(row)


def no_scoring(ctx):
    # This checks declared output locations without opening their contents.
    forbidden = ("EVALUATION_LOCK.json", "EVALUATION_JOBS.json", "EVALUATION_EXIT.json",
                 "LANGUAGE_PRESERVATION.json", "RUNTIME_RESULTS.json", "RESULTS.json",
                 "SUMMARY.json", "evaluation", "regression", "runtime", "state_interventions")
    present = [name for name in forbidden if (ctx.evidence / name).exists()]
    if present:
        raise RecoveryRefused("scoring has begun or artifacts already exist: " + ", ".join(present))


def verified_campaign(ctx):
    lock = read(ctx.evidence / "TRAINING_LOCK.json")
    if (lock.get("schema") != "cog-fusion-002-lock-v1" or lock.get("seeds") != list(SEEDS)
            or lock.get("arms") != list(ARMS) or lock.get("updates") != 1200
            or lock.get("cpu_cap_seconds_per_fit") != 3600
            or lock.get("selected_checkpoint") != "fixed_final_update"):
        raise RecoveryRefused("campaign is not the fixed 002 protocol")
    for relative, expected in lock["source_sha256"].items():
        path = ctx.root.parent / relative
        if not path.is_file() or digest(path) != expected:
            raise RecoveryRefused("locked source/data changed: " + relative)
    jobs = read(ctx.evidence / "CAMPAIGN_JOBS.json")
    identities = [(j.get("arm"), j.get("seed")) for j in jobs]
    if len(jobs) != 6 or set(identities) != {(a, s) for a in ARMS for s in SEEDS}:
        raise RecoveryRefused("campaign job inventory differs from the locked six fits")
    for job in jobs:
        command = job.get("command")
        if (not isinstance(command, list) or not all(isinstance(x, str) for x in command)
                or len(command) != 28 or Path(command[0]).resolve() != Path(sys.executable).resolve()
                or Path(command[1]).resolve() != (ctx.root / "train.py").resolve()):
            raise RecoveryRefused("unexpected original executable or command geometry")
        options = dict(zip(command[2::2], command[3::2]))
        paths = {"--parent": ctx.parent / f"evidence/runs/memory_seed{job['seed']}/final.pt",
                 "--train-data": ctx.evidence / "data/train.json",
                 "--shakespeare": ctx.parent / "evidence/reproduction_inputs/train.bin",
                 "--output": ctx.run(job)}
        expected = {"--arm": job["arm"], "--seed": str(job["seed"]), "--steps": "1200",
                    "--teacher-steps": "100", "--pairs": "4", "--threads": "1",
                    "--cpu-cap": "3600", "--native-lr": ".000075", "--lr": ".0005"}
        if set(options) != set(paths) | set(expected) or any(options[k] != v for k, v in expected.items()):
            raise RecoveryRefused("the original command is not the locked fitting configuration")
        if any(Path(options[k]).resolve() != path.resolve() for k, path in paths.items()):
            raise RecoveryRefused("original command paths differ from locked inputs/output")
        if digest(paths["--parent"]) != lock["parent_sha256"][str(job["seed"])]:
            raise RecoveryRefused("paired parent checkpoint changed")
    return lock, jobs


def protocol(ctx):
    no_scoring(ctx)
    lock, jobs = verified_campaign(ctx)
    return {
        "schema": "cog-fusion-002-interruption-recovery-protocol-v1",
        "source_sha256": digest(Path(__file__)),
        "training_lock_sha256": digest(ctx.evidence / "TRAINING_LOCK.json"),
        "campaign_jobs_sha256": digest(ctx.evidence / "CAMPAIGN_JOBS.json"),
        "fixed_jobs": [{"arm": j["arm"], "seed": j["seed"], "command": j["command"]} for j in jobs],
        "eligibility": ["original CAMPAIGN_EXIT exists and contains a negative worker returncode",
                        "no final.pt, incomplete.pt, optimizer.pt or RESULT.json exists for that attempt",
                        "no SIGXCPU, CPU cap, nonfinite training trace or Python traceback",
                        "original IDENTITY.json and at least one finite sequential training row exist",
                        "no declared held-scoring artifact exists"],
        "retry_limit_per_interrupted_job": 1,
        "retry_policy": "all eligible jobs, original campaign order, serial execution; no performance-based selection",
        "training_budget": {"updates_per_attempt": 1200, "cpu_cap_seconds_per_attempt": 3600,
                            "original_partial_work_reported_separately": True,
                            "failed_attempt_cpu_and_token_position_costs_are_additional": True},
        "inputs_and_command": "exact original argv, same parent/seed/data/source/config; restart from parent without partial weights",
        "preservation": "rename original run under failed_attempts; copy and hash original stdout and campaign receipts; retain original stdout and CAMPAIGN_EXIT",
        "restart_policy": "exclusive RECOVERY_STARTED marker; no automatic resume or second retry even if recovery itself is interrupted",
        "cause_claim": "negative signal establishes abnormal termination, not the underlying cause; the cause may remain unknown",
        "score_boundary": "artifact checks cover declared scoring paths; no checkpoint is loaded or held performance inspected here",
    }


def declare(ctx):
    if (ctx.evidence / "CAMPAIGN_EXIT.json").exists():
        raise RecoveryRefused("declare the operational rule before the original campaign exit")
    value = protocol(ctx)
    value.update(time_utc=stamp(), original_campaign_exit_present_at_declaration=False,
                 held_model_scores_inspected=False)
    create(ctx.evidence / "RECOVERY_PROTOCOL.json", value)
    return value


def tree_manifest(directory):
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise RecoveryRefused("archive must not contain symbolic links")
        if path.is_file():
            stat = path.stat()
            result[str(path.relative_to(directory))] = {
                "sha256": digest(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return result


def trace_summary(path):
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    rows, trailing = [], 0
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError):
            if index == len(lines) - 1 and not line.endswith(b"\n"):
                trailing = len(line)
                break
            raise RecoveryRefused("malformed completed training trace row")
        required = ("loss", "gradient_norm", "cpu_seconds", "wall_seconds")
        if (row.get("step") != len(rows) + 1 or any(type(row.get(k)) not in (int, float) for k in required)
                or any(not math.isfinite(v) for v in row.values() if isinstance(v, (int, float)))
                or row["cpu_seconds"] < 0 or row["wall_seconds"] < 0
                or (rows and (row["cpu_seconds"] < rows[-1]["cpu_seconds"]
                              or row["wall_seconds"] < rows[-1]["wall_seconds"]))):
            raise RecoveryRefused("nonfinite or inconsistent scientific training trace")
        rows.append(row)
    if not rows or len(rows) > 1200:
        raise RecoveryRefused("no admitted partial training trace")
    if rows[-1]["cpu_seconds"] >= 3600:
        raise RecoveryRefused("original attempt reached the prospective CPU cap")
    fields = ("step", "cpu_seconds", "wall_seconds")
    counters = {}
    for row in rows:
        for key, value in row.items():
            if type(value) is int and key not in ("step", "loss", "gradient_norm", "cpu_seconds", "wall_seconds"):
                counters[key] = counters.get(key, 0) + value
    return {"completed_updates": len(rows), "trailing_partial_line_bytes": trailing,
            "first_reported": {k: rows[0][k] for k in fields},
            "last_reported": {k: rows[-1][k] for k in fields},
            "training_loop_cpu_seconds_lower_bound": rows[-1]["cpu_seconds"],
            "reported_counters_lower_bound": counters,
            "scope": "reported training-loop CPU lower bound; excludes unreported interrupted work and initialization"}


def plan_recovery(ctx):
    no_scoring(ctx)
    expected = protocol(ctx)
    declared = read(ctx.evidence / "RECOVERY_PROTOCOL.json")
    if any(declared.get(k) != v for k, v in expected.items()):
        raise RecoveryRefused("prospective recovery declaration or its bound sources changed")
    if declared.get("original_campaign_exit_present_at_declaration") is not False:
        raise RecoveryRefused("recovery rule was not declared prospectively")
    if (ctx.evidence / "RECOVERY_STARTED.json").exists() or (ctx.evidence / "RECOVERY_RESULT.json").exists():
        raise RecoveryRefused("a recovery attempt already exists; no second launch or implicit resume")
    exit_path = ctx.evidence / "CAMPAIGN_EXIT.json"
    if not exit_path.is_file():
        raise RecoveryRefused("original campaign is not confirmed stopped; CAMPAIGN_EXIT is required")
    exits = read(exit_path)
    lock, jobs = verified_campaign(ctx)
    if (len(exits) != 6 or {(r.get("arm"), r.get("seed")) for r in exits}
            != {(a, s) for a in ARMS for s in SEEDS}):
        raise RecoveryRefused("original campaign exit receipt is incomplete")
    by_name = {ctx.name(row): row for row in exits}
    eligible = []
    for job in jobs:
        name = ctx.name(job)
        code = by_name[name].get("returncode")
        if type(code) is not int:
            raise RecoveryRefused("original worker has no definite integer exit status")
        if code == 0:
            endpoint = read(ctx.run(job) / "RESULT.json")
            if (endpoint.get("status") != "completed" or endpoint.get("steps") != 1200
                    or endpoint.get("checkpoint_sha256") != digest(ctx.run(job) / "final.pt")):
                raise RecoveryRefused("zero-exit sibling has no completed fixed endpoint: " + name)
            continue
        if code > 0 or code == -int(signal.SIGXCPU):
            raise RecoveryRefused("non-signal, scientific or CPU-cap failure is not eligible: " + name)
        try:
            sig = signal.Signals(-code).name
        except ValueError as exc:
            raise RecoveryRefused("unknown negative worker returncode") from exc
        run = ctx.run(job)
        archive = ctx.evidence / "failed_attempts" / name
        retry_log = ctx.evidence / "logs" / (name + ".recovery.log")
        if archive.exists() or retry_log.exists():
            raise RecoveryRefused("an interrupted-job retry/archive already exists: " + name)
        if not run.is_dir() or not (run / "IDENTITY.json").is_file() or not (run / "train.jsonl").is_file():
            raise RecoveryRefused("original identity and partial trace are required: " + name)
        if any((run / leaf).exists() for leaf in ("final.pt", "incomplete.pt", "optimizer.pt", "RESULT.json")):
            raise RecoveryRefused("an endpoint or CPU-cap checkpoint exists; no attempt selection: " + name)
        identity = read(run / "IDENTITY.json")
        options = dict(zip(job["command"][2::2], job["command"][3::2]))
        required = {"study": "COG-FUSION-002", "arm": job["arm"], "seed": job["seed"],
                    "steps": 1200, "teacher_steps": 100, "batch_families": 4, "threads": 1,
                    "cpu_cap": 3600, "native_lr": .000075, "interface_lr": .0005,
                    "parent_sha256": digest(options["--parent"]),
                    "train_data_sha256": digest(options["--train-data"]),
                    "shakespeare_train_sha256": digest(options["--shakespeare"])}
        if any(identity.get(k) != v for k, v in required.items()):
            raise RecoveryRefused("original attempt identity differs from the exact command: " + name)
        if not identity.get("source_sha256") or any(lock["source_sha256"].get(k) != v
                for k, v in identity["source_sha256"].items()):
            raise RecoveryRefused("original attempt used changed fitting sources")
        log = ctx.evidence / "logs" / (name + ".log")
        content = log.read_bytes()
        if any(marker in content for marker in (b"Traceback (most recent call last)",
                b"prospective CPU cap exceeded", b"nonfinite training", b"FloatingPointError")):
            raise RecoveryRefused("original log reports scientific/CPU-cap/Python failure: " + name)
        summary = trace_summary(run / "train.jsonl")
        eligible.append({**job, "name": name, "original_returncode": code, "original_signal": sig,
            "failed_attempt": str(archive.relative_to(ctx.root)),
            "original_run_manifest": tree_manifest(run), "original_log_sha256": digest(log),
            "original_log_bytes": log.stat().st_size, "original_trace": summary,
            "original_identity_mtime_ns": (run / "IDENTITY.json").stat().st_mtime_ns,
            "original_trace_mtime_ns": (run / "train.jsonl").stat().st_mtime_ns})
    if not eligible:
        raise RecoveryRefused("no signal-interrupted jobs require recovery")
    return {"time_utc": stamp(), "original_campaign_exit_sha256": digest(exit_path),
            "original_campaign_exit_mtime_ns": exit_path.stat().st_mtime_ns,
            "recovery_protocol_sha256": digest(ctx.evidence / "RECOVERY_PROTOCOL.json"),
            "jobs": eligible}


def run_process(command, cwd, logfile):
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start, started = time.perf_counter(), stamp()
    with logfile.open("x") as handle:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT)
        returncode = process.wait()
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {"returncode": returncode, "pid": process.pid, "started_utc": started, "finished_utc": stamp(),
            "wall_seconds": time.perf_counter() - start,
            "child_cpu_seconds_including_initialization": after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime}


def execute(ctx, *, runner=run_process):
    plan = plan_recovery(ctx)
    create(ctx.evidence / "RECOVERY_STARTED.json", plan)
    outcome = {"schema": "cog-fusion-002-interruption-recovery-result-v1", "status": "failed",
        "started_utc": stamp(), "recovery_protocol_sha256": plan["recovery_protocol_sha256"],
        "original_campaign_exit_sha256": plan["original_campaign_exit_sha256"],
        "held_model_scores_inspected": False, "recovered": []}
    try:
        for job in plan["jobs"]:
            no_scoring(ctx)
            verified_campaign(ctx)
            original = ctx.run(job)
            archive = ctx.root / job["failed_attempt"]
            log = ctx.evidence / "logs" / (job["name"] + ".log")
            if tree_manifest(original) != job["original_run_manifest"] or digest(log) != job["original_log_sha256"]:
                raise RecoveryRefused("original attempt changed after the completed-campaign preflight")
            archive.parent.mkdir(parents=True, exist_ok=True)
            original.rename(archive)
            if tree_manifest(archive) != job["original_run_manifest"]:
                raise RecoveryRefused("original run did not archive byte-for-byte")
            shutil.copy2(log, archive / "ORIGINAL_STDOUT.log")
            for leaf in ("TRAINING_LOCK.json", "CAMPAIGN_JOBS.json", "CAMPAIGN_EXIT.json"):
                shutil.copy2(ctx.evidence / leaf, archive / ("ORIGINAL_" + leaf))
            create(archive / "ARCHIVE_MANIFEST.json", job)
            if digest(archive / "ORIGINAL_STDOUT.log") != job["original_log_sha256"]:
                raise RecoveryRefused("original stdout did not archive byte-for-byte")
            row = {"arm": job["arm"], "seed": job["seed"], "status": "failed",
                   "failed_attempt": job["failed_attempt"], "original_returncode": job["original_returncode"],
                   "original_signal": job["original_signal"], "original_trace": job["original_trace"],
                   "command": job["command"], "archive_manifest_sha256": digest(archive / "ARCHIVE_MANIFEST.json")}
            outcome["recovered"].append(row)
            retry_log = ctx.evidence / "logs" / (job["name"] + ".recovery.log")
            row["process"] = runner(job["command"], ctx.root, retry_log)
            row["retry_log_sha256"] = digest(retry_log)
            row["combined_cpu_seconds_lower_bound"] = (job["original_trace"]["training_loop_cpu_seconds_lower_bound"]
                + row["process"]["child_cpu_seconds_including_initialization"])
            if row["process"]["returncode"] != 0:
                raise RecoveryRefused("single recovery attempt failed; no second retry: " + job["name"])
            endpoint = read(original / "RESULT.json")
            identity = read(original / "IDENTITY.json")
            if identity != read(archive / "IDENTITY.json"):
                raise RecoveryRefused("replacement attempt identity changed")
            if (endpoint.get("status") != "completed" or endpoint.get("steps") != 1200
                    or any(endpoint.get(k) != v for k, v in identity.items())
                    or endpoint.get("checkpoint_sha256") != digest(original / "final.pt")):
                raise RecoveryRefused("replacement did not publish the exact fixed endpoint")
            trace = trace_summary(original / "train.jsonl")
            if trace["completed_updates"] != 1200 or trace["trailing_partial_line_bytes"]:
                raise RecoveryRefused("replacement training trace is incomplete")
            row.update(status="completed", checkpoint_sha256=endpoint["checkpoint_sha256"],
                       replacement_trace=trace, replacement_result_sha256=digest(original / "RESULT.json"),
                       retry_training_loop_cpu_seconds=endpoint["cpu_seconds"],
                       retry_training_loop_wall_seconds=endpoint["wall_seconds"])
            old_counters = job["original_trace"]["reported_counters_lower_bound"]
            new_counters = trace["reported_counters_lower_bound"]
            row["combined_reported_counters_lower_bound"] = {
                key: old_counters.get(key, 0) + new_counters.get(key, 0)
                for key in old_counters.keys() | new_counters.keys()}
        if digest(ctx.evidence / "CAMPAIGN_EXIT.json") != plan["original_campaign_exit_sha256"]:
            raise RecoveryRefused("original campaign exit receipt changed")
        no_scoring(ctx)
        verified_campaign(ctx)
        for seed in SEEDS:
            pair = [read(ctx.run({"arm": arm, "seed": seed}) / "RESULT.json") for arm in ARMS]
            for key in ("initial_tensor_sha256", "batch_trace_sha256", "train_families_sha256", "shakespeare_train_sha256"):
                if (not isinstance(pair[0].get(key), str) or len(pair[0][key]) != 64
                        or pair[0][key] != pair[1].get(key)):
                    raise RecoveryRefused(f"recovered matched-arm provenance differs for seed {seed}: {key}")
        outcome["paired_initial_tensors_and_training_draws_verified"] = True
        outcome["status"] = "completed"
    except BaseException as exc:
        outcome["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        outcome["finished_utc"] = stamp()
        create(ctx.evidence / "RECOVERY_RESULT.json", outcome)
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--declare", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    ctx = Context()
    result = declare(ctx) if args.declare else plan_recovery(ctx) if args.check else execute(ctx)
    print(json.dumps({"mode": "declare" if args.declare else "check" if args.check else "execute",
                      "status": result.get("status", "ready"),
                      "eligible_jobs": [j["name"] for j in result.get("jobs", [])]}, sort_keys=True))


if __name__ == "__main__":
    main()
