"""Operational recovery checks with temporary text/checkpoint-byte fixtures.

These tests do not launch, signal, train, or score a model.
"""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import recover_interrupted as recovery


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def trace(path, steps, *, cpu_scale=1.0):
    path.write_text("".join(json.dumps({"step": index, "loss": 1.0, "gradient_norm": .5,
        "cpu_seconds": index * cpu_scale, "wall_seconds": index * cpu_scale + .1,
        "physical_token_positions": 10, "input_chars": 4,
        "teacher_typed_execution": index <= 100}) + "\n" for index in range(1, steps + 1)))


@pytest.fixture
def fixture(tmp_path):
    ctx = recovery.Context(tmp_path / "cog_fusion_002")
    ctx.root.mkdir()
    (ctx.root / "train.py").write_text("# mock training source; never executed\n")
    train = ctx.evidence / "data/train.json"
    put(train, {"families": []})
    language = ctx.parent / "evidence/reproduction_inputs/train.bin"
    language.parent.mkdir(parents=True)
    language.write_bytes(b"training-only-fixture")
    parents = {}
    for seed in recovery.SEEDS:
        path = ctx.parent / f"evidence/runs/memory_seed{seed}/final.pt"
        path.parent.mkdir(parents=True)
        path.write_bytes(f"parent-{seed}".encode())
        parents[str(seed)] = recovery.digest(path)
    sources = {str(path.relative_to(ctx.root.parent)): recovery.digest(path)
               for path in (ctx.root / "train.py", train)}
    lock = {"schema": "cog-fusion-002-lock-v1", "seeds": list(recovery.SEEDS), "arms": list(recovery.ARMS),
            "updates": 1200, "cpu_cap_seconds_per_fit": 3600, "selected_checkpoint": "fixed_final_update",
            "source_sha256": sources, "parent_sha256": parents}
    put(ctx.evidence / "TRAINING_LOCK.json", lock)
    jobs, exits = [], []
    for seed in recovery.SEEDS:
        for arm in recovery.ARMS:
            job = {"arm": arm, "seed": seed, "pid": seed}
            run = ctx.run(job)
            run.mkdir(parents=True)
            parent = ctx.parent / f"evidence/runs/memory_seed{seed}/final.pt"
            job["command"] = [sys.executable, str(ctx.root / "train.py"), "--parent", str(parent),
                "--train-data", str(train), "--shakespeare", str(language), "--output", str(run),
                "--arm", arm, "--seed", str(seed), "--steps", "1200", "--teacher-steps", "100",
                "--pairs", "4", "--threads", "1", "--cpu-cap", "3600", "--native-lr", ".000075", "--lr", ".0005"]
            identity = {"study": "COG-FUSION-002", "arm": arm, "seed": seed, "steps": 1200,
                "teacher_steps": 100, "batch_families": 4, "threads": 1, "cpu_cap": 3600,
                "native_lr": .000075, "interface_lr": .0005, "parent_sha256": parents[str(seed)],
                "train_data_sha256": recovery.digest(train), "shakespeare_train_sha256": recovery.digest(language),
                "source_sha256": sources, "initial_tensor_sha256": f"{seed:064x}",
                "train_families_sha256": "b" * 64}
            put(run / "IDENTITY.json", identity)
            trace(run / "train.jsonl", 3)
            log = ctx.evidence / "logs" / (ctx.name(job) + ".log")
            log.parent.mkdir(exist_ok=True)
            log.write_bytes(b"original stdout bytes\n")
            interrupted = arm == "continuous" and seed == 17
            exits.append({"arm": arm, "seed": seed, "returncode": -9 if interrupted else 0})
            if not interrupted:
                (run / "final.pt").write_bytes(b"fixture checkpoint")
                put(run / "RESULT.json", {**identity, "status": "completed", "batch_trace_sha256": "a" * 64,
                    "checkpoint_sha256": recovery.digest(run / "final.pt"), "cpu_seconds": 600., "wall_seconds": 650.})
            jobs.append(job)
    put(ctx.evidence / "CAMPAIGN_JOBS.json", jobs)
    return ctx, jobs, exits


def stopped(fixture):
    ctx, jobs, exits = fixture
    recovery.declare(ctx)
    put(ctx.evidence / "CAMPAIGN_EXIT.json", exits)
    return ctx, jobs, exits


def test_recovery_requires_original_campaign_exit_and_prospective_rule(fixture):
    ctx, _, exits = fixture
    recovery.declare(ctx)
    with pytest.raises(recovery.RecoveryRefused, match="CAMPAIGN_EXIT"):
        recovery.plan_recovery(ctx)
    assert not (ctx.evidence / "RECOVERY_STARTED.json").exists()
    with pytest.raises(FileExistsError):
        recovery.declare(ctx)
    put(ctx.evidence / "CAMPAIGN_EXIT.json", exits)
    with pytest.raises(recovery.RecoveryRefused, match="before the original campaign exit"):
        recovery.declare(ctx)


@pytest.mark.parametrize("problem", ["positive", "cpu_signal", "cap_trace", "incomplete", "final", "traceback", "nonfinite", "scoring", "source", "existing_archive", "second_launch"])
def test_recovery_refuses_scientific_completed_changed_or_already_attempted_cases(fixture, problem):
    ctx, jobs, exits = stopped(fixture)
    run = ctx.run(jobs[0])
    if problem in ("positive", "cpu_signal"):
        exits[0]["returncode"] = 1 if problem == "positive" else -24
        put(ctx.evidence / "CAMPAIGN_EXIT.json", exits)
    elif problem == "cap_trace":
        trace(run / "train.jsonl", 3, cpu_scale=1200.)
    elif problem in ("incomplete", "final"):
        (run / (problem + ".pt")).write_bytes(b"existing endpoint must not be selected away")
    elif problem == "traceback":
        (ctx.evidence / "logs/continuous_seed17.log").write_text("Traceback (most recent call last)\n")
    elif problem == "nonfinite":
        rows = [json.loads(x) for x in (run / "train.jsonl").read_text().splitlines()]
        rows[-1]["loss"] = float("nan")
        (run / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    elif problem == "scoring":
        put(ctx.evidence / "EVALUATION_LOCK.json", {})
    elif problem == "source":
        (ctx.root / "train.py").write_text("# changed source\n")
    elif problem == "existing_archive":
        (ctx.evidence / "failed_attempts/continuous_seed17").mkdir(parents=True)
    else:
        put(ctx.evidence / "RECOVERY_STARTED.json", {})
    with pytest.raises(recovery.RecoveryRefused):
        recovery.plan_recovery(ctx)
    assert run.is_dir()


def test_serial_exact_retry_preserves_bytes_identity_exit_and_cost(fixture):
    ctx, jobs, _ = stopped(fixture)
    original = ctx.run(jobs[0])
    old_files = {p.name: p.read_bytes() for p in original.iterdir()}
    old_log = (ctx.evidence / "logs/continuous_seed17.log").read_bytes()
    old_exit = (ctx.evidence / "CAMPAIGN_EXIT.json").read_bytes()
    calls = []

    def runner(command, cwd, logfile):
        calls.append(list(command))
        assert command == jobs[0]["command"] and cwd == ctx.root
        assert not original.exists()
        original.mkdir()
        archive = ctx.evidence / "failed_attempts/continuous_seed17"
        identity = recovery.read(archive / "IDENTITY.json")
        put(original / "IDENTITY.json", identity)
        trace(original / "train.jsonl", 1200, cpu_scale=.5)
        (original / "final.pt").write_bytes(b"replacement fixture checkpoint")
        put(original / "RESULT.json", {**identity, "status": "completed", "batch_trace_sha256": "a" * 64,
            "checkpoint_sha256": recovery.digest(original / "final.pt"), "cpu_seconds": 600., "wall_seconds": 650.})
        logfile.write_text("fixture-only runner; no training performed\n")
        return {"returncode": 0, "pid": 123, "started_utc": "fixture start", "finished_utc": "fixture end",
                "wall_seconds": 650.1, "child_cpu_seconds_including_initialization": 600.2}

    result = recovery.execute(ctx, runner=runner)
    assert result["status"] == "completed" and result["paired_initial_tensors_and_training_draws_verified"]
    assert calls == [jobs[0]["command"]]
    archive = ctx.root / result["recovered"][0]["failed_attempt"]
    for name, before in old_files.items():
        assert (archive / name).read_bytes() == before
    assert (archive / "ORIGINAL_STDOUT.log").read_bytes() == old_log
    assert (ctx.evidence / "logs/continuous_seed17.log").read_bytes() == old_log
    assert (ctx.evidence / "CAMPAIGN_EXIT.json").read_bytes() == old_exit
    row = result["recovered"][0]
    assert row["combined_cpu_seconds_lower_bound"] == 603.2
    assert row["original_trace"]["reported_counters_lower_bound"]["physical_token_positions"] == 30
    with pytest.raises(recovery.RecoveryRefused, match="already exists"):
        recovery.execute(ctx, runner=runner)
    assert len(calls) == 1


def test_failed_retry_is_retained_and_never_repeated(fixture):
    ctx, jobs, _ = stopped(fixture)
    calls = []

    def failed(command, cwd, logfile):
        calls.append(command)
        logfile.write_text("mock retry failed\n")
        return {"returncode": -9, "child_cpu_seconds_including_initialization": .1}

    with pytest.raises(recovery.RecoveryRefused, match="no second retry"):
        recovery.execute(ctx, runner=failed)
    result = recovery.read(ctx.evidence / "RECOVERY_RESULT.json")
    assert result["status"] == "failed" and result["recovered"][0]["status"] == "failed"
    assert (ctx.evidence / "failed_attempts/continuous_seed17/IDENTITY.json").is_file()
    with pytest.raises(recovery.RecoveryRefused, match="already exists"):
        recovery.execute(ctx, runner=failed)
    assert len(calls) == 1


def test_partial_last_trace_write_keeps_only_verified_cost_lower_bound(fixture):
    ctx, jobs, _ = stopped(fixture)
    path = ctx.run(jobs[0]) / "train.jsonl"
    with path.open("ab") as handle:
        handle.write(b'{"step":4,"loss":')
    job, = recovery.plan_recovery(ctx)["jobs"]
    assert job["original_trace"]["completed_updates"] == 3
    assert job["original_trace"]["trailing_partial_line_bytes"] > 0
    assert job["original_trace"]["training_loop_cpu_seconds_lower_bound"] == 3.
