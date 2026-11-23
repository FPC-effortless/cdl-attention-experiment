"""Orchestration tests use invented receipts, bytes, and deterministic callbacks.

No real checkpoint is loaded and no held candidate performance is measured.
"""
import importlib.util
import json
from pathlib import Path
import shutil

import pytest


MODULE = Path(__file__).resolve().parents[1] / "run_evaluations.py"
SPEC = importlib.util.spec_from_file_location("evaluation_runner_002", MODULE)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


@pytest.fixture
def completed_campaign(tmp_path, monkeypatch):
    here, parent = tmp_path / "cog_fusion_002", tmp_path / "cog_fusion_001"
    monkeypatch.setattr(runner, "HERE", here)
    monkeypatch.setattr(runner, "PARENT", parent)
    monkeypatch.setattr(runner, "parent_checkpoint", lambda seed: parent / "evidence/runs" / f"memory_seed{seed}" / "final.pt")
    for source in runner.evaluation_sources():
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("# Controlled fixture source: " + source.name + "\n")
    data = here / "evidence/data"
    train = [{"fixture": index} for index in range(128)]
    panels = {panel: ([{"fixture": panel}] if panel != "train_probe" else train) for panel in runner.PANELS}
    put(data / "train.json", {"families": train})
    for panel, families in panels.items():
        payload = {"families": families}
        if panel == "train_probe":
            payload.update(source_split="train", source_indices=list(range(128)))
        put(data / (panel + ".json"), payload)
    put(data / "corpus.json", {"train": train, **panels})
    put(data / "DATA_MANIFEST.json", {
        "corpus_sha256": runner.sha256(data / "corpus.json"),
        "split_files": {panel: {"filename": panel + ".json", "sha256": runner.sha256(data / (panel + ".json"))}
                        for panel in ("train", *runner.PANELS)}})
    sources = runner.hashes([here / name for name in ("common.py", "data.py", "evaluate.py")]
                            + [data / "train.json", data / "DATA_MANIFEST.json"])
    legacy = parent / "evidence/corpus.json"
    put(legacy, {panel: [{"fixture": "legacy_" + panel}] for panel in runner.REGRESSION_PANELS})
    put(parent / "evidence/DATA_MANIFEST.json", {"sha256": runner.sha256(legacy)})
    put(parent / "evidence/TRAINING_LOCK.json", {"source_sha256": {"evidence/corpus.json": runner.sha256(legacy)}})
    put(parent / "evidence/EVALUATION_LOCK.json", {
        "source_sha256": {"fusion/evaluate.py": runner.sha256(parent / "fusion/evaluate.py")}})
    parent_hashes = {}
    for seed in runner.SEEDS:
        path = runner.parent_checkpoint(seed)
        path.parent.mkdir(parents=True)
        path.write_bytes(f"parent fixture {seed}".encode())
        parent_hashes[str(seed)] = runner.sha256(path)
        put(parent / "evidence/evaluation" / f"memory_seed{seed}" / "RESULT.json", {
            "checkpoint_sha256": parent_hashes[str(seed)], "corpus_sha256": runner.sha256(legacy),
            "model_identity": "f" * 64, "panels": {panel: {"fixture": True} for panel in runner.REGRESSION_PANELS}})
    training = {"schema": "cog-fusion-002-lock-v1", "updates": 1200, "seeds": list(runner.SEEDS),
                "arms": list(runner.ARMS), "selected_checkpoint": "fixed_final_update", "source_sha256": sources,
                "parent_sha256": parent_hashes, "teacher_updates": 100, "batch_families": 4,
                "threads_per_fit": 1, "cpu_cap_seconds_per_fit": 3600, "native_lr": .000075, "interface_lr": .0005}
    put(here / "evidence/TRAINING_LOCK.json", training)
    exits = []
    for seed in runner.SEEDS:
        for arm in runner.ARMS:
            run = here / "evidence/runs" / f"{arm}_seed{seed}"
            run.mkdir(parents=True)
            (run / "final.pt").write_bytes(f"trained fixture {arm} {seed}".encode())
            (run / "optimizer.pt").write_bytes(b"optimizer fixture")
            identity = {"study": "COG-FUSION-002", "arm": arm, "seed": seed, "steps": 1200,
                        "teacher_steps": 100, "batch_families": 4, "threads": 1,
                        "cpu_cap": 3600,
                        "native_lr": .000075, "interface_lr": .0005,
                        "parent_sha256": parent_hashes[str(seed)], "train_data_sha256": runner.sha256(data / "train.json"),
                        "source_sha256": sources, "initial_tensor_sha256": "a" * 64,
                        "train_families_sha256": "b" * 64, "shakespeare_train_sha256": "c" * 64}
            put(run / "IDENTITY.json", identity)
            put(run / "RESULT.json", {**identity, "status": "completed", "final_model_identity": "d" * 64,
                                      "batch_trace_sha256": "e" * 64, "checkpoint_sha256": runner.sha256(run / "final.pt")})
            with (run / "train.jsonl").open("w") as handle:
                for step in range(1, 1201):
                    handle.write(json.dumps({"step": step, "loss": .1, "gradient_norm": .2,
                                             "teacher_typed_execution": step <= 100}) + "\n")
            exits.append({"arm": arm, "seed": seed, "returncode": 0})
    put(here / "evidence/CAMPAIGN_EXIT.json", exits)
    return here


def test_read_only_completion_guard_and_lock_validation(completed_campaign):
    lock = runner.build_lock(workers=3, batch_families=4)
    assert len(lock["endpoints"]) == 9
    assert len(lock["completion"]["fits"]) == 6
    assert not (completed_campaign / "evidence/EVALUATION_LOCK.json").exists()
    assert not (completed_campaign / "evidence/evaluation").exists()
    path = completed_campaign / "evidence/EVALUATION_LOCK.json"
    runner.write_new_json(path, lock)
    assert runner.verify_evaluation_lock(path, runner.sha256(path)) == lock
    with pytest.raises(FileExistsError):
        runner.build_lock()


@pytest.mark.parametrize("corruption", ["missing_final", "incomplete", "checkpoint", "source", "short_trace", "nonfinite", "draw_mismatch", "cpu_cap"])
def test_any_unfinished_or_mismatched_fit_blocks_all_scoring(completed_campaign, corruption):
    run = completed_campaign / "evidence/runs/continuous_seed17"
    if corruption == "missing_final":
        (run / "RESULT.json").unlink()
    elif corruption == "incomplete":
        result = runner.read_json(run / "RESULT.json")
        result["steps"] = 1199
        put(run / "RESULT.json", result)
    elif corruption == "checkpoint":
        (run / "final.pt").write_bytes(b"changed endpoint")
    elif corruption == "source":
        (completed_campaign / "evaluate.py").write_text("changed evaluator")
    elif corruption == "short_trace":
        lines = (run / "train.jsonl").read_text().splitlines()
        (run / "train.jsonl").write_text("\n".join(lines[:-1]) + "\n")
    elif corruption == "nonfinite":
        lines = (run / "train.jsonl").read_text().splitlines()
        row = json.loads(lines[-1]); row["loss"] = float("nan")
        lines[-1] = json.dumps(row)
        (run / "train.jsonl").write_text("\n".join(lines) + "\n")
    elif corruption == "draw_mismatch":
        result = runner.read_json(run / "RESULT.json")
        result["batch_trace_sha256"] = "0" * 64
        put(run / "RESULT.json", result)
    else:
        for filename in ("RESULT.json", "IDENTITY.json"):
            receipt = runner.read_json(run / filename)
            receipt["cpu_cap"] = 7200
            put(run / filename, receipt)
    with pytest.raises((ValueError, FileNotFoundError)):
        runner.launch(check_only=True)
    assert not (completed_campaign / "evidence/EVALUATION_LOCK.json").exists()
    assert not (completed_campaign / "evidence/evaluation").exists()


def test_existing_target_is_preserved_before_any_completion_check(completed_campaign, monkeypatch):
    target = completed_campaign / "evidence/evaluation/parent_seed29/test.json"
    put(target, {"keep": "original evidence"})
    original = target.read_bytes()
    monkeypatch.setattr(runner, "verify_completed_fits", lambda: pytest.fail("target absence must be checked first"))
    with pytest.raises(FileExistsError):
        runner.launch()
    assert target.read_bytes() == original
    assert not (completed_campaign / "evidence/EVALUATION_LOCK.json").exists()


def test_explicit_signal_recovery_keeps_both_attempts_bound(completed_campaign):
    exits_path = completed_campaign / "evidence/CAMPAIGN_EXIT.json"
    exits = runner.read_json(exits_path)
    exits[0]["returncode"] = -9
    put(exits_path, exits)
    with pytest.raises(FileNotFoundError):
        runner.build_lock()
    run = completed_campaign / "evidence/runs/continuous_seed17"
    archive = completed_campaign / "evidence/failed_attempts/continuous_seed17_attempt1"
    archive.mkdir(parents=True)
    for name in ("IDENTITY.json", "train.jsonl"):
        shutil.copyfile(run / name, archive / name)
    put(completed_campaign / "evidence/RECOVERY_RESULT.json", {"status": "completed", "recovered": [{
        "arm": "continuous", "seed": 17, "status": "completed", "checkpoint_sha256": runner.sha256(run / "final.pt"),
        "failed_attempt": str(archive.relative_to(completed_campaign)), "original_returncode": -9}]})
    lock = runner.build_lock()
    assert lock["completion"]["recovery"]["status"] == "completed"
    bound = lock["completion"]["receipt_sha256"]
    assert runner.relative(archive / "train.jsonl") in bound
    assert bound[runner.relative(exits_path)] == runner.sha256(exits_path)
    (archive / "IDENTITY.json").write_text("{}")
    with pytest.raises(ValueError, match="recovery changed"):
        runner.build_lock()


def test_process_reported_training_failure_cannot_be_relabelled_as_recovery(completed_campaign):
    path = completed_campaign / "evidence/CAMPAIGN_EXIT.json"
    exits = runner.read_json(path); exits[0]["returncode"] = 1
    put(path, exits)
    with pytest.raises(ValueError, match="process-reported failure"):
        runner.build_lock()


class ModelFixture:
    memory_enabled = True

    def __init__(self, identity):
        self.identity = identity

    def eval(self):
        return self

    def model_identity(self):
        return self.identity

    def parameter_summary(self):
        return {"total": 1077834, "active": 997674}

    def init_state(self, batch):
        return {"fixture_batch": batch}


@pytest.mark.parametrize("arm", ["continuous", "reset_training", "parent"])
def test_one_load_all_panels_exact_policy_and_legacy_receipt_reuse(completed_campaign, arm):
    lock = runner.build_lock()
    endpoint = next(entry for entry in lock["endpoints"] if entry["arm"] == arm and entry["seed"] == 17)
    calls = {"load": [], "new": [], "legacy": [], "interventions": [], "verify": []}

    def load(path):
        calls["load"].append(path)
        return ModelFixture(endpoint["model_identity"])

    def new(model, families, **policy):
        calls["new"].append(policy)
        return {"summary": {"families": len(families)}, "family_rows": [], "rows": [], "policy": policy}

    def legacy(model, families, **policy):
        calls["legacy"].append(policy)
        return {"summary": {"families": len(families)}, "rows": []}

    def interventions(model, families):
        calls["interventions"].append(len(families))
        return {"summary": {"fixture": True}, "rows": []}

    kwargs = dict(load_model=load, evaluate_new=new, evaluate_legacy=legacy, legacy_interventions=interventions,
                  measure_state_bytes=lambda state: 10592, verify_inputs=lambda: calls["verify"].append(True))
    result = runner.score_endpoint(lock, endpoint, "test-lock", **kwargs)
    assert len(calls["load"]) == 1 and len(calls["new"]) == 5
    assert calls["verify"] == [True]
    for index, panel in enumerate(runner.PANELS):
        assert calls["new"][index] == {"batch_families": 4, **runner.panel_policy(arm, panel)}
        observed = runner.read_json(completed_campaign / "evidence/evaluation" / endpoint["name"] / (panel + ".json"))
        assert observed["neural_state_tensor_bytes"] == 10592
        assert observed["checkpoint_sha256"] == endpoint["checkpoint_sha256"]
        assert observed["source_sha256"] == lock["source_sha256"]
        assert observed["policy"]["reset_after_answer"] is (arm == "reset_training" and panel == "train_probe")
    regression = completed_campaign / "evidence/regression" / endpoint["name"]
    if arm == "parent":
        original = runner.PARENT / "evidence/evaluation/memory_seed17/RESULT.json"
        assert (regression / "RESULT.json").read_bytes() == original.read_bytes()
        reused = runner.read_json(regression / "REUSED.json")
        assert reused["source_sha256"] == reused["copied_result_sha256"] == runner.sha256(original)
        assert not calls["legacy"] and not calls["interventions"]
    else:
        assert calls["legacy"] == [{"batch_families": 8, "diagnostics": panel != "long"} for panel in runner.REGRESSION_PANELS]
        assert calls["interventions"] == [1]
    assert result["status"] == "completed"
    with pytest.raises(FileExistsError):
        runner.score_endpoint(lock, endpoint, "test-lock", **kwargs)
    assert len(calls["load"]) == 1


def test_worker_checks_completion_before_any_candidate_work(monkeypatch):
    def blocked(*_):
        raise ValueError("fits incomplete")
    monkeypatch.setattr(runner, "verify_evaluation_lock", blocked)
    monkeypatch.setattr(runner, "score_endpoint", lambda *_: pytest.fail("candidate must not execute"))
    with pytest.raises(ValueError, match="fits incomplete"):
        runner.worker("continuous_seed17", "fixture")
