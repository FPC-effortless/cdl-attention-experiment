"""Lock and score the nine fixed endpoints after all six 002 fits complete.

One single-thread worker loads each endpoint once for all five new panels and
the inherited regression evaluator. Frozen-parent regression receipts are reused
byte-for-byte. This runner never changes a checkpoint, fits a model, selects an
endpoint, or adds a new intervention. Existing output targets are never replaced.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from common import HERE, PARENT, SEEDS, ARMS, parent_checkpoint, sha256


PANELS = ("train_probe", "dev", "test", "fresh", "long")
REGRESSION_PANELS = ("test", "fresh", "long")
STEPS = 1200
LOCK_SCHEMA = "cog-fusion-002-evaluation-lock-v1"


def read_json(path):
    return json.loads(Path(path).read_text())


def write_new_json(path, value):
    """Exclusive creation also protects against a second concurrent launcher."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def relative(path):
    return str(Path(path).resolve().relative_to(HERE.parent.resolve()))


def hashes(paths):
    return {relative(path): sha256(path) for path in paths}


def verify_hashes(expected, description):
    for name, wanted in expected.items():
        path = HERE.parent / name
        if not path.is_file() or sha256(path) != wanted:
            raise ValueError(f"{description} changed or missing: {name}")


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def endpoint_names():
    return [f"{arm}_seed{seed}" for seed in SEEDS for arm in (*ARMS, "parent")]


def evaluation_sources():
    return [*(HERE / name for name in ("common.py", "data.py", "evaluate.py", "run_evaluations.py", "summarize.py")),
            *(PARENT / "fusion" / name for name in ("__init__.py", "data.py", "engine.py", "evaluate.py", "model.py")),
            *(PARENT / "fusion/native" / name for name in ("__init__.py", "base_model.py", "model.py", "scripted_core.py"))]


def refuse_existing_targets():
    evidence = HERE / "evidence"
    targets = [evidence / name for name in ("EVALUATION_LOCK.json", "EVALUATION_JOBS.json", "EVALUATION_EXIT.json")]
    targets += [evidence / category / name for category in ("evaluation", "regression") for name in endpoint_names()]
    existing = [str(path) for path in targets if path.exists() or path.is_symlink()]
    if existing:
        raise FileExistsError("evaluation targets already exist; no overwrite or implicit resume: " + ", ".join(existing))


def verify_completed_fits():
    """Read receipts and hashes only. Import no candidate and score no data."""
    evidence = HERE / "evidence"
    training_path = evidence / "TRAINING_LOCK.json"
    training = read_json(training_path)
    if (training.get("schema") != "cog-fusion-002-lock-v1" or training.get("updates") != STEPS
            or training.get("seeds") != list(SEEDS) or training.get("arms") != list(ARMS)
            or training.get("selected_checkpoint") != "fixed_final_update"):
        raise ValueError("training lock does not describe the fixed six 1200-update endpoints")
    verify_hashes(training["source_sha256"], "locked training dependency")
    exit_path = evidence / "CAMPAIGN_EXIT.json"
    exits = read_json(exit_path)
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    observed = [(row.get("arm"), row.get("seed")) for row in exits]
    if len(observed) != 6 or set(observed) != expected or any(type(row.get("returncode")) is not int for row in exits):
        raise ValueError("campaign exit receipt must identify all six fixed fits")
    failed = {(row["arm"], row["seed"]): row for row in exits if row["returncode"] != 0}
    recovered, recovery = {}, None
    recovery_paths = []
    recovery_path = evidence / "RECOVERY_RESULT.json"
    if failed:
        if any(row["returncode"] >= 0 for row in failed.values()):
            raise ValueError("a scientific or process-reported failure cannot be admitted as signal recovery")
        recovery = read_json(recovery_path)
        entries = recovery.get("recovered", [])
        recovered = {(row.get("arm"), row.get("seed")): row for row in entries}
        if (recovery.get("status") != "completed" or len(entries) != len(failed) or set(recovered) != set(failed)
                or any(row.get("status") != "completed" for row in entries)):
            raise ValueError("every interrupted fit requires an explicit completed recovery receipt")
        recovery_paths.append(recovery_path)
        for row in entries:
            archive = (HERE / row["failed_attempt"]).resolve()
            archive.relative_to(HERE.resolve())
            if not archive.is_dir() or not all((archive / name).is_file() for name in ("IDENTITY.json", "train.jsonl")):
                raise ValueError("the interrupted attempt's identity and training log must remain preserved")
            recovery_paths.extend(sorted(path for path in archive.rglob("*") if path.is_file()))
    elif recovery_path.exists():
        raise ValueError("unexpected recovery receipt without an interrupted original fit")
    train_hash = training["source_sha256"][relative(evidence / "data/train.json")]
    receipt_paths = [training_path, exit_path, *recovery_paths]
    records = []
    for seed in SEEDS:
        parent_hash = sha256(parent_checkpoint(seed))
        if parent_hash != training["parent_sha256"][str(seed)]:
            raise ValueError(f"frozen parent changed for seed {seed}")
        pair = []
        for arm in ARMS:
            name = f"{arm}_seed{seed}"
            run = evidence / "runs" / name
            receipt = read_json(run / "RESULT.json")
            identity = read_json(run / "IDENTITY.json")
            required = {"study": "COG-FUSION-002", "status": "completed", "arm": arm, "seed": seed,
                        "steps": STEPS, "teacher_steps": training["teacher_updates"],
                        "batch_families": training["batch_families"], "threads": training["threads_per_fit"],
                        "cpu_cap": training["cpu_cap_seconds_per_fit"],
                        "native_lr": training["native_lr"], "interface_lr": training["interface_lr"],
                        "parent_sha256": parent_hash, "train_data_sha256": train_hash}
            if any(receipt.get(key) != value for key, value in required.items()):
                raise ValueError(f"incomplete or mismatched fixed endpoint: {name}")
            if any(receipt.get(key) != value for key, value in identity.items()):
                raise ValueError(f"identity/result mismatch: {name}")
            if not identity.get("source_sha256") or any(training["source_sha256"].get(key) != value
                    for key, value in identity["source_sha256"].items()):
                raise ValueError(f"fit source receipt differs from the training lock: {name}")
            checkpoint = run / "final.pt"
            if sha256(checkpoint) != receipt.get("checkpoint_sha256"):
                raise ValueError(f"final checkpoint hash mismatch: {name}")
            if (arm, seed) in recovered:
                recovered_fit = recovered[(arm, seed)]
                archive_identity = read_json(HERE / recovered_fit["failed_attempt"] / "IDENTITY.json")
                if (recovered_fit.get("checkpoint_sha256") != receipt["checkpoint_sha256"]
                        or archive_identity != identity):
                    raise ValueError(f"recovery changed initialization/configuration or selected another endpoint: {name}")
            count = 0
            with (run / "train.jsonl").open() as handle:
                for count, line in enumerate(handle, 1):
                    row = json.loads(line)
                    finite = all(math.isfinite(value) for value in row.values() if isinstance(value, float))
                    if (row.get("step") != count or not finite
                            or not isinstance(row.get("loss"), (int, float))
                            or not isinstance(row.get("gradient_norm"), (int, float))
                            or row.get("teacher_typed_execution") is not (count <= training["teacher_updates"])):
                        raise ValueError(f"nonfinite, incomplete or inconsistent training trace: {name} row {count}")
            if count != STEPS:
                raise ValueError(f"training trace has {count} updates, expected {STEPS}: {name}")
            for field in ("initial_tensor_sha256", "batch_trace_sha256", "final_model_identity"):
                if not isinstance(receipt.get(field), str) or len(receipt[field]) != 64:
                    raise ValueError(f"missing {field}: {name}")
            receipt_paths += [run / name for name in ("RESULT.json", "IDENTITY.json", "train.jsonl", "optimizer.pt")]
            pair.append(receipt)
            records.append({"name": name, "arm": arm, "seed": seed, "checkpoint": relative(checkpoint),
                            "checkpoint_sha256": receipt["checkpoint_sha256"],
                            "model_identity": receipt["final_model_identity"], "steps": count})
        for key in ("initial_tensor_sha256", "batch_trace_sha256", "train_families_sha256", "shakespeare_train_sha256"):
            if pair[0][key] != pair[1][key]:
                raise ValueError(f"matched-arm provenance differs for seed {seed}: {key}")
    return {"training_lock_sha256": sha256(training_path), "receipt_sha256": hashes(receipt_paths),
            "training_source_sha256": training["source_sha256"], "fits": records,
            "matched_initial_tensors_and_training_draws": True, "recovery": recovery}


def build_lock(*, workers=3, batch_families=4):
    """Complete the read-only preflight before creating the evaluation lock."""
    if type(workers) is not int or not 1 <= workers <= 6 or type(batch_families) is not int or batch_families < 1:
        raise ValueError("use one to six workers and a positive family batch size")
    refuse_existing_targets()
    completion = verify_completed_fits()
    evidence = HERE / "evidence"
    manifest_path = evidence / "data/DATA_MANIFEST.json"
    manifest = read_json(manifest_path)
    data_paths = [manifest_path, evidence / "data/corpus.json"]
    if sha256(data_paths[-1]) != manifest["corpus_sha256"]:
        raise ValueError("combined corpus differs from the pretraining manifest")
    for split in ("train", *PANELS):
        entry = manifest["split_files"][split]
        path = evidence / "data" / f"{split}.json"
        if entry["filename"] != path.name or sha256(path) != entry["sha256"]:
            raise ValueError("split differs from the pretraining manifest: " + split)
        data_paths.append(path)
    train = read_json(evidence / "data/train.json")["families"]
    probe = read_json(evidence / "data/train_probe.json")
    if (probe.get("source_split") != "train" or probe.get("source_indices") != list(range(128))
            or probe["families"] != train[:128] or len(probe["families"]) != 128):
        raise ValueError("TRAIN probe must be the fixed first 128 training families")
    legacy_lock_path = PARENT / "evidence/TRAINING_LOCK.json"
    legacy_lock = read_json(legacy_lock_path)
    legacy_corpus = PARENT / "evidence/corpus.json"
    if sha256(legacy_corpus) != legacy_lock["source_sha256"]["evidence/corpus.json"]:
        raise ValueError("legacy regression corpus differs from its original training lock")
    legacy_evaluation_lock_path = PARENT / "evidence/EVALUATION_LOCK.json"
    legacy_evaluation_lock = read_json(legacy_evaluation_lock_path)
    if sha256(PARENT / "fusion/evaluate.py") != legacy_evaluation_lock["source_sha256"]["fusion/evaluate.py"]:
        raise ValueError("legacy evaluator changed from the original evaluation lock")
    data_paths += [legacy_corpus, PARENT / "evidence/DATA_MANIFEST.json"]
    reused = []
    endpoints = [dict(entry) for entry in completion["fits"]]
    for seed in SEEDS:
        path = PARENT / "evidence/evaluation" / f"memory_seed{seed}" / "RESULT.json"
        receipt = read_json(path)
        checkpoint = parent_checkpoint(seed)
        if (receipt.get("checkpoint_sha256") != sha256(checkpoint)
                or receipt.get("corpus_sha256") != sha256(legacy_corpus)
                or set(receipt.get("panels", {})) != set(REGRESSION_PANELS)):
            raise ValueError(f"parent regression receipt has incompatible provenance: seed {seed}")
        reused.append({"seed": seed, "source": relative(path), "source_sha256": sha256(path)})
        endpoints.append({"name": f"parent_seed{seed}", "arm": "parent", "seed": seed,
                          "checkpoint": relative(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                          "model_identity": receipt["model_identity"], "steps": None})
    endpoints.sort(key=lambda entry: endpoint_names().index(entry["name"]))
    for entry in endpoints:
        entry.update(evaluation_dir=relative(evidence / "evaluation" / entry["name"]),
                     regression_dir=relative(evidence / "regression" / entry["name"]))
    return {"schema": LOCK_SCHEMA, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "completion": completion, "source_sha256": hashes(evaluation_sources()),
            "data_sha256": hashes(data_paths),
            "legacy_lock_sha256": hashes([legacy_lock_path, legacy_evaluation_lock_path]),
            "endpoints": endpoints, "reused_parent_regressions": reused,
            "panels": list(PANELS), "regression_panels": list(REGRESSION_PANELS),
            "batch_families": batch_families, "parallel_workers": workers, "threads_per_worker": 1,
            "policy": {"train_probe": "reset only for reset_training; continuous for continuous and parent",
                       "held": "continuous neural state for all arms and parents", "reader": "strict",
                       "extra_002_diagnostics": False, "regressions": "unchanged 001 evaluator and diagnostics",
                       "endpoint_selection": "all fixed final endpoints; no quality-dependent selection",
                       "overwrite_or_implicit_resume": False},
            "environment": {"python": sys.version, "platform": platform.platform(),
                            "torch": metadata.version("torch"), "numpy": metadata.version("numpy"),
                            "cpu_count": os.cpu_count(), "accelerator": "CPU",
                            "clock": "perf_counter/process_time; evaluation durations are not serving benchmarks"}}


def verify_evaluation_lock(path, expected_hash):
    if sha256(path) != expected_hash:
        raise ValueError("evaluation lock changed")
    lock = read_json(path)
    if lock.get("schema") != LOCK_SCHEMA or [entry["name"] for entry in lock["endpoints"]] != endpoint_names():
        raise ValueError("unexpected evaluation lock or endpoint inventory")
    for field in ("source_sha256", "data_sha256", "legacy_lock_sha256"):
        verify_hashes(lock[field], "evaluation input")
    completion = verify_completed_fits()
    if completion != lock["completion"]:
        raise ValueError("completed training provenance changed after evaluation lock")
    verify_hashes({entry["checkpoint"]: entry["checkpoint_sha256"] for entry in lock["endpoints"]}, "endpoint")
    verify_hashes({entry["source"]: entry["source_sha256"] for entry in lock["reused_parent_regressions"]}, "parent regression")
    return lock


def panel_policy(arm, panel):
    if arm not in (*ARMS, "parent") or panel not in PANELS:
        raise ValueError("unknown declared arm or panel")
    return {"reset_after_answer": arm == "reset_training" and panel == "train_probe",
            "memory_enabled": None, "reader": "strict"}


def score_endpoint(lock, endpoint, lock_hash, *, load_model=None, evaluate_new=None,
                   evaluate_legacy=None, legacy_interventions=None, measure_state_bytes=None, verify_inputs=None):
    """Load one model, score all panels, and write each complete artifact once.

    Optional callables are used only by fixture tests of orchestration. CLI workers
    always use the immutable real model and evaluator implementations.
    """
    if load_model is None:
        from evaluate import evaluate_panel as evaluate_new
        from fusion.evaluate import evaluate_panel as evaluate_legacy, fixed_state_interventions as legacy_interventions
        from fusion.model import FusionModel, state_bytes as measure_state_bytes
        load_model = FusionModel.from_checkpoint
    output = HERE.parent / endpoint["evaluation_dir"]
    regression = HERE.parent / endpoint["regression_dir"]
    output.mkdir(parents=True, exist_ok=True)
    regression.mkdir(parents=True, exist_ok=True)
    planned = [output / (panel + ".json") for panel in PANELS] + [output / "RESULT.json", regression / "RESULT.json"]
    planned += ([regression / "REUSED.json"] if endpoint["arm"] == "parent" else
                [regression / (panel + ".json") for panel in (*REGRESSION_PANELS, "interventions")])
    if any(path.exists() or path.is_symlink() for path in planned):
        raise FileExistsError("worker output already exists")
    write_new_json(output / "WORKER_STARTED.json", {"name": endpoint["name"], "pid": os.getpid(),
                   "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "evaluation_lock_sha256": lock_hash})
    wall_started, cpu_started = time.perf_counter(), time.process_time()
    model = load_model(HERE.parent / endpoint["checkpoint"]).eval()
    if model.model_identity() != endpoint["model_identity"] or not model.memory_enabled:
        raise ValueError("loaded model differs from the locked memory-enabled endpoint")
    shared = {"checkpoint_sha256": endpoint["checkpoint_sha256"], "model_identity": model.model_identity(),
              "parameters": model.parameter_summary(), "neural_state_tensor_bytes": measure_state_bytes(model.init_state(1)),
              "source_sha256": lock["source_sha256"], "evaluation_lock_sha256": lock_hash,
              "arm": endpoint["arm"], "seed": endpoint["seed"]}
    summaries = {}
    for panel in PANELS:
        path = HERE / "evidence/data" / (panel + ".json")
        families = read_json(path)["families"]
        observed = evaluate_new(model, families, batch_families=lock["batch_families"],
                                **panel_policy(endpoint["arm"], panel))
        observed.update(shared, schema="cog-fusion-002-evaluation-v1", panel=panel,
                        data_sha256=lock["data_sha256"][relative(path)], family_sha256=canonical_digest(families))
        write_new_json(output / (panel + ".json"), observed)
        summaries[panel] = observed["summary"]
        print(json.dumps({"endpoint": endpoint["name"], "panel": panel, "summary": observed["summary"]}), flush=True)
    if endpoint["arm"] == "parent":
        reused = next(entry for entry in lock["reused_parent_regressions"] if entry["seed"] == endpoint["seed"])
        source = HERE.parent / reused["source"]
        with (regression / "RESULT.json").open("xb") as handle:
            handle.write(source.read_bytes())
        write_new_json(regression / "REUSED.json", {"schema": "cog-fusion-002-reused-regression-v1", **reused,
                       "checkpoint_sha256": endpoint["checkpoint_sha256"], "evaluation_lock_sha256": lock_hash,
                       "copied_result_sha256": sha256(regression / "RESULT.json"),
                       "status": "byte-for-byte original 001 receipt; no new parent regression scoring"})
    else:
        corpus_path = PARENT / "evidence/corpus.json"
        corpus = read_json(corpus_path)
        legacy_result = {"checkpoint_sha256": endpoint["checkpoint_sha256"],
                         "corpus_sha256": lock["data_sha256"][relative(corpus_path)],
                         "model_identity": model.model_identity(), "parameters": model.parameter_summary(),
                         "state_tensor_bytes": shared["neural_state_tensor_bytes"], "panels": {},
                         "evaluation_lock_sha256": lock_hash, "source_sha256": lock["source_sha256"],
                         "arm": endpoint["arm"], "seed": endpoint["seed"],
                         "status": "exposed 001 regression; unchanged evaluator functions"}
        for panel in REGRESSION_PANELS:
            # Keep the inherited evaluator's original batch size and diagnostics.
            observed = evaluate_legacy(model, corpus[panel], batch_families=8, diagnostics=panel != "long")
            write_new_json(regression / (panel + ".json"), observed)
            legacy_result["panels"][panel] = observed["summary"]
        intervention = legacy_interventions(model, corpus["test"])
        write_new_json(regression / "interventions.json", intervention)
        legacy_result["interventions"] = intervention["summary"]
        write_new_json(regression / "RESULT.json", legacy_result)
    result = {**shared, "status": "completed", "panels": summaries,
              "panel_sha256": {panel: sha256(output / (panel + ".json")) for panel in PANELS},
              "regression_result_sha256": sha256(regression / "RESULT.json"),
              "wall_seconds": time.perf_counter() - wall_started, "cpu_seconds": time.process_time() - cpu_started,
              "timing_status": "evaluation workload; not an isolated serving benchmark"}
    if verify_inputs is not None:
        verify_inputs()
    write_new_json(output / "RESULT.json", result)
    return result


def worker(name, lock_hash):
    lock_path = HERE / "evidence/EVALUATION_LOCK.json"
    lock = verify_evaluation_lock(lock_path, lock_hash)
    endpoint = next(entry for entry in lock["endpoints"] if entry["name"] == name)
    # Import and load a candidate only after verifying all six completed fits.
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    score_endpoint(lock, endpoint, lock_hash, verify_inputs=lambda: verify_evaluation_lock(lock_path, lock_hash))


def launch(*, workers=3, batch_families=4, check_only=False):
    lock = build_lock(workers=workers, batch_families=batch_families)
    if check_only:
        print(json.dumps({"status": "ready; no output created", "endpoints": endpoint_names(), "panels": PANELS}), flush=True)
        return lock
    evidence = HERE / "evidence"
    lock_path = evidence / "EVALUATION_LOCK.json"
    write_new_json(lock_path, lock)
    lock_hash = sha256(lock_path)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    jobs = [{"name": endpoint["name"], "command": [sys.executable, str(HERE / "run_evaluations.py"),
             "--worker", endpoint["name"], "--lock-sha256", lock_hash]} for endpoint in lock["endpoints"]]
    write_new_json(evidence / "EVALUATION_JOBS.json", {"evaluation_lock_sha256": lock_hash, "jobs": jobs})
    pending, running, completed = list(jobs), [], []
    while pending or running:
        while pending and len(running) < workers:
            job = pending.pop(0)
            output = evidence / "evaluation" / job["name"]
            output.mkdir(parents=True, exist_ok=False)
            handle = (output / "console.log").open("x")
            process = subprocess.Popen(job["command"], cwd=HERE, env=env, stdout=handle, stderr=subprocess.STDOUT)
            running.append({**job, "process": process, "handle": handle})
            print(json.dumps({"started": job["name"], "pid": process.pid}), flush=True)
        for job in list(running):
            code = job["process"].poll()
            if code is None:
                continue
            job["handle"].close()
            completed.append({"name": job["name"], "pid": job["process"].pid, "returncode": code})
            running.remove(job)
            print(json.dumps(completed[-1]), flush=True)
        if running:
            time.sleep(1)
    verify_evaluation_lock(lock_path, lock_hash)
    completion = {"evaluation_lock_sha256": lock_hash, "workers": completed,
                  "status": "completed" if all(row["returncode"] == 0 for row in completed) else "failed"}
    write_new_json(evidence / "EVALUATION_EXIT.json", completion)
    if completion["status"] != "completed":
        raise RuntimeError("one or more evaluations failed; partial outputs retained without overwrite")
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3, help="independent single-thread endpoint workers, maximum six")
    parser.add_argument("--batch-families", type=int, default=4)
    parser.add_argument("--check-only", action="store_true", help="validate all inputs and output absence without scoring or writing")
    parser.add_argument("--worker", choices=endpoint_names(), help=argparse.SUPPRESS)
    parser.add_argument("--lock-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if not args.lock_sha256 or args.check_only:
            parser.error("a worker requires its launcher lock hash")
        worker(args.worker, args.lock_sha256)
    else:
        if args.lock_sha256:
            parser.error("--lock-sha256 is a worker-only argument")
        launch(workers=args.workers, batch_families=args.batch_families, check_only=args.check_only)


if __name__ == "__main__":
    main()
