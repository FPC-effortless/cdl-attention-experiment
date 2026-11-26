"""Isolated CPU cost accounting after fixed training and evaluation complete.

The benchmark reports observed costs, not an efficiency win. Its short supplied
scenario is a timing fixture; the held-out qualification receipts remain the
quality evidence. Durable publication, dependency validation and answer-length
differences stay inside the measured serving path.
"""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import platform
import sqlite3
import statistics
import tempfile
import time

import numpy as np
import torch

from .model import FusionModel, state_bytes
from .preservation import completed_runs
from .session import Session
from .store import Operation, Store


HERE = Path(__file__).resolve().parents[1]
REPEATS = 7
PACKET_TRIALS = 3
PACKET_ITERATIONS = 200
OBSERVATIONS = ("Amara owns the chest.", "The coin is in the chest.",
                "Cora owns the crate.", "The chest belongs to Bela.")
QUESTION = "Who owns the container holding the coin?"
EXPECTED_ANSWER = "Bela."
STAGES = ("tokenization", "interpretation", "neural_decode", "neural_serialization",
          "rng_serialization", "resolution", "publication", "validation_within_publication",
          "record_update_within_publication")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values) -> dict:
    values = list(map(float, values))
    if not values or not all(np.isfinite(values)):
        raise ValueError("timing sample must contain finite observations")
    q1, q3 = np.percentile(values, [25, 75], method="linear").tolist()
    return {"n": len(values), "median": statistics.median(values), "q1": q1, "q3": q3,
            "iqr": q3 - q1, "min": min(values), "max": max(values), "samples": values}


def measured(function):
    wall, cpu = time.perf_counter(), time.process_time()
    value = function()
    return value, {"wall_seconds": time.perf_counter() - wall,
                   "cpu_seconds": time.process_time() - cpu}


@contextmanager
def timed_methods(targets):
    """Transparent same-process timing wrappers; original methods are restored.

    These wrappers do not change parameters, labels, inputs, outputs or state.
    Their small timing overhead is included in both serving arms. Nested stage
    times are explicitly labelled and must not be summed twice.
    """
    samples = {label: [] for _, _, label in targets}
    originals = []
    try:
        for obj, attribute, label in targets:
            original = getattr(obj, attribute)
            own_attribute = attribute in vars(obj)
            @wraps(original)
            def wrapper(*args, _original=original, _label=label, **kwargs):
                wall, cpu = time.perf_counter(), time.process_time()
                try:
                    return _original(*args, **kwargs)
                finally:
                    samples[_label].append({"wall_seconds": time.perf_counter() - wall,
                                            "cpu_seconds": time.process_time() - cpu})
            originals.append((obj, attribute, original, own_attribute))
            setattr(obj, attribute, wrapper)
        yield samples
    finally:
        for obj, attribute, original, own_attribute in reversed(originals):
            if own_attribute:
                setattr(obj, attribute, original)
            else:
                delattr(obj, attribute)


def instrument(session):
    return timed_methods([
        (session.model, "encode_texts", "tokenization"),
        (session.model, "encode_event", "interpretation"),
        (session.model, "decode", "neural_decode"),
        (session, "_state_blob", "neural_serialization"),
        (session, "_rng_blob", "rng_serialization"),
        (session.store, "resolve", "resolution"),
        (session.store, "commit_request", "publication"),
        (session.store, "_validate_packet", "validation_within_publication"),
        (session.store, "_write_record", "record_update_within_publication"),
    ])


def stage_totals(samples):
    return {label: {"calls": len(rows), "wall_seconds": sum(row["wall_seconds"] for row in rows),
                    "cpu_seconds": sum(row["cpu_seconds"] for row in rows)}
            for label, rows in samples.items()}


def stage_difference(after, before):
    return {label: {key: after[label][key] - before[label][key]
                    for key in ("calls", "wall_seconds", "cpu_seconds")} for label in after}


def database_sizes(store: Store) -> dict:
    paths = [Path(store.path + suffix) for suffix in ("", "-wal", "-shm", "-journal")]
    files = {path.name: path.stat().st_size for path in paths if path.exists()}
    return {"files": files, "total_bytes": sum(files.values()),
            "logical_page_bytes": store.db.execute("PRAGMA page_count").fetchone()[0] *
                                  store.db.execute("PRAGMA page_size").fetchone()[0],
            "current_records": store.db.execute("SELECT COUNT(*) FROM records WHERE present=1").fetchone()[0],
            "retained_events": store.db.execute("SELECT COUNT(*) FROM events").fetchone()[0],
            "retained_receipts": store.db.execute("SELECT COUNT(*) FROM receipts").fetchone()[0],
            "journal_mode": store.db.execute("PRAGMA journal_mode").fetchone()[0],
            "synchronous": store.db.execute("PRAGMA synchronous").fetchone()[0]}


def serving_trial(model: FusionModel, path: Path, *, repeat: int) -> dict:
    session, startup = measured(lambda: Session(model, path, scope="world", seed=0, max_chars=12))
    try:
        requests = []
        with instrument(session) as stages:
            serve_wall, serve_cpu = time.perf_counter(), time.process_time()
            for index, text in enumerate(OBSERVATIONS):
                before = stage_totals(stages)
                reply, cost = measured(lambda text=text, index=index:
                    session.observe(text, f"observation-{index}"))
                requests.append({"kind": "observation", "text": text, "cost": cost,
                                 "stages": stage_difference(stage_totals(stages), before), "reply": reply})
            before = stage_totals(stages)
            reply, cost = measured(lambda: session.ask(QUESTION, "question", temperature=0))
            requests.append({"kind": "question", "text": QUESTION, "cost": cost,
                             "stages": stage_difference(stage_totals(stages), before), "reply": reply})
            serving = {"wall_seconds": time.perf_counter() - serve_wall,
                       "cpu_seconds": time.process_time() - serve_cpu,
                       "stages": stage_totals(stages)}
        counters_before_restore = dict(session.counters)
        snapshot = session.store.snapshot_state()
        sizes = database_sizes(session.store)
        tensor_bytes = state_bytes(session.state)
        if tensor_bytes != 10592 or counters_before_restore["historical_tokens_replayed"] != 0:
            raise AssertionError("unexpected continuation size or historical token replay")
        if counters_before_restore["input_tokens"] != sum(len(text) + 1 for text in (*OBSERVATIONS, QUESTION)):
            raise AssertionError("new-input character accounting is inconsistent")
        store_counters = dict(session.store.counters)
    finally:
        _, close_cost = measured(session.close)
    resumed, restore_cost = measured(lambda: Session(model, path, scope="world", seed=0, max_chars=12))
    try:
        restored_snapshot = resumed.store.snapshot_state()
        coherent_restore = (dict(resumed.counters) == counters_before_restore and
                            restored_snapshot["neural_state"] == snapshot["neural_state"] and
                            restored_snapshot["rng_state"] == snapshot["rng_state"])
        if not coherent_restore:
            raise AssertionError("restore changed persisted counters or continuation bytes")
        with instrument(resumed) as restored_stages:
            continued, continuation_cost = measured(lambda: resumed.ask(QUESTION, "question-after-restart", temperature=0))
            continuation_cost["stages"] = stage_totals(restored_stages)
        counter_delta = {key: resumed.counters[key] - counters_before_restore[key] for key in resumed.counters}
        if counter_delta["input_tokens"] != len(QUESTION) + 1 or counter_delta["historical_tokens_replayed"] != 0:
            raise AssertionError("restart continuation processed historical input")
        after_sizes = database_sizes(resumed.store)
        return {"repeat": repeat, "session_startup": startup, "serving": serving,
                "requests": requests, "answer": reply["text"], "fixture_answer_exact": reply["text"] == EXPECTED_ANSWER,
                "counters_after_first_answer": counters_before_restore, "store_counters": store_counters,
                "neural_tensor_bytes": tensor_bytes, "neural_state_blob_bytes": len(snapshot["neural_state"]),
                "rng_blob_bytes": len(snapshot["rng_state"]), "database_after_first_answer": sizes,
                "close": close_cost, "restore": restore_cost, "coherent_restore": coherent_restore,
                "continuation": continuation_cost, "continuation_reply": continued,
                "continuation_counter_delta": counter_delta, "database_after_continuation": after_sizes}
    finally:
        resumed.close()


def summarize_serving(rows):
    result = {"session_startup_wall_seconds": distribution(row["session_startup"]["wall_seconds"] for row in rows),
              "four_observations_and_question_wall_seconds": distribution(row["serving"]["wall_seconds"] for row in rows),
              "four_observations_and_question_cpu_seconds": distribution(row["serving"]["cpu_seconds"] for row in rows),
              "new_question_wall_seconds": distribution(row["requests"][-1]["cost"]["wall_seconds"] for row in rows),
              "restore_wall_seconds": distribution(row["restore"]["wall_seconds"] for row in rows),
              "continuation_question_wall_seconds": distribution(row["continuation"]["wall_seconds"] for row in rows),
              "restore_and_continuation_wall_seconds": distribution(row["restore"]["wall_seconds"] +
                                                                     row["continuation"]["wall_seconds"] for row in rows),
              "fixture_correct_answers": sum(row["fixture_answer_exact"] for row in rows),
              "fixture_total": len(rows), "all_restores_coherent": all(row["coherent_restore"] for row in rows),
              "historical_input_tokens_replayed": sum(row["counters_after_first_answer"]["historical_tokens_replayed"] +
                                                        row["continuation_counter_delta"]["historical_tokens_replayed"] for row in rows),
              "stages": {}}
    for stage in STAGES:
        result["stages"][stage] = distribution(row["serving"]["stages"][stage]["wall_seconds"] for row in rows)
    return result


def serving_comparison(checkpoints: dict, workdir: Path) -> dict:
    models, loads, warmups, trials = {}, {}, {}, {"memory": [], "no_memory": []}
    for arm in ("memory", "no_memory"):
        model, first = measured(lambda arm=arm: FusionModel.from_checkpoint(checkpoints[arm]).eval())
        repeated_loads = []
        for _ in range(REPEATS):
            model, cost = measured(lambda arm=arm: FusionModel.from_checkpoint(checkpoints[arm]).eval())
            repeated_loads.append(cost)
        models[arm] = model
        loads[arm] = {"first_in_process": first,
                      "repeated_warm_load_wall_seconds": distribution(row["wall_seconds"] for row in repeated_loads),
                      "repeated_warm_load_cpu_seconds": distribution(row["cpu_seconds"] for row in repeated_loads),
                      "note": "OS file caches are not flushed; this is not a physical cold-disk claim"}
        warmups[arm] = serving_trial(model, workdir / f"warmup-{arm}.sqlite", repeat=-1)
    # Alternate arm order to reduce systematic order effects. Each trial uses
    # a new database and freshly initialized continuation, not repeated receipts.
    for repeat in range(REPEATS):
        order = ("memory", "no_memory") if repeat % 2 == 0 else ("no_memory", "memory")
        for arm in order:
            trials[arm].append(serving_trial(models[arm], workdir / f"{arm}-{repeat}.sqlite", repeat=repeat))
    summaries = {arm: summarize_serving(rows) for arm, rows in trials.items()}
    memory_cost = summaries["memory"]["four_observations_and_question_wall_seconds"]["median"]
    control_cost = summaries["no_memory"]["four_observations_and_question_wall_seconds"]["median"]
    return {"seed": 17, "repeats_per_arm": REPEATS, "observations": OBSERVATIONS, "question": QUESTION,
            "expected_fixture_answer": EXPECTED_ANSWER, "fixture_is_qualification": False,
            "model_load": loads, "warmup_excluded_from_measured_trials": warmups,
            "summaries": summaries, "trials": trials,
            "observed_memory_to_control_serving_ratio": memory_cost / control_cost,
            "stage_overlap": ["publication includes dependency validation and protected record updates",
                              "do not add *_within_publication times to publication a second time",
                              "outer serving time also includes checks, packing, Python control and timer-wrapper overhead"],
            "restart_scope": "close and reopen a session in the same process; process startup is not measured"}


def packet_comparison(record_count: int, path: Path) -> dict:
    store, initialize = measured(lambda: Store(path, "typed-packet-benchmark", capacity=record_count))
    try:
        records = [Operation("set", ("bench", f"record-{i}", "owner"), "Amara", "initial")
                   for i in range(record_count - 1)]
        records.append(Operation("link", ("bench", "target", "container"), "record-0", "initial"))
        _, index_fill = measured(lambda: store.commit_request("initialize", {"records": record_count}, operations=records))
        key = ("bench", "target", "container")
        packet, packet_construction = measured(lambda: store.resolve(key, "owner"))
        if packet.value != "Amara" or packet.read_count != 2:
            raise AssertionError("packet fixture did not establish the intended two-hop result")
        for _ in range(10):
            store.resolve(key, "owner")
            if not store.validate_packet(packet):
                raise AssertionError("fresh packet failed validation")
        trials = []
        for trial in range(PACKET_TRIALS):
            order = ("reuse_with_validation", "repeated_resolve") if trial % 2 == 0 else ("repeated_resolve", "reuse_with_validation")
            for mode in order:
                before = dict(store.counters)
                def work():
                    for _ in range(PACKET_ITERATIONS):
                        if mode == "reuse_with_validation":
                            if not store.validate_packet(packet):
                                raise AssertionError("cached packet invalidated unexpectedly")
                            value = packet.value
                        else:
                            value = store.resolve(key, "owner").value
                        if value != "Amara":
                            raise AssertionError("packet policies disagreed on the factual result")
                _, cost = measured(work)
                trials.append({"trial": trial, "mode": mode, "iterations": PACKET_ITERATIONS,
                               **cost, "wall_seconds_per_use": cost["wall_seconds"] / PACKET_ITERATIONS,
                               "counter_delta": {name: store.counters[name] - before[name] for name in before}})
        summaries = {mode: distribution(row["wall_seconds_per_use"] for row in trials if row["mode"] == mode)
                     for mode in ("reuse_with_validation", "repeated_resolve")}
        return {"current_records": record_count, "database_initialization": initialize,
                "record_and_index_population": index_fill, "packet_construction": packet_construction,
                "index_policy": "SQLite exact logical-key primary index; all population writes charged separately",
                "hops": 2, "trials": trials, "summaries_seconds_per_use": summaries,
                "reuse_to_resolve_ratio": summaries["reuse_with_validation"]["median"] / summaries["repeated_resolve"]["median"],
                "all_values_identical": True, "database": database_sizes(store),
                "limits": ["supplied typed records, not learned query or language performance",
                           "unchanged dependencies during each trial; invalidation correctness has separate tests",
                           "validation reconstructs dependencies in this implementation and may cost more than resolution"]}
    finally:
        store.close()


def storage_retention(path: Path) -> dict:
    """Same four-entry capacity, supplied typed writes, no model inference."""
    keys = [("retention", entity, "owner") for entity in ("chest", "crate", "box", "vault")]
    expected = dict(zip(keys, ("Amara", "Bela", "Cora", "Davi")))
    recent = deque(maxlen=4)
    with Store(path, "typed-storage-policy", capacity=4) as store:
        initial = [Operation("set", key, value, "initial") for key, value in expected.items()]
        store.commit_request("initial", {"count": 4}, operations=initial)
        recent.extend((operation.key, operation.value) for operation in initial)
        operations = []
        for index in range(64):
            key = keys[1 + index % 3]
            value = ("Eren", "Faye", "Gita", "Hadi")[index % 4]
            operations.append(Operation("set", key, value, f"overwrite-{index}"))
            expected[key] = value
            recent.append((key, value))
        # Publication is grouped because this is a storage-policy fixture,
        # not a per-event persistence timing comparison.
        store.commit_request("overwrites", {"count": 64}, operations=operations)
        protected = store.current_records()
        recent_current = dict(recent)
        target = keys[0]
        return {"qualification": "supplied typed storage-policy fixture; no learned semantic capability claim",
                "initial_unique_keys": 4, "unrelated_overwrite_events": 64, "capacity_entries_each": 4,
                "protected_current_entries": len(protected), "recent_event_entries": len(recent),
                "recent_distinct_current_keys": len(recent_current),
                "target_key": target, "target_expected": expected[target],
                "protected_target_retained": protected.get(target) == expected[target],
                "recent_target_retained": recent_current.get(target) == expected[target],
                "protected_correct_current_facts": sum(protected.get(key) == value for key, value in expected.items()),
                "recent_correct_current_facts": sum(recent_current.get(key) == value for key, value in expected.items()),
                "queried_current_facts": 4, "database": database_sizes(store),
                "limits": ["current-record capacity is fixed; durable event and receipt storage still grows",
                           "recent-event cache spends entries on repeated versions of the same identities"]}


def completed_evaluations(root: Path, fits: list[dict]) -> dict:
    output = {}
    for fit in fits:
        path = root / fit["name"] / "RESULT.json"
        if not path.is_file():
            raise RuntimeError(f"evaluation incomplete: {fit['name']}")
        result = json.loads(path.read_text())
        if result["checkpoint_sha256"] != fit["checkpoint_sha256"]:
            raise ValueError("evaluation checkpoint identity differs from completed training")
        panels = {}
        for panel in ("test", "fresh", "long"):
            if panel not in result["panels"]:
                raise RuntimeError(f"evaluation panel missing: {fit['name']}/{panel}")
            summary = result["panels"][panel]
            strict = summary["modes"]["strict"]
            panels[panel] = {"answer_accuracy": strict["exact_answers"] / summary["worlds"],
                             "complete_pair_accuracy": strict["complete_pairs"] / summary["families"],
                             "whole_history_pair_accuracy": strict["strict_whole_history_pairs"] / summary["families"]}
        primary_pass = all(value >= .95 for panel in ("test", "fresh") for value in panels[panel].values())
        output[fit["name"]] = {"evaluation_receipt_sha256": sha256(path), "panels": panels,
                               "primary_accuracy_thresholds_pass": primary_pass}
    return output


def environment() -> dict:
    cpu_model = None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.partition(":")[2].strip()
                break
    limits = {}
    for name in ("cpu.max", "memory.max"):
        path = Path("/sys/fs/cgroup") / name
        if path.exists():
            limits[name] = path.read_text().strip()
    clock = time.get_clock_info("perf_counter")
    return {"time_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
            "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "sqlite": sqlite3.sqlite_version, "cpu_model": cpu_model, "os_cpu_count": os.cpu_count(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "cgroup_limits": limits, "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(), "device": "cpu",
            "clock": {"implementation": clock.implementation, "monotonic": clock.monotonic,
                      "adjustable": clock.adjustable, "resolution_seconds": clock.resolution}}


def run(args):
    fits = completed_runs(args.runs)
    evaluation = completed_evaluations(args.evaluation_root, fits)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.workdir is None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        workdir = Path(tempfile.mkdtemp(prefix="benchmark-work-", dir=args.output.parent))
    else:
        workdir = args.workdir
        workdir.mkdir(parents=True, exist_ok=False)
    checkpoints = {arm: args.runs / f"{arm}_seed17" / "final.pt" for arm in ("memory", "no_memory")}
    serving = serving_comparison(checkpoints, workdir)
    packets = [packet_comparison(size, workdir / f"packets-{size}.sqlite") for size in (64, 1024)]
    retention = storage_retention(workdir / "retention.sqlite")
    sources = sorted((Path(__file__).parent).rglob("*.py"))
    output = {"schema": "cog-fusion-cost-v1", "environment": environment(),
              "source_sha256": {str(path.relative_to(HERE)): sha256(path) for path in sources},
              "checkpoint_sha256": {arm: sha256(path) for arm, path in checkpoints.items()},
              "quality_context": evaluation,
              "efficiency_superiority_admitted": False,
              "claim_policy": "diagnostic costs only; a failed quality gate cannot earn an efficiency win, and this fixture does not compare against attention baselines",
              "serving": serving, "packet_reuse": packets, "typed_storage_retention": retention,
              "work_directory": str(workdir),
              "limits": ["single host CPU implementation; seven trials do not establish hardware-independent performance",
                         "both arms retain the same memory infrastructure even when the evidence bridge is disabled",
                         "different answer lengths and correctness are reported and can explain cost differences",
                         "timing wrappers add small overhead to both arms",
                         "warmup is explicit; OS file caches are not forcibly cleared",
                         "restart cost excludes interpreter/process startup"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output),
                      "serving_memory_to_control_ratio": serving["observed_memory_to_control_serving_ratio"],
                      "packet_reuse_to_resolve_ratios": {row["current_records"]: row["reuse_to_resolve_ratio"] for row in packets},
                      "efficiency_superiority_admitted": False}), flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=HERE / "evidence" / "runs")
    parser.add_argument("--evaluation-root", type=Path, default=HERE / "evidence" / "evaluation")
    parser.add_argument("--output", type=Path, default=HERE / "evidence" / "COST_RESULTS.json")
    parser.add_argument("--workdir", type=Path)
    run(parser.parse_args())
