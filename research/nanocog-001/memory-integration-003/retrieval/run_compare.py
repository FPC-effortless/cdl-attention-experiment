#!/usr/bin/env python3
"""Paired exact-ranking experiment on the same 50k-message single-scope DB."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_store(root, name):
    path = root / "nanocog/memory/store.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.MemoryStore


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def measure(call):
    begin = time.perf_counter_ns()
    value = call()
    return value, (time.perf_counter_ns() - begin) / 1e6


def stats(values):
    return {"median_ms": statistics.median(values), "minimum_ms": min(values),
            "maximum_ms": max(values), "samples_ms": values}


def content(i):
    base = (f"Synthetic note needle{i:05d}. This project keeps unrelated workplace observations "
            "and returns the correct record from the current archive without confusing other notes.")
    if i % 10 == 0:
        base += " Café Yorùbá e\u0301 中文 Ελληνικά Straße."
    if i % 101 == 0:
        base += " Orion launch Thursday."
    return base


QUERIES = (
    ("early-unique", "needle00001"), ("middle-unique", "needle25001"),
    ("late-unique", "needle49999"), ("all-records-ascii", "synthetic"),
    ("all-records-multi-term", "synthetic project archive"),
    ("selective-phrase", "Orion launch"), ("unicode-subset", "Yorùbá 中文 café"),
    ("mixed-broad", "cafe synthetic"), ("absent", "zzzznotpresentzzzz"),
)


def main(args):
    started, cpu = time.perf_counter(), time.process_time()
    baseline = load_store(args.baseline_root, "memory003_baseline_store")
    candidate = load_store(args.candidate_root, "memory003_candidate_store")
    before = {"baseline": digest(args.baseline_root / "nanocog/memory/store.py"),
              "candidate": digest(args.candidate_root / "nanocog/memory/store.py"),
              "harness": digest(__file__)}
    report = {"purpose": "exact ranked-result equivalence and descriptive single-scope cost",
              "models_or_training": False, "n_sources": 50000, "unicode_sources": 5000,
              "scopes": 1, "ordinary_sessions": 200, "source_sha256_before": before,
              "python": platform.python_version(), "warmups": 1, "measured_trials": 5,
              "queries": [], "first_queries": [], "passed": True}
    with tempfile.TemporaryDirectory(prefix="memory003-ranked-", dir=args.work_dir) as tmp:
        folder = Path(tmp)
        path = folder / "single-scope.sqlite3"
        report["filesystem"] = subprocess.check_output(["df", "-PT", str(folder)], text=True).strip()
        stream = hashlib.sha256()
        begin = time.perf_counter()
        with baseline(path, principal_id="synthetic", namespace="one-scope") as store:
            for i in range(50000):
                text = content(i)
                store.append_message(f"chat{i % 200}", "user", text, event_id=f"event{i:05d}")
                stream.update((canonical([i, text]) + "\n").encode())
                if i % 1000 == 0:
                    key = f"project.launch{i // 1000:02d}"
                    store.propose(key, f"Thursday needle{i:05d}", f"event{i:05d}", proposal_id=f"proposal{i}")
                    store.accept(f"proposal{i}", authority="synthetic-host")
            store.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            report["journal_mode"] = store.journal_mode
            report["sqlite"] = store.db.execute("SELECT sqlite_version()").fetchone()[0]
        report["fixture_ingestion_seconds"] = time.perf_counter() - begin
        report["input_sha256"] = stream.hexdigest()
        report["database_bytes"] = path.stat().st_size

        # Fresh SQLite connections; OS pages are not flushed. This is explicitly
        # not a cold physical-device measurement.
        for cls, label in ((baseline, "baseline"), (candidate, "candidate")):
            with cls(path, principal_id="synthetic", namespace="one-scope") as store:
                result, duration = measure(lambda: store.recall("synthetic", limit=8, max_chars=8000))
                report["first_queries"].append({"implementation": label, "query": "synthetic",
                                                "duration_ms": duration,
                                                "result_sha256": hashlib.sha256(canonical(result).encode()).hexdigest()})
        assert report["first_queries"][0]["result_sha256"] == report["first_queries"][1]["result_sha256"]
        with baseline(path, principal_id="synthetic", namespace="one-scope") as old, \
             candidate(path, principal_id="synthetic", namespace="one-scope") as new:
            stores = {"baseline": old, "candidate": new}
            for label, query in QUERIES:
                samples = {"baseline": [], "candidate": []}
                fingerprints = []
                for trial in range(6):
                    order = ("baseline", "candidate") if trial % 2 == 0 else ("candidate", "baseline")
                    values = {}
                    for arm in order:
                        values[arm], duration = measure(lambda: stores[arm].recall(query, limit=8, max_chars=8000))
                        if trial:
                            samples[arm].append(duration)
                    assert values["baseline"] == values["candidate"], (label, trial)
                    fingerprints.append(hashlib.sha256(canonical(values["baseline"]).encode()).hexdigest())
                assert len(set(fingerprints)) == 1, (label, "result changed without writes")
                report["queries"].append({"label": label, "query": query,
                                           "baseline": stats(samples["baseline"]),
                                           "candidate": stats(samples["candidate"]),
                                           "all_six_pairs_exact": True,
                                           "result_sha256": fingerprints[0]})
                print(json.dumps({"query": label, "exact": True,
                                  "baseline_ms": statistics.median(samples["baseline"]),
                                  "candidate_ms": statistics.median(samples["candidate"])}), flush=True)
                if time.process_time() - cpu > 300 or time.perf_counter() - started > 600:
                    raise RuntimeError("paired diagnostic exceeded its resource allocation")
            report["integrity"] = new.db.execute("PRAGMA integrity_check").fetchone()[0]
            assert report["integrity"] == "ok"
    after = {"baseline": digest(args.baseline_root / "nanocog/memory/store.py"),
             "candidate": digest(args.candidate_root / "nanocog/memory/store.py"),
             "harness": digest(__file__)}
    assert before == after, "source changed during run"
    report.update(source_sha256_after=after, wall_seconds=time.perf_counter() - started,
                  cpu_seconds=time.process_time() - cpu,
                  max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  scope="finite lexical behavior only; no semantic or lifetime qualification")
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "wall_seconds": report["wall_seconds"], "passed": True}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
