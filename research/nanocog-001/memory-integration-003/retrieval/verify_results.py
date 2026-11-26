#!/usr/bin/env python3
"""Fail closed on incomplete retrieval evidence; summarize recorded measurements."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent
RUNTIME = HERE.parents[1]
BASELINE_SHA = "3bf7561d194e19cc32a6ab4377ca4656aea86800ce6d026fd0a291ab18ec47c1"
CANDIDATE_SHA = "3e701337c26bfad404bb98c882bfe0638efb88b826576b9725e6be715c428f3c"
FINAL_SHA = "89dc05ae971c52b73b65bf37df1349ef16473528d1f4a8c695a4c891f0ea6085"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(name):
    return json.loads((HERE / name).read_text())


def lexical_parts(path):
    tree = ast.parse(Path(path).read_text())
    parts = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {
                "_word_tokens", "_rank_terms", "_lexical_score"}:
            parts[node.name] = ast.dump(node)
        if isinstance(node, ast.Assign) and any(isinstance(x, ast.Name) and
                x.id == "_WORD_TOKEN_RE" for x in node.targets):
            parts["_WORD_TOKEN_RE"] = ast.dump(node)
        if isinstance(node, ast.ClassDef) and node.name == "MemoryStore":
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name in {"messages", "recall"}:
                    parts[method.name] = ast.dump(method)
    assert len(parts) == 6
    return parts


def verify_scale(report, store_sha):
    assert not report.get("execution_error"), report.get("execution_error")
    assert report["skipped_stages"] == []
    assert report["source_changed_during_run"] is False
    assert report["source_hashes_start"] == report["source_hashes_end"]
    assert report["source_hashes_start"]["store.py"] == store_sha
    assert report["harness_sha256"] == digest(RUNTIME / "memory-validation-002/scale/run_scale.py")
    assert report["protocol_sha256"] == digest(RUNTIME / "memory-validation-002/scale/PROTOCOL.md")
    assert [s["n_sources"] for s in report["stages"]] == [1000, 10000, 50000]
    checks = 0
    stages = []
    for stage, required in zip(report["stages"], (294, 1543, 3022)):
        probes = [stage["same_process"], stage["fresh_process"]["result"],
                  stage["restored_process"]["result"]]
        for probe in probes:
            assert probe["checks"] == required
            assert probe["failures"] == []
            assert probe["source_hashes"] == report["source_hashes_start"]
            assert probe["exact_source_hits"] == probe["exact_source_total"] == 48
            assert probe["current_fact_hits"] == probe["current_fact_total"]
            assert probe["scope_isolation_probes"] == 24
            assert probe["lexical_control"] == {"car": True, "automobile": False}
            assert all(x["actual"] <= x["cap"] for x in probe["context_sizes"])
            checks += probe["checks"]
        assert stage["backup"]["schema_version"] == stage["restore"]["schema_version"] == 2
        assert stage["backup"]["sha256"] == stage["restore"]["sha256"]
        stages.append({
            "n_sources": stage["n_sources"], "n_per_scope": stage["n_sources"] // 8,
            "checks": required * 3,
            "broad_8000_median_ms_by_process": [statistics.median(
                x["cost"]["wall_s"] * 1000 for x in probe["context_sizes"] if x["cap"] == 8000)
                for probe in probes],
            "source_recall_median_ms": probes[0]["source_recall_latency"]["median_ms"],
            "fact_recall_median_ms": probes[0]["fact_recall_latency"]["median_ms"],
            "ingestion_cumulative": stage["ingest_cumulative"],
            "database_bytes": stage["sizes_after_checkpoint"]["main_bytes"],
            "backup_wall_seconds": stage["backup_cost"]["wall_s"],
            "restore_wall_seconds": stage["restore_cost"]["wall_s"],
        })
    for name, required in (("cli", 43), ("temporal", 10)):
        assert report[name]["checks"] == required
        assert report[name]["failures"] == []
        checks += required
    assert checks == 14630
    return {"checks": checks, "failures": 0, "stages": stages, "cost": report["total"]}


def main(args):
    baseline, candidate = read("baseline-scale.json"), read("candidate-scale.json")
    paired = read("paired-single-scope.json")
    before = baseline["source_hashes_start"]
    after = candidate["source_hashes_start"]
    assert set(before) == set(after)
    assert [key for key in before if before[key] != after[key]] == ["store.py"]
    for old, new in zip(baseline["stages"], candidate["stages"]):
        assert old["input_sha256"] == new["input_sha256"]
        assert old["manifest_sha256"] == new["manifest_sha256"]
        assert old["sizes_after_checkpoint"] == new["sizes_after_checkpoint"]
    assert paired["passed"] is True and paired["integrity"] == "ok"
    assert paired["source_sha256_before"] == paired["source_sha256_after"]
    assert paired["source_sha256_before"] == {
        "baseline": BASELINE_SHA, "candidate": CANDIDATE_SHA,
        "harness": digest(HERE / "run_compare.py")}
    assert len(paired["queries"]) == 9
    assert all(q["all_six_pairs_exact"] for q in paired["queries"])
    assert paired["first_queries"][0]["result_sha256"] == paired["first_queries"][1]["result_sha256"]
    candidate_path = args.candidate_root / "nanocog/memory/store.py"
    final_path = RUNTIME / "nanocog/memory/store.py"
    assert digest(candidate_path) == CANDIDATE_SHA
    assert digest(final_path) == FINAL_SHA
    assert lexical_parts(candidate_path) == lexical_parts(final_path)
    result = {
        "passed": True,
        "baseline": verify_scale(baseline, BASELINE_SHA),
        "retrieval_candidate": verify_scale(candidate, CANDIDATE_SHA),
        "paired_query_comparisons": 54, "paired_query_mismatches": 0,
        "paired_same_database": {"n_sources": paired["n_sources"], "scopes": paired["scopes"],
            "unicode_sources": paired["unicode_sources"], "database_bytes": paired["database_bytes"],
            "queries": paired["queries"], "first_queries": paired["first_queries"],
            "wall_seconds": paired["wall_seconds"], "cpu_seconds": paired["cpu_seconds"]},
        "source_sha256": {"baseline": BASELINE_SHA, "timed_retrieval_candidate": CANDIDATE_SHA,
                          "final_combined_store": FINAL_SHA},
        "final_combined_lexical_paths_match_timed_candidate": True,
        "interpretation": "Exact lexical serving improvement on these finite ASCII-heavy workloads. "
            "No model accuracy, universal Unicode speedup, constant-time scaling or lifetime guarantee.",
    }
    result["measured_total_cpu_seconds"] = sum(x["total"]["parent_cpu_s"] + x["total"]["child_cpu_s"]
                                               for x in (baseline, candidate)) + paired["cpu_seconds"]
    result["measured_total_wall_seconds"] = sum(x["total"]["wall_s"] for x in (baseline, candidate)) + paired["wall_seconds"]
    assert result["measured_total_cpu_seconds"] <= 600
    assert result["measured_total_wall_seconds"] <= 1200
    result["evidence_sha256"] = {name: digest(HERE / name) for name in (
        "baseline-scale.json", "candidate-scale.json", "paired-single-scope.json", "run_compare.py",
        "retrieval-only.patch", "PROTOCOL.md", "environment.json")}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": True, "scale_checks_each": 14630, "paired_exact_comparisons": 54,
                      "output": str(args.output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "verified-summary.json")
    main(parser.parse_args())
