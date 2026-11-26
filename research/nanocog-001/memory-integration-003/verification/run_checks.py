#!/usr/bin/env python3
"""Source-matched final package/system gate, with optional private CLI smoke."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sqlite3
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def hashes():
    paths = sorted((ROOT / "nanocog").rglob("*.py"))
    paths += sorted((ROOT / "memory-integration-003").rglob("*.py"))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def command(args, env, name, timeout=240):
    tick = time.perf_counter()
    result = subprocess.run(args, cwd=ROOT, env=env, text=True, capture_output=True, timeout=timeout)
    (HERE / f"{name}.txt").write_text(result.stdout + result.stderr)
    return {"command": args, "exit_code": result.returncode, "wall_seconds": time.perf_counter() - tick}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pulse-source", type=Path)
    p.add_argument("--kit", type=Path)
    p.add_argument("--checkpoint", type=Path)
    args = p.parse_args()
    if (args.kit is None) != (args.checkpoint is None):
        p.error("kit and checkpoint must be supplied together")
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    if args.pulse_source:
        env["PULSE006_SOURCE"] = str(args.pulse_source.resolve())
    before = hashes()
    report = {"python": platform.python_version(), "sqlite": sqlite3.sqlite_version,
              "source_sha256_before": before,
              "optional_pulse_source_enabled": bool(args.pulse_source),
              "test_scope": "infrastructure/reference tests; learned qualification is a separate gate"}
    report["package"] = command([sys.executable, "-m", "pytest", "-q", "nanocog",
                                  "--junitxml=" + str(HERE / "full-suite.xml")], env, "full-suite")
    tree = ET.parse(HERE / "full-suite.xml")
    suites = list(tree.getroot()) if tree.getroot().tag == "testsuites" else [tree.getroot()]
    report["package"]["counts"] = {k: sum(int(s.attrib.get(k, 0)) for s in suites)
                                    for k in ("tests", "failures", "errors", "skipped")}
    report["system"] = command([sys.executable, "memory-validation-002/storage/run_adversarial.py",
                                 "--output", str(HERE / "system-results.json")], env, "system")
    system = json.loads((HERE / "system-results.json").read_text())
    report["system"]["counts"] = system["counts"]
    report["system"]["source_unchanged"] = system["source_unchanged"]
    report["independent_review"] = command([sys.executable, "-m", "pytest", "-q",
                                            "memory-integration-003/review/test_adapter_review.py"],
                                           env, "independent-review")
    if args.kit:
        cli = ROOT / "memory-integration-003/learned/factual_cli.py"
        with tempfile.TemporaryDirectory(prefix="memory003-cli-") as temp:
            common = [sys.executable, str(cli), "--kit", str(args.kit.resolve()),
                      "--checkpoint", str(args.checkpoint.resolve()),
                      "--db", str(Path(temp) / "persistent.sqlite3"), "--principal", "cli-check"]
            cases = [
                ("write", ["--session", "one", "--request-id", "first", "--event", "Amara holds the chest."], None),
                ("read-new-process", ["--session", "two", "--request-id", "ask-one", "--question", "Who holds the chest?"], "Amara"),
                ("correct", ["--session", "three", "--request-id", "second", "--event", "Amara gave the chest to Gus."], None),
                ("read-correction", ["--session", "four", "--request-id", "ask-two", "--question", "Who holds the chest?"], "Gus"),
            ]
            rows = []
            for name, extra, answer in cases:
                run = command(common + extra, env, "cli-" + name, timeout=90)
                payload = json.loads((HERE / ("cli-" + name + ".txt")).read_text()) if run["exit_code"] == 0 else None
                passed = run["exit_code"] == 0 and payload is not None
                if passed and answer is not None:
                    passed = payload["result"]["answer"] == answer
                elif passed:
                    passed = payload["result"]["status"] == "written" and payload["result"]["fact"] is not None
                rows.append({"name": name, "passed": passed, "execution": run, "result": payload})
            report["private_fitted_cli"] = {"passed": sum(r["passed"] for r in rows), "total": len(rows), "cases": rows}
    after = hashes()
    report["source_sha256_after"] = after
    report["source_unchanged"] = before == after
    package_counts = report["package"]["counts"]
    report["passed"] = bool(
        report["source_unchanged"] and report["package"]["exit_code"] == 0 and
        package_counts["failures"] == package_counts["errors"] == 0 and
        report["system"]["exit_code"] == 0 and system["counts"] == {"PASS": 9, "FAIL": 0, "ERROR": 0} and
        system["source_unchanged"] and report["independent_review"]["exit_code"] == 0 and
        (not args.kit or report["private_fitted_cli"]["passed"] == report["private_fitted_cli"]["total"]))
    (HERE / "RESULTS.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": report["passed"], "package": package_counts,
                      "system": system["counts"], "source_unchanged": report["source_unchanged"],
                      "private_fitted_cli": report.get("private_fitted_cli", {}).get("passed")}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
