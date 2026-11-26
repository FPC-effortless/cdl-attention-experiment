"""Run the independent decoder adversaries and retain per-case evidence."""
import argparse
import hashlib
import io
import json
import platform
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from nanocog.memory import test_adversarial_decoder


def hashes():
    paths = ("nanocog/memory/decoder.py", "nanocog/memory/codec.py", "nanocog/memory/store.py",
             "nanocog/model.py", "nanocog/interaction.py", "nanocog/comm/packet.py",
             "nanocog/state/records.py", "nanocog/memory/test_adversarial_decoder.py")
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths}


class Result(unittest.TextTestResult):
    def startTest(self, test):
        self.started = time.perf_counter()
        super().startTest(test)

    def record(self, test, status, err=None):
        self.rows.append(dict(test=test.id(), status=status,
                              seconds=time.perf_counter() - self.started,
                              traceback=None if err is None else self._exc_info_to_string(err, test)))

    def addSuccess(self, test):
        self.record(test, "passed")
        super().addSuccess(test)

    def addFailure(self, test, err):
        self.record(test, "failed", err)
        super().addFailure(test, err)

    def addError(self, test, err):
        self.record(test, "error", err)
        super().addError(test, err)

    def addSubTest(self, test, subtest, err):
        if err is not None:
            self.record(subtest, "failed-subtest", err)
        super().addSubTest(test, subtest, err)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    if not args.label.replace("-", "").replace("_", "").isalnum():
        raise ValueError("label must be a simple file stem")
    torch.set_num_threads(1)
    evidence = dict(label=args.label, fixture="seeded untrained reference models; no fitting",
                    python=sys.version, torch=str(torch.__version__), sqlite=sqlite3.sqlite_version,
                    platform=platform.platform(), threads=torch.get_num_threads(),
                    git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    source_sha256_before=hashes())
    output = io.StringIO()
    Result.rows = []
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_adversarial_decoder)
    result = unittest.TextTestRunner(stream=output, verbosity=2, resultclass=Result).run(suite)
    evidence.update(tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                    success=result.wasSuccessful(), seconds=time.perf_counter() - started,
                    cases=result.rows, source_sha256_after=hashes())
    folder = Path(__file__).resolve().parent
    (folder / f"{args.label}.json").write_text(json.dumps(evidence, indent=2) + "\n")
    (folder / f"{args.label}.txt").write_text(output.getvalue())
    print(json.dumps({k: evidence[k] for k in ("label", "tests_run", "failures", "errors", "success", "seconds")}))
    for row in evidence["cases"]:
        if row["status"] != "passed":
            print(row["test"] + ": " + row["status"])
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
