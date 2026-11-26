"""Freeze the current Python package so concurrent edits cannot mix workers."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

with tempfile.TemporaryDirectory(prefix="nanocog-source-snapshot-") as name:
    snapshot = Path(name) / "research" / "nanocog-001"
    script_dir = snapshot / "memory-validation-002" / "decoder"
    script_dir.mkdir(parents=True)
    shutil.copytree(ROOT / "nanocog", snapshot / "nanocog",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    script = script_dir / "system_exercise.py"
    shutil.copyfile(HERE / script.name, script)
    result = subprocess.run([sys.executable, str(script)], check=False)
    report = json.loads((script_dir / "system_results.json").read_text())
    report["frozen_source_snapshot"] = True
    report["accepted_as_validation"] = (report["source_sha256_before"] == report["source_sha256_after"])
    (HERE / "system_results.json").write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(result.returncode)
