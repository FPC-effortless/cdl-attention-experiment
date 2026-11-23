#!/usr/bin/env python3
"""Check the abstract Lean source and retain an exact verification receipt.

Usage: python verify.py --lean /absolute/path/to/lean
This does not download dependencies, train a model, or verify Python refinement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lean", default="lean")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    source = root / "UnifiedLaws.lean"
    text = source.read_text(encoding="utf-8")
    # Direct proof placeholders and custom axioms are forbidden in this source.
    forbidden = re.findall(r"\b(?:sorry|admit|sorryAx)\b|^\s*axiom\s+", text, re.M)
    if forbidden:
        raise SystemExit(f"Forbidden proof token(s): {forbidden}")
    names = re.findall(r"^theorem\s+(\w+)", text, re.M)
    if not names or len(names) != len(set(names)):
        raise SystemExit("Empty or duplicate theorem inventory")

    started = time.time()
    version = subprocess.run([args.lean, "--version"], cwd=root,
                             text=True, capture_output=True, timeout=20)
    if version.returncode or "version 4.19.0," not in version.stdout:
        raise SystemExit("Expected Lean 4.19.0: " + version.stdout + version.stderr)

    compile_result = subprocess.run(
        [args.lean, "-o", "UnifiedLaws.olean", "UnifiedLaws.lean"],
        cwd=root, text=True, capture_output=True, timeout=60)
    (root / "lean-check.log").write_text(
        version.stdout + compile_result.stdout + compile_result.stderr,
        encoding="utf-8")
    if compile_result.returncode:
        raise SystemExit("Lean compilation failed; see lean-check.log")

    (root / "AxiomAudit.lean").write_text(
        "import UnifiedLaws\n\n" + "\n".join(
            "#print axioms COGUnified." + name for name in names) + "\n",
        encoding="utf-8")
    env = os.environ.copy()
    env["LEAN_PATH"] = str(root)
    audit_result = subprocess.run(
        [args.lean, "AxiomAudit.lean"], cwd=root, env=env,
        text=True, capture_output=True, timeout=60)
    audit = audit_result.stdout + audit_result.stderr
    (root / "axiom-audit.log").write_text(audit, encoding="utf-8")
    if audit_result.returncode:
        raise SystemExit("Axiom audit failed; see axiom-audit.log")
    checked = re.findall(r"^'COGUnified\.(\w+)'", audit, re.M)
    if checked != names:
        raise SystemExit("Axiom audit did not cover exact theorem inventory")
    used = set()
    for group in re.findall(r"depends on axioms: \[([^]]*)\]", audit):
        used.update(name.strip() for name in group.split(",") if name.strip())
    allowed = {"propext", "Quot.sound", "Classical.choice"}
    if used - allowed:
        raise SystemExit("Unexpected axioms: " + repr(sorted(used - allowed)))

    receipt = {
        "status": "CHECKED_ABSTRACT_LEAN_CONTRACTS_ONLY",
        "compiler": version.stdout.strip(),
        "source_sha256": digest(source),
        "theorem_count": len(names),
        "theorems": names,
        "compile_returncode": compile_result.returncode,
        "axiom_audit_returncode": audit_result.returncode,
        "axioms_used": sorted(used),
        "custom_axioms": [],
        "proof_placeholders": [],
        "lean_check_log_sha256": digest(root / "lean-check.log"),
        "axiom_audit_log_sha256": digest(root / "axiom-audit.log"),
        "elapsed_seconds": time.time() - started,
        "started_unix": started,
        "not_verified": [
            "Python or PyTorch refinement", "learned binding or generalization",
            "optimizer convergence", "floating-point implementation equivalence",
            "universal no-forgetting", "trained model quality or efficiency",
            "atomicity/serializability of a deployed storage engine",
            "complete runtime dependency discovery",
            "generic finite-capacity counting bound (hand proof only)"
        ]
    }
    (root / "verification.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: receipt[key] for key in (
        "status", "compiler", "source_sha256", "theorem_count",
        "compile_returncode", "axiom_audit_returncode", "axioms_used")}, indent=2))


if __name__ == "__main__":
    main()
