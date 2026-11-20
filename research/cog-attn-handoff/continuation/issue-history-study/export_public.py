"""Export source, non-text predictions and public issue provenance for GitHub."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n")


def main():
    destination = Path(sys.argv[1]).resolve()
    destination.mkdir(parents=True)
    source_files = (
        "README.md", "PROTOCOL.md", "REPRODUCE.md", "NLI_SHA256SUMS",
        "acquire.py", "prepare.py", "launch.py", "audit.py", "check_snapshots.py",
        "export_public.py", "lab-protocol.json", "audit.json", "snapshot-crosscheck.json",
        "recipe/design.json", "recipe/entry.py", "recipe/models.py", "recipe/evaluation.py",
        "recipe/semantic.py", "recipe/state_store.py", "recipe/wr006.py",
        "recipe/frozen-plan.sha256", "recipe/semantic-timing.json", "recipe/cohort-counts.json",
        "recipe/candidate.json", "recipe/control.json", "recipe/semantic-recipe.json",
    )
    for filename in source_files:
        path = destination / filename
        path.parent.mkdir(exist_ok=True)
        shutil.copyfile(ROOT / filename, path)
    manifest = read(ROOT / "recipe/source-manifest.json")
    for cohort in manifest["cohorts"]:
        for issue in cohort["issues"]:
            issue.pop("reporter")
    manifest["local_manifest_sha256"] = hashlib.sha256(
        (ROOT / "recipe/source-manifest.json").read_bytes()).hexdigest()
    write(destination / "selected-issues.json", manifest)
    status = read(ROOT / "lab-status.json")
    public_runs = destination / "runs"
    public_runs.mkdir()
    summary = []
    for task in status["tasks"]:
        if task["role"] != "experiment":
            continue
        run = task["result"]["run_id"]
        source = ROOT / "ecosystem" / "runs" / run
        report = read(source / "output/results.json")
        record = read(source / "run.json")
        name = f"{report['arm']}-{report['seed']}"
        write(public_runs / f"{name}.json", report)
        predictions = (source / "output/predictions.json").read_bytes()
        (public_runs / f"{name}-predictions.json.gz").write_bytes(gzip.compress(predictions, mtime=0))
        summary.append({key: record[key] for key in
                        ("id", "seed", "name", "source_kind", "execution_purpose", "status",
                         "split_identity", "elapsed_seconds", "metrics", "inputs", "artifacts")})
    replay = next(task for task in status["tasks"] if task["role"] == "replication")
    semantic = read(ROOT / "semantic-receipt.json")
    write(destination / "run-index.json", {
        "primary": summary, "replay": replay["result"],
        "semantic_checkpoint_inputs": semantic["inputs"],
        "semantic_run": {"id": semantic["id"], "metrics": semantic["metrics"],
                         "artifacts": semantic["artifacts"], "elapsed_seconds": semantic["elapsed_seconds"]},
    })
    paths = sorted(p for p in destination.rglob("*") if p.is_file())
    (destination / "SHA256SUMS").write_text("".join(
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(destination)}\n" for p in paths
    ))


if __name__ == "__main__":
    main()
