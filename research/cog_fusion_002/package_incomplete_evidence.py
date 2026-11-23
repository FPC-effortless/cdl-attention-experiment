"""Preserve an observed incomplete campaign without relaxing its original gates.

This is a separate delivery path from package_evidence.py. It neither loads a
model nor asserts that the unresolved process is alive or terminated. Copies
are checked against the independent incomplete-campaign audit and the original
files before and after packaging; canonical evidence is never moved or edited.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

import package_evidence as shared


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
PARENT = HERE.with_name("cog_fusion_001")
SEEDS = (17, 29, 41)
ARMS = ("continuous", "reset_training")
MISSING = ("continuous", 17)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def research_path(relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("unsafe research-relative receipt path: " + relative)
    result = HERE.parent / path
    result.resolve().relative_to(REPOSITORY.resolve())
    return result


def no_complete_campaign_or_scores():
    forbidden = ("RECOVERY_RESULT.json", "TRAINING_AUDIT.json", "QUALIFICATION.json",
                 "EVALUATION_LOCK.json", "EVALUATION_JOBS.json", "EVALUATION_EXIT.json",
                 "RUNTIME_RESULTS.json", "LANGUAGE_PRESERVATION.json",
                 "evaluation", "regression", "runtime", "state_interventions")
    present = [name for name in forbidden if (HERE / "evidence" / name).exists()]
    if present:
        raise ValueError("the observed incomplete/unscored disposition changed: " + ", ".join(present))
    unresolved = HERE / "evidence/runs/continuous_seed17"
    if any((unresolved / name).exists() for name in ("final.pt", "incomplete.pt", "optimizer.pt", "RESULT.json")):
        raise ValueError("the unresolved fit now has a terminal/checkpoint artifact; reassess before delivery")


def admitted_inputs():
    no_complete_campaign_or_scores()
    evidence = HERE / "evidence"
    audit_path = evidence / "INCOMPLETE_CAMPAIGN_AUDIT.json"
    audit = read(audit_path)
    if (audit.get("schema") != "cog-fusion-002-incomplete-campaign-audit-v1"
            or audit.get("campaign_status") != "incomplete" or audit.get("campaign_complete") is not False
            or audit.get("audit_status") != "completed"
            or audit.get("qualification_status") != "not_evaluated_incomplete_campaign"
            or audit.get("individual_completed_fit_integrity_pass") is not True):
        raise ValueError("the independent audit must admit individual artifacts and retain incomplete qualification")
    expected = {(arm, seed) for seed in SEEDS for arm in ARMS} - {MISSING}
    runs = audit.get("runs", [])
    if (len(runs) != 5 or {(row.get("arm"), row.get("seed")) for row in runs} != expected
            or any(row.get("individual_integrity_pass") is not True for row in runs)):
        raise ValueError("exactly the five retained fixed endpoints must pass individual integrity")
    names = {f"{arm}_seed{seed}" for arm, seed in expected}
    if set(audit.get("completed_fit_names", [])) != names:
        raise ValueError("incomplete audit endpoint inventory differs")
    required = [audit_path, evidence / "TRAINING_LOCK.json", evidence / "PRETRAIN_REVIEW.json",
                evidence / "CAMPAIGN_JOBS.json", evidence / "CAMPAIGN_EXIT.json",
                evidence / "RECOVERY_PROTOCOL.json", evidence / "RECOVERY_STARTED.json",
                evidence / "RUNTIME_PROTOCOL.json", evidence / "LANGUAGE_PROTOCOL.json",
                evidence / "COG_FUSION_002_LEARNING.png", evidence / "data/DATA_MANIFEST.json",
                evidence / "data/corpus.json", evidence / "runs/continuous_seed17/IDENTITY.json",
                evidence / "runs/continuous_seed17/train.jsonl", evidence / "logs/continuous_seed17.recovery.log"]
    required += [evidence / "data" / (name + ".json") for name in ("train", "dev", "test", "fresh", "long", "train_probe")]
    for row in runs:
        folder = evidence / "runs" / f"{row['arm']}_seed{row['seed']}"
        required += [folder / name for name in ("final.pt", "optimizer.pt", "IDENTITY.json", "RESULT.json", "train.jsonl")]
        result = read(folder / "RESULT.json")
        if (result.get("status") != "completed" or result.get("steps") != 1200
                or row.get("steps") != 1200
                or result["checkpoint_sha256"] != row["checkpoint_sha256"]
                or shared.sha256(folder / "final.pt") != row["checkpoint_sha256"]
                or shared.sha256(folder / "optimizer.pt") != row["optimizer_sha256"]
                or shared.sha256(folder / "RESULT.json") != row["result_sha256"]):
            raise ValueError("a retained individual endpoint changed after the partial audit")
    exits = read(evidence / "CAMPAIGN_EXIT.json")
    expected_exit = {(a, s): (-9 if (a, s) == MISSING else 0) for s in SEEDS for a in ARMS}
    if (len(exits) != 6 or {(r["arm"], r["seed"]): r["returncode"] for r in exits} != expected_exit):
        raise ValueError("original observed exit statuses differ")
    lock = read(evidence / "TRAINING_LOCK.json")
    for seed in SEEDS:
        path = PARENT / f"evidence/runs/memory_seed{seed}/final.pt"
        if shared.sha256(path) != lock["parent_sha256"][str(seed)]:
            raise ValueError("frozen parent input changed")
        required.append(path)
    inputs = PARENT / "evidence/reproduction_inputs"
    required += [inputs / name for name in ("MANIFEST.json", "COG_CONNECTED_1M.pt", "train.bin", "val.bin", "test.bin",
                                           "tokenizer.json", "dataset.json", "source.json")]
    required += [PARENT / "evidence/corpus.json", PARENT / "evidence/calibration/final.pt"]
    for name, metadata in read(inputs / "MANIFEST.json").items():
        if Path(name).name != name or shared.file_metadata(inputs / name) != metadata:
            raise ValueError("portable parent input differs: " + name)
    for field in ("input_sha256", "source_sha256"):
        for relative, expected_hash in audit[field].items():
            path = research_path(relative)
            if shared.file_metadata(path)["sha256"] != expected_hash:
                raise ValueError("partial-audited input/source changed: " + relative)
            required.append(path)
    snapshots = audit.get("stable_snapshots", {})
    if not snapshots:
        raise ValueError("the unresolved attempt requires explicitly bound observed snapshot copies")
    expected_trace = "cog_fusion_002/evidence/runs/continuous_seed17/train.jsonl"
    if expected_trace not in snapshots:
        raise ValueError("the current unresolved trace is absent from the observed snapshots")
    for original, snapshot in snapshots.items():
        metadata = {"bytes": snapshot["bytes"], "sha256": snapshot["sha256"]}
        for path in (research_path(original), research_path(snapshot["snapshot"])):
            if shared.file_metadata(path) != metadata:
                raise ValueError("source or copied observed snapshot changed: " + str(path))
            required.append(path)
    for path in required:
        shared.file_metadata(path)
    return audit, required


def restoration_guide(publication, audit):
    repository = publication["repository_full_name"]
    commit = publication["commit_sha"]
    rows = "\n".join(f"| {name} | 1,200 | Present |" for name in sorted(audit["completed_fit_names"]))
    return f"""# COG-FUSION-002 — incomplete campaign evidence

**This campaign is incomplete and has no held evaluation or qualification
result.** Five individual endpoints completed the fixed 1,200 updates. The
continuous seed-17 endpoint is absent. Its original attempt has a confirmed
negative return code (-9); its permitted replacement has no terminal receipt.
The lost execution handle and the stable copied files do not establish process
liveness, termination, an out-of-memory event, or a session timeout.

| Retained endpoint | Updates | Final and optimizer state |
|---|---|---|
{rows}
| continuous_seed17 | Incomplete | No final or optimizer checkpoint |

The independent incomplete-campaign audit checks individual artifacts, draw
replay, state integrity and recorded work. Completion of that audit does **not**
mean completion of the scientific campaign. No held scores were produced; no
targeted-repair, causal-regime, novel-wording, or language-preservation gate can
be decided from the five-endpoint subset.

## Restore the pinned source and observed evidence

Source commit: `{commit}`

- [Source and instructions](https://github.com/{repository}/tree/{commit}/research/cog_fusion_002)
- [Research report](https://github.com/{repository}/blob/{commit}/research/cog_fusion_002/RESULT.md)
- [Draft pull request]({publication['pull_request_url']})

Extract at the repository root. This archive complements the verified source
checkout and does not duplicate its published files.

```bash
git clone https://github.com/{repository}.git
cd {repository.split('/')[-1]}
git checkout {commit}
unzip -t /absolute/path/to/COG_FUSION_002_INCOMPLETE_EVIDENCE.zip
unzip /absolute/path/to/COG_FUSION_002_INCOMPLETE_EVIDENCE.zip
python3.12 - <<'PY'
import hashlib, json
from pathlib import Path
manifest = json.loads(Path("COG_FUSION_002_INCOMPLETE_EVIDENCE_MANIFEST.json").read_text())
publication = json.loads(Path("COG_FUSION_002_SOURCE_PUBLICATION.json").read_text())
for name, expected in list(manifest["members"].items()) + list(publication["files"].items()):
    path = Path(name)
    assert path.is_file() and path.stat().st_size == expected["bytes"], name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected["sha256"], name
print("Observed evidence and pinned source SHA-256 checks passed.")
PY
cd research/cog_fusion_002
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r ../cog_fusion_001/requirements.txt
```

The recorded environment used Python 3.12.14, Torch 2.10.0+cpu, NumPy 2.3.5
and pytest 8.4.2. CPU execution is sufficient for the supported integrity checks.
Keep source bytes and line endings unchanged because the retained locks bind
their hashes.

## Supported repeat checks

```bash
python -m pytest tests -q -ra
python review/audit_incomplete_campaign.py --output fresh_incomplete_audit.json --snapshot-dir fresh_incomplete_snapshots
```

Use fresh output and snapshot paths. The second command audits the five
individually completed endpoints and makes new observed copies of the partial
evidence. It does not infer terminal status, load a missing seed-17 endpoint,
run inference, or score held tasks.

The original `review/audit_training.py`, complete-campaign packaging, and
declared campaign scoring gates remain blocked by the missing sixth endpoint.
The once-only recovery marker is preserved. Do not remove it, synthesize a
`RECOVERY_RESULT.json`, resume partial weights, or silently launch another fit
under the original protocol. Any later continuation requires a separately
declared plan; these archived observations do not authorize one.

## What is preserved

All existing 002 raw data, labels, five final and optimizer/RNG checkpoints,
training traces, logs, calibration artifacts, review receipts and the learning
figure are retained. The original 13-update interrupted attempt remains under
`evidence/failed_attempts/`. The unresolved replacement remains in its canonical
`evidence/runs/continuous_seed17/` location, with independently bound observed
copies identified by `stable_snapshots` in `INCOMPLETE_CAMPAIGN_AUDIT.json`.
The original source files and all protocols remain unchanged.

Required frozen 001 memory parents, the original native checkpoint, exact
language inputs, calibration checkpoint and old corpus retain their sibling
paths. Every archive member has a byte count and SHA-256; packaging verifies
ZIP CRC and checks source/copy byte stability. A stable observation window is
not evidence of process death. No canonical input was moved, edited or silently
checkpointed during packaging.

The learning figure contains training-loss observations only. The confirmed
interrupted attempt and the unresolved replacement both contribute recorded
work; their logged CPU/token counts are lower bounds when unreported work is
unknown. This execution outcome does not establish that the model can or cannot
learn the intended repair.
"""


def package(publication_path, destination):
    started = stamp()
    publication_metadata = shared.file_metadata(publication_path)
    publication = read(publication_path)
    if shared.file_metadata(publication_path) != publication_metadata:
        raise ValueError("source publication receipt changed while being read")
    published = shared.publication_files(publication)
    audit, required = admitted_inputs()
    destination = Path(destination).resolve()
    if any(destination.is_relative_to(study.resolve()) for study in (HERE, PARENT)):
        raise ValueError("delivery must be outside both retained study directories")
    names = ("COG_FUSION_002_INCOMPLETE_EVIDENCE.zip", "COG_FUSION_002_RESTORE.md",
             "COG_FUSION_002_INCOMPLETE_PACKAGING.json", "COG_FUSION_002_LEARNING.png")
    if any((destination / name).exists() or (destination / name).is_symlink() for name in names):
        raise FileExistsError("incomplete delivery already exists; choose a fresh destination")
    candidates = list((HERE / "evidence").rglob("*")) + list((HERE / "review").rglob("*"))
    candidates += list((PARENT / "evidence/reproduction_inputs").rglob("*"))
    candidates += list((PARENT / "evidence/calibration").rglob("*")) + required
    members, sqlite_checks = {}, {}
    for path in candidates:
        if path.is_symlink():
            raise ValueError("symbolic link in observed evidence: " + str(path))
        if not path.is_file() or any(part in ("__pycache__", ".pytest_cache") for part in path.parts):
            continue
        name = str(path.resolve().relative_to(REPOSITORY.resolve()))
        if path.suffix == ".pyc" or path.name.endswith(("-wal", "-shm", "-journal")):
            raise ValueError("unclosed SQLite sidecar or cache in observed evidence: " + name)
        with path.open("rb") as handle:
            is_database = handle.read(16) == b"SQLite format 3\0"
        if is_database or path.suffix in (".sqlite", ".sqlite3", ".db"):
            if name not in sqlite_checks:
                sqlite_checks[name] = shared.sqlite_integrity(path)
        if name in published:
            continue
        if path.suffix in (".py", ".md"):
            raise ValueError("unpublished implementation/protocol in observed evidence: " + name)
        members[name] = path
    for path in required:
        name = str(path.resolve().relative_to(REPOSITORY.resolve()))
        if name not in published and name not in members:
            raise ValueError("required observed artifact has no delivery path: " + name)
    originals = {name: shared.file_metadata(path) for name, path in members.items()}
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / names[0]
    with tempfile.TemporaryDirectory(prefix=".observed-evidence-", dir=destination) as temporary:
        stage = Path(temporary)
        staged = {}
        for name, path in members.items():
            copied = stage / name
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, copied)
            if shared.file_metadata(copied) != originals[name] or shared.file_metadata(path) != originals[name]:
                raise ValueError("observed evidence changed while being copied: " + name)
            staged[name] = copied
        guide = stage / names[1]
        guide.write_text(restoration_guide(publication, audit))
        staged[guide.name] = guide
        source_record = stage / "COG_FUSION_002_SOURCE_PUBLICATION.json"
        shutil.copy2(publication_path, source_record)
        if (shared.file_metadata(source_record) != publication_metadata
                or shared.file_metadata(publication_path) != publication_metadata):
            raise ValueError("source publication receipt changed during copying")
        staged[source_record.name] = source_record
        if admitted_inputs()[0] != audit:
            raise ValueError("the incomplete-campaign audit changed during copying")
        shared.publication_files(publication)
        snapshot_verified = stamp()
        manifest = {
            "schema": "cog-fusion-002-incomplete-evidence-v1", "source_commit": publication["commit_sha"],
            "extract_at": "repository_root", "campaign_status": "incomplete", "campaign_complete": False,
            "qualification_status": "not_evaluated_incomplete_campaign", "unresolved_process_status": "unknown",
            "unresolved_exit_code": None, "completed_fit_names": audit["completed_fit_names"],
            "incomplete_campaign_audit_sha256": shared.sha256(HERE / "evidence/INCOMPLETE_CAMPAIGN_AUDIT.json"),
            "observation": {"packaging_started_utc": started, "source_and_copy_verified_utc": snapshot_verified,
                "interpretation": "stable byte copies during an observation window; no claim of process liveness or termination",
                "canonical_files_moved_or_modified": False},
            "stable_snapshots": audit["stable_snapshots"], "sqlite_checks": sqlite_checks,
            "excludes": "only files supplied by the verified complete two-study publication map",
            "members": {name: shared.file_metadata(path) for name, path in sorted(staged.items())}}
        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
            for name, path in sorted(staged.items()):
                bundle.write(path, name)
            bundle.writestr("COG_FUSION_002_INCOMPLETE_EVIDENCE_MANIFEST.json", manifest_bytes)
        with zipfile.ZipFile(archive) as bundle:
            if bundle.testzip() is not None:
                raise ValueError("incomplete evidence ZIP CRC verification failed")
            if (len(bundle.namelist()) != len(staged) + 1
                    or set(bundle.namelist()) != set(staged) | {"COG_FUSION_002_INCOMPLETE_EVIDENCE_MANIFEST.json"}
                    or bundle.read("COG_FUSION_002_INCOMPLETE_EVIDENCE_MANIFEST.json") != manifest_bytes):
                raise ValueError("incomplete archive inventory or manifest differs")
            for name, metadata in manifest["members"].items():
                value = hashlib.sha256()
                with bundle.open(name) as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        value.update(chunk)
                if bundle.getinfo(name).file_size != metadata["bytes"] or value.hexdigest() != metadata["sha256"]:
                    raise ValueError("archived observed bytes differ: " + name)
            uncompressed_bytes = sum(item.file_size for item in bundle.infolist())
        for name, metadata in originals.items():
            if shared.file_metadata(members[name]) != metadata:
                raise ValueError("observed source changed during archive verification: " + name)
        if admitted_inputs()[0] != audit:
            raise ValueError("the incomplete-campaign audit changed during archive verification")
        shared.publication_files(publication)
        if shared.file_metadata(publication_path) != publication_metadata:
            raise ValueError("source publication receipt changed during archive verification")
        shutil.copyfile(guide, destination / names[1])
        learning = HERE / "evidence/COG_FUSION_002_LEARNING.png"
        shutil.copyfile(learning, destination / names[3])
        if shared.file_metadata(destination / names[3]) != shared.file_metadata(learning):
            raise ValueError("standalone learning figure copy differs")
    receipt = {"status": "verified", "campaign_status": "incomplete", "campaign_complete": False,
        "qualification_status": "not_evaluated_incomplete_campaign", "unresolved_process_status": "unknown",
        "unresolved_exit_code": None, "archive": str(archive), "bytes": archive.stat().st_size,
        "sha256": shared.sha256(archive), "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "members": len(staged) + 1, "uncompressed_member_bytes": uncompressed_bytes,
        "source_commit": publication["commit_sha"], "packaging_started_utc": started, "verified_utc": stamp(),
        "zip_crc": "pass", "all_member_sha256": "pass", "source_copy_byte_stability": "pass",
        "canonical_files_moved_or_modified": False, "sqlite_databases_checked": len(sqlite_checks)}
    with (destination / names[2]).open("x") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-record", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    package(args.source_record, args.output_root)
