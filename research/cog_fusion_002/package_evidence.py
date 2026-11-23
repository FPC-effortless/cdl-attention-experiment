"""Package raw research evidence that complements a verified source publication.

Run after source publication. The archive is extracted at the repository root.
Git-backed files are excluded; their exact bytes are supplied by the pinned
checkout. This does not upload an archive or alter any experimental evidence.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import zipfile


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
PARENT = HERE.with_name("cog_fusion_001")


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def repository_path(relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or str(path) != relative:
        raise ValueError("unsafe repository-relative member name: " + relative)
    return REPOSITORY / path


def file_metadata(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("required regular evidence/source file missing: " + str(path))
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def publication_files(publication):
    prefixes = {"research/cog_fusion_001/", "research/cog_fusion_002/"}
    if (publication.get("schema") != "cog-fusion-002-source-publication-v1"
            or publication.get("status") != "verified"
            or set(publication.get("verified_study_prefixes", [])) != prefixes
            or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", publication.get("repository_full_name", ""))):
        raise ValueError("a verified, complete two-study source publication is required")
    for key in ("commit_sha", "source_parent_commit_sha", "parent_tree_sha", "tree_sha"):
        if not re.fullmatch(r"[0-9a-f]{40}", publication.get(key, "")):
            raise ValueError("invalid published Git identity: " + key)
    published = publication["files"]
    if (not published or not all(any(name.startswith(prefix) for prefix in prefixes) for name in published)
            or not all(any(name.startswith(prefix) for name in published) for prefix in prefixes)):
        raise ValueError("published file inventory is outside the verified study prefixes")
    for relative, metadata in published.items():
        path = repository_path(relative)
        if file_metadata(path) != {"bytes": metadata["bytes"], "sha256": metadata["sha256"]}:
            raise ValueError("published source differs from local bytes: " + relative)
        raw = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if blob != metadata["git_blob_sha"]:
            raise ValueError("published Git blob differs from local bytes: " + relative)
    return published


def required_artifacts():
    evidence = HERE / "evidence"
    names = ("QUALIFICATION.json", "TRAINING_AUDIT.json", "TRAINING_LOCK.json", "PRETRAIN_REVIEW.json",
             "EVALUATION_LOCK.json", "EVALUATION_EXIT.json", "CAMPAIGN_JOBS.json", "CAMPAIGN_EXIT.json",
             "RUNTIME_RESULTS.json", "LANGUAGE_PRESERVATION.json", "RUNTIME_PROTOCOL.json", "LANGUAGE_PROTOCOL.json",
             "COG_FUSION_002_RESULTS.png", "COG_FUSION_002_LEARNING.png", "data/DATA_MANIFEST.json", "data/corpus.json")
    paths = [evidence / name for name in names]
    paths += [evidence / "data" / (split + ".json") for split in ("train", "dev", "test", "fresh", "long", "train_probe")]
    for seed in (17, 29, 41):
        paths.append(PARENT / f"evidence/runs/memory_seed{seed}/final.pt")
        for arm in ("continuous", "reset_training"):
            folder = evidence / "runs" / f"{arm}_seed{seed}"
            paths += [folder / leaf for leaf in ("final.pt", "optimizer.pt", "IDENTITY.json", "RESULT.json", "train.jsonl")]
        for arm in ("continuous", "reset_training", "parent"):
            folder = evidence / "evaluation" / f"{arm}_seed{seed}"
            paths += [folder / (panel + ".json") for panel in ("train_probe", "dev", "test", "fresh", "long", "RESULT")]
            paths.append(evidence / "regression" / f"{arm}_seed{seed}" / "RESULT.json")
            paths.append(evidence / "state_interventions" / f"{arm}_seed{seed}.json")
            runtime_arm = "parent_001" if arm == "parent" else arm
            folder = evidence / "runtime" / f"{runtime_arm}_seed{seed}"
            paths += [folder / "RESULT.json", folder / "exposed_001/session.sqlite", folder / "new_schedule/session.sqlite"]
    inputs = PARENT / "evidence/reproduction_inputs"
    paths += [inputs / leaf for leaf in ("MANIFEST.json", "COG_CONNECTED_1M.pt", "train.bin", "val.bin", "test.bin",
                                        "tokenizer.json", "dataset.json", "source.json")]
    paths += [PARENT / "evidence/corpus.json", PARENT / "evidence/calibration/final.pt"]
    for path in paths:
        file_metadata(path)
    for name, metadata in read(inputs / "MANIFEST.json").items():
        path = inputs / name
        if Path(name).name != name or file_metadata(path) != metadata:
            raise ValueError("portable parent input differs from its retained manifest: " + name)
    training = read(evidence / "TRAINING_AUDIT.json")
    if training.get("pass") is not True or training.get("status") != "completed":
        raise ValueError("training integrity did not pass")
    lock = read(evidence / "TRAINING_LOCK.json")
    for seed in (17, 29, 41):
        if sha256(PARENT / f"evidence/runs/memory_seed{seed}/final.pt") != lock["parent_sha256"][str(seed)]:
            raise ValueError("required frozen parent changed")
        for arm in ("continuous", "reset_training"):
            folder = evidence / "runs" / f"{arm}_seed{seed}"
            result = read(folder / "RESULT.json")
            if (result.get("status") != "completed" or result.get("steps") != 1200
                    or result.get("checkpoint_sha256") != sha256(folder / "final.pt")):
                raise ValueError("missing or changed fixed final endpoint")
    # Include every file the final audit and reconciliation actually consumed,
    # even when an inherited raw dependency sits outside the usual directories.
    qualification = read(evidence / "QUALIFICATION.json")
    for relative, expected in training["input_sha256"].items():
        path = HERE.parent / relative
        path.resolve().relative_to(REPOSITORY.resolve())
        if file_metadata(path)["sha256"] != expected:
            raise ValueError("training-audited input changed: " + relative)
        paths.append(path)
    for recorded, expected in qualification["source_receipt_sha256"].items():
        path = Path(recorded)
        if path.is_absolute():
            # Historical receipts retain the original workspace prefix. Restore
            # by the stable repository/study suffix, without editing receipts.
            starts = [i for i, part in enumerate(path.parts[:-1]) if part == "research"
                      and path.parts[i + 1] in ("cog_fusion_001", "cog_fusion_002")]
            if len(starts) != 1:
                raise ValueError("qualified path has no unambiguous study suffix: " + recorded)
            path = repository_path(str(Path(*path.parts[starts[0]:])))
        elif path.parts[0] == "research":
            path = repository_path(recorded)
        else:
            path = HERE / path
        path.resolve().relative_to(REPOSITORY.resolve())
        if file_metadata(path)["sha256"] != expected:
            raise ValueError("qualified evidence changed: " + recorded)
        paths.append(path)
    return paths, qualification


def sqlite_integrity(path):
    path = Path(path)
    for suffix in ("-wal", "-shm", "-journal"):
        if Path(str(path) + suffix).exists():
            raise ValueError("SQLite sidecar remains; close the writer before packaging: " + str(path))
    before = file_metadata(path)
    # immutable=1 prevents readonly opening from creating WAL/SHM files. It is
    # appropriate only after the explicit no-sidecar check above.
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)) as database:
        if database.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise ValueError("SQLite integrity check failed: " + str(path))
        if database.execute("PRAGMA foreign_key_check").fetchall():
            raise ValueError("SQLite foreign-key check failed: " + str(path))
    if file_metadata(path) != before:
        raise ValueError("SQLite evidence changed during readonly inspection: " + str(path))
    return {"integrity_check": "ok", "foreign_key_check": "ok", "inspection": "immutable readonly; original bytes preserved"}


def restoration_guide(publication, qualification):
    commit = publication["commit_sha"]
    repository = publication["repository_full_name"]
    base = f"https://github.com/{repository}"
    source = f"{base}/tree/{commit}/research/cog_fusion_002"
    report = f"{base}/blob/{commit}/research/cog_fusion_002/RESULT.md"
    decisions = qualification["decisions"]
    rows = "\n".join(f"| {label} | {'PASS' if decisions[key] else 'FAIL'} |" for key, label in (
        ("targeted_familiar_repair_qualified", "Combined familiar repair"),
        ("specific_regime_advantage_qualified", "Specific advantage of continuous training"),
        ("all_continuous_novel_wording_absolute_gates", "New-wording absolute gates"),
        ("all_continuous_original_full_language_recovery", "Language within 10% NLL of original full model")))
    return f"""# COG-FUSION-002 — evidence and reproduction

This completed study tests multi-turn training in the unchanged 997,674-active-
parameter model. It retains three matched seeds, both training policies, fixed
1,200-update endpoints, and the frozen parents. Read the report for measured
performance and failed gates; the existence of a runnable checkpoint is not a
general-purpose capability claim.

| Prespecified decision | Result |
|---|---|
{rows}

- [Source and README]({source})
- [Complete research report]({report})
- [Draft pull request]({publication['pull_request_url']})

Source commit: `{commit}`

## Restore

The source checkout supplies implementations, protocols, tests, reports and
public evidence summaries. This archive supplies complementary raw evidence,
exact inputs and checkpoints. Extract at the **repository root**. Both studies
retain their sibling paths under `research/`.

```bash
git clone {base}.git
cd {repository.split('/')[-1]}
git checkout {commit}
unzip -t /absolute/path/to/COG_FUSION_002_EVIDENCE.zip
unzip /absolute/path/to/COG_FUSION_002_EVIDENCE.zip
python3.12 - <<'PY'
import hashlib, json
from pathlib import Path
manifest = json.loads(Path("COG_FUSION_002_EVIDENCE_MANIFEST.json").read_text())
publication = json.loads(Path("COG_FUSION_002_SOURCE_PUBLICATION.json").read_text())
for name, expected in list(manifest["members"].items()) + list(publication["files"].items()):
    path = Path(name)
    assert path.is_file() and path.stat().st_size == expected["bytes"], name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected["sha256"], name
print("Extracted evidence and pinned source SHA-256 checks passed.")
PY
cd research/cog_fusion_002
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r ../cog_fusion_001/requirements.txt
```

The recorded environment used Python 3.12.14, Torch 2.10.0+cpu, NumPy 2.3.5
and pytest 8.4.2. Matplotlib is required only for plot regeneration. CPU execution
is sufficient. Preserve source-file bytes and line endings because the locks and
checkpoint identities bind them.

`COG_FUSION_002_EVIDENCE_MANIFEST.json` records the byte length and SHA-256 of
each archive member except the manifest itself. The source-publication receipt
records the exact published file hashes, parent commit, final tree and draft.
The packaging receipt also records a completed ZIP CRC check.

## Verify engineering and retained evidence

```bash
python -m pytest tests -q -ra
python review/audit_training.py --output fresh_training_audit.json
python summarize.py --output fresh_qualification.json
```

The training audit reconstructs all family and rehearsal draws, inspects final
and optimizer/RNG states, and accounts for interrupted work. It performs no
model inference. The summary independently reconciles the saved generated
histories and uses the prespecified family-level bootstrap. These commands
write fresh outputs; the original evidence remains intact.

## Repeat a model evaluation

```bash
python evaluate.py --checkpoint evidence/runs/continuous_seed17/final.pt --data evidence/data/test.json --output fresh_test_seed17.json
python evaluate.py --checkpoint evidence/runs/reset_training_seed17/final.pt --data evidence/data/train_probe.json --reset-after-answer --output fresh_reset_probe_seed17.json
python preservation_eval.py --output fresh_language_preservation.json
python runtime_eval.py --output fresh_runtime/RUNTIME_RESULTS.json
```

Seed 17 is a fixed representative, not a chosen best seed. All primary held
evaluations carry state, including evaluation of the reset-trained control.
The reset option belongs only to that arm's TRAIN adequacy probe or an explicitly
labelled diagnostic. The original campaign launchers refuse existing targets;
use new filenames for repeats. The runtime command uses a fresh **output parent
directory** and creates its databases below `fresh_runtime/runtime/`; changing
only the result filename inside the original evidence directory would reuse
existing database paths and is refused.

## Use a saved persistent session

```bash
PYTHONPATH=../cog_fusion_001 python -m fusion.cli --checkpoint evidence/runs/continuous_seed17/final.pt --db demo.sqlite observe "Amara owns the chest."
PYTHONPATH=../cog_fusion_001 python -m fusion.cli --checkpoint evidence/runs/continuous_seed17/final.pt --db demo.sqlite observe "The coin is in the chest."
PYTHONPATH=../cog_fusion_001 python -m fusion.cli --checkpoint evidence/runs/continuous_seed17/final.pt --db demo.sqlite ask "Who owns the container holding the coin?"
PYTHONPATH=../cog_fusion_001 python -m fusion.cli --checkpoint evidence/runs/continuous_seed17/final.pt --db demo.sqlite observe "The chest belongs to Bela."
PYTHONPATH=../cog_fusion_001 python -m fusion.cli --checkpoint evidence/runs/continuous_seed17/final.pt --db demo.sqlite ask "Who owns the container holding the coin?"
```

Each command starts a fresh process and resumes the saved state. Use a new
database when changing checkpoints. The model uses the unchanged fixed entity
vocabulary and explicit observation/question boundaries. Receipts and factual
storage can grow even though the neural continuation is fixed at 10,592 bytes.
Persistence preserves accepted state, including any learned interpretation error.

## Contents and scope

The archive includes all six adapted final and optimizer/RNG checkpoints;
every training trace; any preserved interrupted attempt; the exact new corpus
and split files; generated evaluations and old-panel regression traces; runtime
databases and request receipts; preselected state interventions; calibration
evidence; and the two result figures. Required 001 memory parent checkpoints,
the original native checkpoint, exact language inputs, old corpus and calibration
checkpoint are included under the sibling study. Source and public summaries
remain in the pinned checkout rather than being duplicated in this archive.
Every archived SQLite database passed an immutable readonly integrity check.
No WAL, SHM or rollback-journal sidecar was discarded or silently checkpointed.

The fixed TRAIN-pool adequacy probe includes families not sampled by every seed.
It is a training-distribution check, not pure memorization or held generalization.
The old panels and language windows were previously exposed. The new runtime
schedule is descriptive. The study does not measure arbitrary tasks, new entity
vocabularies, general long-context reasoning, joint language-driven numerical
computation or quality-matched efficiency against attention.
"""


def package(publication_path, destination):
    publication = read(publication_path)
    published = publication_files(publication)
    evidence = HERE / "evidence"
    required, qualification = required_artifacts()
    destination = Path(destination).resolve()
    if any(destination.is_relative_to(study.resolve()) for study in (HERE, PARENT)):
        raise ValueError("deliverables must be outside both retained study directories")
    archive = destination / "COG_FUSION_002_EVIDENCE.zip"
    guide = destination / "COG_FUSION_002_RESTORE.md"
    outputs = [archive, guide, destination / "COG_FUSION_002_PACKAGING.json"]
    outputs += [destination / name for name in ("COG_FUSION_002_RESULTS.png", "COG_FUSION_002_LEARNING.png")]
    if any(path.exists() or path.is_symlink() for path in outputs):
        raise FileExistsError("deliverables already exist; choose a fresh destination")
    candidates = list(evidence.rglob("*")) + list((HERE / "review").rglob("*"))
    candidates += list((PARENT / "evidence/reproduction_inputs").rglob("*"))
    candidates += list((PARENT / "evidence/calibration").rglob("*"))
    candidates += required
    members, databases = {}, {}
    for path in candidates:
        if path.is_symlink():
            raise ValueError("symbolic link in retained evidence: " + str(path))
        if not path.is_file() or any(part in ("__pycache__", ".pytest_cache") for part in path.parts):
            continue
        relative = str(path.resolve().relative_to(REPOSITORY.resolve()))
        if path.suffix == ".pyc" or path.name.endswith(("-wal", "-shm", "-journal")):
            raise ValueError("unclosed database or cache in retained evidence: " + relative)
        with path.open("rb") as handle:
            is_database = handle.read(16) == b"SQLite format 3\0"
        if is_database or path.suffix in (".sqlite", ".sqlite3", ".db"):
            if relative not in databases:
                databases[relative] = sqlite_integrity(path)
        if relative in published:
            continue
        if path.suffix in (".py", ".md"):
            raise ValueError("an implementation or protocol is missing from the verified checkout: " + relative)
        members[relative] = path
    for path in required:
        relative = str(path.resolve().relative_to(REPOSITORY.resolve()))
        if relative not in published and relative not in members:
            raise ValueError("required artifact is supplied by neither checkout nor archive: " + relative)
    destination.mkdir(parents=True, exist_ok=True)
    with guide.open("x") as handle:
        handle.write(restoration_guide(publication, qualification))
    members[guide.name] = guide
    members["COG_FUSION_002_SOURCE_PUBLICATION.json"] = Path(publication_path).resolve()
    manifest = {"schema": "cog-fusion-002-complementary-evidence-v1",
                "source_commit": publication["commit_sha"], "extract_at": "repository_root",
                "excludes": "only paths in the verified complete two-study publication file map",
                "sqlite_checks": databases,
                "required_artifact_delivery": {str(path.resolve().relative_to(REPOSITORY.resolve())):
                    "pinned_checkout" if str(path.resolve().relative_to(REPOSITORY.resolve())) in published else "archive"
                    for path in required},
                "members": {name: file_metadata(path) for name, path in sorted(members.items())}}
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for name, path in sorted(members.items()):
            bundle.write(path, name)
        bundle.writestr("COG_FUSION_002_EVIDENCE_MANIFEST.json", manifest_bytes)
    with zipfile.ZipFile(archive) as bundle:
        bad = bundle.testzip()
        if bad is not None:
            raise ValueError("ZIP CRC failed: " + bad)
        if (len(bundle.namelist()) != len(members) + 1
                or set(bundle.namelist()) != set(members) | {"COG_FUSION_002_EVIDENCE_MANIFEST.json"}):
            raise ValueError("archive member inventory differs")
        if bundle.read("COG_FUSION_002_EVIDENCE_MANIFEST.json") != manifest_bytes:
            raise ValueError("archived manifest bytes differ")
        for name, metadata in manifest["members"].items():
            value = hashlib.sha256()
            with bundle.open(name) as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    value.update(chunk)
            if bundle.getinfo(name).file_size != metadata["bytes"] or value.hexdigest() != metadata["sha256"]:
                raise ValueError("archived byte hash differs: " + name)
        uncompressed_bytes = sum(item.file_size for item in bundle.infolist())
    # The manifest binds the inputs at archive creation. Recheck live source and
    # evidence bytes so a concurrent modification cannot masquerade as a stable
    # completed packaging operation.
    for name, metadata in manifest["members"].items():
        if file_metadata(members[name]) != metadata:
            raise ValueError("source evidence changed while packaging: " + name)
    publication_files(publication)
    for name in ("COG_FUSION_002_RESULTS.png", "COG_FUSION_002_LEARNING.png"):
        shutil.copyfile(evidence / name, destination / name)
    receipt = {"status": "verified", "archive": str(archive), "bytes": archive.stat().st_size,
               "sha256": sha256(archive), "members": len(members) + 1,
               "uncompressed_member_bytes": uncompressed_bytes,
               "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
               "zip_crc": "pass", "all_member_sha256": "pass", "sqlite_integrity": "pass",
               "sqlite_databases_checked": len(databases), "source_commit": publication["commit_sha"]}
    (destination / "COG_FUSION_002_PACKAGING.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-record", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    package(args.source_record, args.output_root)
