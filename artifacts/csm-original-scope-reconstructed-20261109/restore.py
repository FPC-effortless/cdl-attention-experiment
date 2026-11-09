#!/usr/bin/env python3
"""Verify the original CSM archives and restore missing files without overwrites."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile


ARCHIVES = {
    "CSM-FOUNDATION-001-review.zip": {
        "bytes": 14034393,
        "sha256": "ac8291f94c04a4d73e395f5eba005b8ea5773813e2829cd86514ced605c1233f",
    },
    "CSM-GRAPH-001-verified-project.zip": {
        "bytes": 4864191,
        "sha256": "9b1f30ca4a6008d4ecb9c0f7c7e6ab46e89d53107c75985467152e1c9f9ac32d",
    },
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_name(name: str) -> None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Unsafe archive path: {name}")


def checked_archive(directory: Path, name: str) -> dict[str, bytes]:
    archive = directory / name
    expected = ARCHIVES[name]
    data = archive.read_bytes()
    if len(data) != expected["bytes"] or digest(data) != expected["sha256"]:
        raise ValueError(f"Archive integrity mismatch: {name}")
    contents = {}
    with zipfile.ZipFile(archive) as zf:
        for entry in zf.infolist():
            safe_name(entry.filename)
            if stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError(f"Archive symlink rejected: {entry.filename}")
            if entry.is_dir():
                continue
            if entry.filename in contents:
                raise ValueError(f"Duplicate archive path: {entry.filename}")
            contents[entry.filename] = zf.read(entry)
    return contents


def collect(directory: Path) -> tuple[dict[str, bytes], dict]:
    foundation = checked_archive(directory, "CSM-FOUNDATION-001-review.zip")
    checksums = json.loads(foundation["CHECKSUMS.json"])
    if set(foundation) != set(checksums) | {"CHECKSUMS.json"}:
        raise ValueError("Foundation manifest does not cover exactly the archive entries")
    for name, expected in checksums.items():
        data = foundation[name]
        if len(data) != expected["bytes"] or digest(data) != expected["sha256"]:
            raise ValueError(f"Foundation entry mismatch: {name}")

    graph = checked_archive(directory, "CSM-GRAPH-001-verified-project.zip")
    prefix = "CSM-GRAPH-001/"
    manifest = json.loads(graph[prefix + "DELIVERY_MANIFEST.json"])
    if digest(graph[prefix + "CSM-GRAPH-001.bundle"]) != manifest["bundle_sha256"]:
        raise ValueError("Graph Git bundle mismatch")
    if digest(graph[prefix + "CSM-GRAPH-001-mathematical-model.md"]) != manifest["report_sha256"]:
        raise ValueError("Graph report mismatch")
    mapping = {}
    for name, expected in manifest["files_sha256"].items():
        safe_name(name)
        data = graph[prefix + "csm-foundation/" + name]
        if digest(data) != expected:
            raise ValueError(f"Graph snapshot entry mismatch: {name}")
        mapping[name] = data

    raw_count = 0
    checkpoint_count = 0
    overlaps = 0
    for name, data in foundation.items():
        if not name.startswith("raw-evidence/"):
            continue
        raw_count += 1
        checkpoint_count += name.endswith("/checkpoint.pt")
        target = "experiments/csm_foundation_001/" + name.removeprefix("raw-evidence/")
        safe_name(target)
        if target in mapping:
            overlaps += 1
            if mapping[target] != data:
                raise ValueError(f"Archives disagree: {target}")
        mapping[target] = data
    if (raw_count, checkpoint_count) != (242, 30):
        raise ValueError(f"Unexpected evidence counts: {raw_count}, {checkpoint_count}")
    return mapping, {
        "archives_verified": 2,
        "foundation_manifest_entries_verified": len(checksums),
        "raw_evidence_files": raw_count,
        "model_optimizer_checkpoints": checkpoint_count,
        "graph_snapshot_files_verified": len(manifest["files_sha256"]),
        "identical_archive_overlap": overlaps,
        "unique_restore_paths": len(mapping),
    }


def check_destination(root: Path, name: str, data: bytes) -> tuple[Path, bool]:
    destination = root / name
    # Refuse existing symlinks in any component, including dangling links.
    current = root
    for component in PurePosixPath(name).parts:
        current = current / component
        if current.is_symlink():
            raise ValueError(f"Destination symlink rejected: {current}")
    if not destination.resolve().is_relative_to(root):
        raise ValueError(f"Destination escapes target: {name}")
    if destination.exists():
        if not destination.is_file() or destination.read_bytes() != data:
            raise ValueError(f"Refusing to overwrite differing file: {destination}")
        return destination, True
    return destination, False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--verify-only", action="store_true", help="Check integrity and destination conflicts without writing")
    args = parser.parse_args()
    root = args.target.resolve()
    mapping, summary = collect(Path(__file__).resolve().parent)
    # Complete preflight before the first write, so an existing conflict causes no partial restore.
    destinations = [(name, *check_destination(root, name, data)) for name, data in mapping.items()]
    missing = [(name, destination) for name, destination, exists in destinations if not exists]
    summary.update({"target": str(root), "existing_identical": len(mapping) - len(missing), "missing_before_restore": len(missing), "verify_only": args.verify_only, "restored_files": 0})
    if not args.verify_only:
        for name, destination in missing:
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Recheck the destination immediately before writing; exclusive creation prevents overwrites.
            _, exists = check_destination(root, name, mapping[name])
            if not exists:
                with destination.open("xb") as output:
                    output.write(mapping[name])
                summary["restored_files"] += 1
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
