#!/usr/bin/env python3
"""Verify and restore the frozen LRC repair handoff without running the model."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile


HERE = Path(__file__).resolve().parent
SHA256 = {
    "LRC_ROUTE_REPAIR_002_Checkpoints_and_Evidence_20261108.zip":
        "06ad6c3d1e6647fa4912ec8631ca7367e5e4c775420f3573c1a024ea5f50daf6",
    "LRC_ROUTE_REPAIR_002_Code.patch":
        "1b25a4582994ec947fc7b2407470efe546640c744747d7a34a32f3e2d18a71eb",
    "lrc_vb001_matched_comparison.zip":
        "20196fc1c54a7fcfc177d977883728fe5e07edccb6491a8b8e8921c5e4efe3f5",
}


def digest(stream):
    return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=HERE / "restored")
    parser.add_argument("--verify-only", action="store_true",
                        help="Check artifact hashes and ZIP integrity without extracting.")
    args = parser.parse_args()
    destination = args.destination.resolve()
    for name, expected in SHA256.items():
        with (HERE / name).open("rb") as source:
            if digest(source) != expected:
                raise ValueError(f"SHA256 mismatch: {name}")

    # Check every archive and destination before writing any extracted file.
    planned, names, reused = [], set(), 0
    with ExitStack() as stack:
        for name in SHA256:
            if not name.endswith(".zip"):
                continue
            archive = stack.enter_context(zipfile.ZipFile(HERE / name))
            if archive.testzip() is not None:
                raise ValueError(f"ZIP integrity failure: {name}")
            for info in archive.infolist():
                relative = PurePosixPath(info.filename)
                kind = stat.S_IFMT(info.external_attr >> 16)
                if (relative.is_absolute() or ".." in relative.parts
                        or "\\" in info.filename or ":" in info.filename
                        or kind not in (0, stat.S_IFREG, stat.S_IFDIR)):
                    raise ValueError(f"Unsafe archive member: {info.filename}")
                if info.is_dir():
                    continue
                target = destination.joinpath(*relative.parts)
                if target in names:
                    raise ValueError(f"Duplicate archive destination: {info.filename}")
                names.add(target)
                if args.verify_only:
                    continue
                for ancestor in (target, *target.parents):
                    if ancestor.is_symlink():
                        raise ValueError(f"Refusing a symbolic-link destination: {ancestor}")
                    if ancestor != target and ancestor.exists() and not ancestor.is_dir():
                        raise ValueError(f"Destination parent is not a directory: {ancestor}")
                    if ancestor == destination:
                        break
                if target.exists():
                    if not target.is_file() or target.stat().st_size != info.file_size:
                        raise ValueError(f"Existing destination differs: {target}")
                    with target.open("rb") as existing, archive.open(info) as source:
                        if digest(existing) != digest(source):
                            raise ValueError(f"Existing destination differs: {target}")
                    reused += 1
                else:
                    planned.append((archive, info, target))
        for archive, info, target in planned:
            target.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation also prevents an overwrite if a file appeared
            # after the preflight checks. ZIP payloads are never executed.
            with archive.open(info) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)

    print(json.dumps({"status": "verified" if args.verify_only else "restored",
                      "artifacts": len(SHA256), "archive_files": len(names),
                      "written": len(planned), "reused": reused,
                      "destination": str(destination)}, indent=2))


if __name__ == "__main__":
    main()
