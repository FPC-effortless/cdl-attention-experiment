"""Reconstruct and verify the original COG-ATTN evidence archive bytes."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
import time
from urllib.parse import quote
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
REPO = "FPC-effortless/cdl-attention-experiment"
DEFAULT_REF = "research/cog-attn-continuation-evidence-20261120"
BUFFER = 1024 * 1024


def remote_url(ref, path):
    return f"https://raw.githubusercontent.com/{REPO}/{quote(ref, safe='/')}/{quote(path, safe='/')}"


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def get_manifest(ref):
    local = HERE / "manifest.json"
    if ref is None and local.is_file():
        return json.loads(local.read_text())
    with urlopen(remote_url(ref or DEFAULT_REF, "manifest.json"), timeout=60) as response:
        return json.load(response)


def part_source(part, ref):
    relative = PurePosixPath(part["path"])
    if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "archives":
        raise ValueError(f"Invalid part path {relative}")
    local = HERE.joinpath(*relative.parts)
    if ref is None and local.is_file():
        return local.open("rb")
    return urlopen(remote_url(ref or DEFAULT_REF, str(relative)), timeout=120)


def copy_part(part, ref, output):
    start = output.tell()
    for attempt in range(3):
        output.seek(start)
        output.truncate()
        received = hashlib.sha256()
        count = 0
        try:
            with part_source(part, ref) as stream:
                while block := stream.read(BUFFER):
                    output.write(block)
                    received.update(block)
                    count += len(block)
        except OSError:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)
            continue
        if count == part["bytes"] and received.hexdigest() == part["sha256"]:
            return
        if attempt == 2:
            raise ValueError(f"Part content mismatch: {part['path']}")
        time.sleep(1 + attempt)


def assemble(entry, target, ref):
    name = entry["name"]
    if re.fullmatch(r"[A-Za-z0-9._-]+[.]tar[.]gz", name) is None:
        raise ValueError(f"Invalid archive name {name!r}")
    final = target / name
    if final.exists():
        if final.stat().st_size == entry["bytes"] and digest(final) == entry["sha256"]:
            return final
        raise FileExistsError(f"Existing file differs: {final}")
    with tempfile.NamedTemporaryFile(dir=target, prefix=".evidence-", delete=False) as output:
        temporary = Path(output.name)
        try:
            for part in entry["parts"]:
                copy_part(part, ref, output)
            output.close()
            if temporary.stat().st_size != entry["bytes"] or digest(temporary) != entry["sha256"]:
                raise ValueError(f"Original archive hash mismatch: {name}")
            os.link(temporary, final)
        finally:
            temporary.unlink(missing_ok=True)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--ref", help="Fetch the manifest and parts from this GitHub commit or branch")
    parser.add_argument("--output", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if not (args.all or args.list or args.names):
        parser.error("choose an archive, --list, or --all")
    entries = {x["name"]: x for x in get_manifest(args.ref)["archives"]}
    if args.list:
        for name in entries:
            print(name)
        return
    requested = list(entries) if args.all else args.names
    if any(name not in entries for name in requested):
        parser.error("unknown archive. See --list")
    args.output.mkdir(parents=True, exist_ok=True)
    for name in requested:
        print(f"Verified {assemble(entries[name], args.output, args.ref)}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        sys.exit(str(error))
