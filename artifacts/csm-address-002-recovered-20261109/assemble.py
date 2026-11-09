"""Reassemble and verify the unchanged recovered CSM archive."""
from pathlib import Path
import hashlib
import json


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "MANIFEST.json").read_text())
    chunks = []
    for part in manifest["parts"]:
        name = part["path"]
        if Path(name).name != name:
            raise SystemExit("Invalid part filename")
        data = (root / name).read_bytes()
        if (len(data) != part["bytes"] or
                hashlib.sha256(data).hexdigest() != part["sha256"]):
            raise SystemExit(f"Part integrity check failed: {name}")
        chunks.append(data)
    payload = b"".join(chunks)
    if (len(payload) != manifest["source_archive_bytes"] or
            hashlib.sha256(payload).hexdigest() != manifest["source_archive_sha256"]):
        raise SystemExit("Assembled archive integrity check failed")
    name = manifest["source_archive"]
    if Path(name).name != name:
        raise SystemExit("Invalid archive filename")
    output = root / name
    if output.exists():
        if output.read_bytes() != payload:
            raise SystemExit("Refusing to overwrite a different existing archive")
    else:
        with output.open("xb") as handle:
            handle.write(payload)
    print(f"Verified {output.name}: {len(payload)} bytes")


if __name__ == "__main__":
    main()
