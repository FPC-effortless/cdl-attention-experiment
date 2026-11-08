#!/usr/bin/env python3
"""Verify a recovered Git bundle and optionally publish its two exact branches.

The default mode only verifies in a temporary repository. Publication is limited
to two previously recorded refs, and refuses a differing existing remote ref.
No model code, checkpoint, or archive payload is executed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
REPOSITORY = "FPC-effortless/cdl-attention-experiment"
PUBLISH_REF = "refs/heads/ops/publish-lrc-route-repair-002-20261108"
REPAIR = "3f71af6eb53827eb7eb3c74e10964f70e8e84460"
HANDOFF = "566ba9933a35cf88a55f830891d8d6e33d6ca524"
EXPECTED_REFS = {
    "refs/heads/research/lrc-route-repair-002-20261108": REPAIR,
    "refs/heads/research/lrc-route-repair-002-handoff-20261108": HANDOFF,
}
PREREQUISITES = [
    "b50e1615e16410b06af34081e83762460c0f9de3",
    "c26ff92b8ecec74045bd2393cdaf956c99a0d9d5",
]
BUNDLE_SHA256 = "4c6d0d2ce2a4a38a8248bd1923781164f1c43f317b676d39146f8009b27c622c"
ARTIFACT_ROOT = "artifacts/lrc-route-repair-002"
EXPECTED_ARTIFACTS = {
    "LRC_ROUTE_REPAIR_002_Checkpoints_and_Evidence_20261108.zip": {
        "bytes": 5489762,
        "sha256": "06ad6c3d1e6647fa4912ec8631ca7367e5e4c775420f3573c1a024ea5f50daf6",
    },
    "LRC_ROUTE_REPAIR_002_Code.patch": {
        "bytes": 184354,
        "sha256": "1b25a4582994ec947fc7b2407470efe546640c744747d7a34a32f3e2d18a71eb",
    },
    "lrc_vb001_matched_comparison.zip": {
        "bytes": 7358645,
        "sha256": "20196fc1c54a7fcfc177d977883728fe5e07edccb6491a8b8e8921c5e4efe3f5",
    },
}


def run(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True,
                          text=True, timeout=180).stdout.strip()


def git(repo: Path, *args: str) -> str:
    return run("git", "-C", str(repo), *args)


def remote_git(repo: Path, *args: str) -> str:
    # GitHub CLI reads GH_TOKEN from the job environment. No credential is
    # inserted into the remote URL, arguments, files, or logs.
    return git(repo, "-c", "credential.helper=", "-c",
               "credential.helper=!gh auth git-credential", *args)


def parse_refs(output: str) -> dict[str, str]:
    return {ref: sha for sha, ref in (line.split() for line in output.splitlines())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    if args.publish:
        if (os.environ.get("GITHUB_REPOSITORY") != REPOSITORY
                or os.environ.get("GITHUB_REF") != PUBLISH_REF
                or not os.environ.get("GH_TOKEN")):
            raise RuntimeError("Publication requires the scoped GitHub Actions job")

    manifest = json.loads((HERE / "manifest.json").read_text())
    if (manifest["schema"] != 1 or manifest["repository"] != REPOSITORY
            or manifest["refs"] != EXPECTED_REFS
            or manifest["prerequisites"] != PREREQUISITES
            or manifest["artifacts"] != EXPECTED_ARTIFACTS
            or manifest["bundle"]["sha256"] != BUNDLE_SHA256):
        raise RuntimeError("Publication manifest differs from the frozen scope")

    source = Path(git(HERE, "rev-parse", "--show-toplevel"))
    if args.publish and git(source, "rev-parse", "--is-shallow-repository") != "false":
        raise RuntimeError("Publication requires the complete prerequisite history")
    with tempfile.TemporaryDirectory(prefix="lrc-original-publication-") as tmp:
        temp = Path(tmp)
        bundle = temp / "recovered-originals.bundle"
        with bundle.open("xb") as output:
            for i, part in enumerate(manifest["bundle"]["parts"]):
                if part["name"] != f"recovered-originals.bundle.part{i:03d}":
                    raise RuntimeError("Unexpected bundle part name or order")
                data = (HERE / part["name"]).read_bytes()
                if (len(data) != part["bytes"]
                        or hashlib.sha256(data).hexdigest() != part["sha256"]):
                    raise RuntimeError(f"Bundle part checksum mismatch: {part['name']}")
                output.write(data)
        with bundle.open("rb") as stream:
            if (bundle.stat().st_size != manifest["bundle"]["bytes"]
                    or hashlib.file_digest(stream, "sha256").hexdigest() != BUNDLE_SHA256):
                raise RuntimeError("Reconstructed bundle checksum mismatch")

        recovered = temp / "recovered.git"
        run("git", "init", "--bare", "--quiet", str(recovered))
        git(recovered, "fetch", "--quiet", "--update-shallow", "--no-tags", str(source),
            *[f"{sha}:refs/publication-prerequisites/{i}"
              for i, sha in enumerate(PREREQUISITES)])
        git(recovered, "bundle", "verify", str(bundle))
        listed = parse_refs(git(recovered, "bundle", "list-heads", str(bundle)))
        if listed != EXPECTED_REFS:
            raise RuntimeError("Bundle refs differ from the recorded originals")
        git(recovered, "fetch", "--quiet", "--no-tags", str(bundle),
            *[f"{ref}:{ref}" for ref in EXPECTED_REFS])
        for ref, expected in EXPECTED_REFS.items():
            if git(recovered, "rev-parse", ref) != expected:
                raise RuntimeError(f"Original commit mismatch: {ref}")
        git(recovered, "fsck", "--connectivity-only", "--no-reflogs", REPAIR, HANDOFF)
        git(recovered, "merge-base", "--is-ancestor",
            "6c3976efc2382e28731009f9637ecf31872e5c13", REPAIR)
        for name, expected in EXPECTED_ARTIFACTS.items():
            payload = subprocess.run(
                ["git", "-C", str(recovered), "cat-file", "blob",
                 f"{HANDOFF}:{ARTIFACT_ROOT}/{name}"],
                check=True, capture_output=True, timeout=30).stdout
            if (len(payload) != expected["bytes"]
                    or hashlib.sha256(payload).hexdigest() != expected["sha256"]):
                raise RuntimeError(f"Recovered artifact checksum mismatch: {name}")

        result = {"status": "verified", "refs": EXPECTED_REFS,
                  "artifacts": EXPECTED_ARTIFACTS,
                  "bundle_sha256": BUNDLE_SHA256}
        if args.publish:
            git(recovered, "remote", "add", "origin", f"https://github.com/{REPOSITORY}.git")
            remote = parse_refs(remote_git(recovered, "ls-remote", "--heads", "origin",
                                           *EXPECTED_REFS))
            for ref, current in remote.items():
                if current != EXPECTED_REFS[ref]:
                    raise RuntimeError(f"Refusing to change a differing remote branch: {ref}")
            missing = [ref for ref in EXPECTED_REFS if ref not in remote]
            if missing:
                # Empty leases are create-only guards: if any branch appeared
                # after the read, Git rejects the entire atomic publication.
                remote_git(recovered, "push", "--atomic", "--porcelain",
                           *[f"--force-with-lease={ref}:" for ref in missing],
                           "origin", *[f"{EXPECTED_REFS[ref]}:{ref}" for ref in missing])
            published = parse_refs(remote_git(recovered, "ls-remote", "--heads", "origin",
                                              *EXPECTED_REFS))
            if published != EXPECTED_REFS:
                raise RuntimeError("Remote verification did not match both original refs")
            result["status"] = "published"
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        # Our subprocess arguments contain no secret values. Server diagnostics
        # are needed to distinguish object-integrity and permission failures.
        print(exc.stderr if isinstance(exc.stderr, str) else "Git operation failed")
        raise
