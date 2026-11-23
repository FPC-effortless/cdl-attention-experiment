"""Prepare an explicit source/review tree; binary evidence is delivered separately."""
from pathlib import Path
import hashlib
import json
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STAGING = REPO.parent


def main():
    selected = []
    for path in HERE.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(HERE)
        if any(part in ("__pycache__", ".pytest_cache", "calibration", "reproduction_inputs") or part.startswith("benchmark-work-") for part in relative.parts):
            continue
        name = relative.as_posix()
        choose = ((path.suffix in (".py", ".md", ".txt", ".xml") and not name.startswith("evidence/"))
                  or path.name == ".gitignore")
        if path.suffix == ".json" and path.name not in ("corpus.json", "CAMPAIGN_JOBS.json"):
            choose = choose or (("runs/" not in name and "evaluation/" not in name)
                                or path.name in ("RESULT.json", "IDENTITY.json"))
        if relative.parts[0] == "interactions" or path.suffix in (".sqlite", ".pt", ".npz", ".zip"):
            choose = False
        if path.name in ("final_engineering_tests.xml", "final_engineering_tests.txt"):
            choose = True
        if choose:
            selected.append(path)
    entries = [{"path": str(path.relative_to(REPO)), "mode": "100644", "type": "blob", "content": path.read_text()}
               for path in sorted(selected)]
    payload = {"repository_full_name": "FPC-effortless/cdl-attention-experiment",
               "base_tree_sha": "697309694762db28cb828b60d81969446bcf7c72", "tree_elements": entries}
    destination = STAGING / "publication_tree.json"
    destination.write_text(json.dumps(payload))
    pathspec = STAGING / "publication_paths.nul"
    pathspec.write_bytes(b"".join(str(p.relative_to(REPO)).encode() + b"\0" for p in sorted(selected)))
    receipt = {"files": len(selected), "payload_bytes": destination.stat().st_size,
               "source_files": {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(selected)}}
    (STAGING / "publication_files.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"files": len(selected), "payload_bytes": destination.stat().st_size, "path": str(destination)}))


if __name__ == "__main__":
    main()
