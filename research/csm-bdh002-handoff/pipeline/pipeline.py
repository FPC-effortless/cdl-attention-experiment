"""Run a pinned CSM historical replay using the Woyengi ecosystem."""
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from forge_data.core import canonical, release
from investigation_world.evidence import EvidenceRecord
from research_lab.adapters import import_forge, import_trainlab
from research_lab.core import Controller
from research_lab.evidence import link, put_node, retrieve
from trainlab.runner import digest, run_experiment, verify_run, write_json

from replay import STEPS

HERE = Path(__file__).resolve().parent
SOURCE = "b7bbfd52b6ffeb339eed471ecc1f860bd04b2fe9"
ECOSYSTEM = "eec439492407b0aa26922c23b5ae40d95289f4eb"
PREFIX = "research/csm-bdh002-handoff/"
DEPS = {"study041": ["study040"], "skill002": ["skill001"], "skill003": ["skill001", "skill002"],
        "skill004": ["skill001", "skill002"], "skill005": ["skill001", "skill002", "skill004"]}


def git(*args):
    return subprocess.check_output(["git", "-C", str(HERE), *args])


def prepare(name, stage):
    folders = [name, *DEPS.get(name, [])]
    files = git("ls-tree", "-r", "--full-tree", "--name-only", SOURCE, "--",
                *[":(top)" + PREFIX + folder for folder in folders],
                ":(top)" + PREFIX + "requirements.txt").decode().splitlines()
    for path in files:
        dest = stage / path.removeprefix(PREFIX)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(git("show", SOURCE + ":" + path))
    shutil.copy(HERE / "replay.py", stage)
    return sorted(str(path.relative_to(stage)) for path in stage.rglob("*") if path.is_file())


def verify_outputs(name, output):
    report = {}
    for _, first, second in STEPS[name]:
        for file in filter(None, (first, second)):
            artifact = output / file
            if file == "timing.csv":
                with artifact.open() as stream:
                    rows = list(csv.DictReader(stream))
                expected = {(b, a) for b in range(30) for a in ("primitive", "deferred", "with_swap")}
                assert len(rows) == 90 and {(int(r["block"]), r["arm"]) for r in rows} == expected
                assert all(int(r["total_ns"]) > 0 for r in rows)
                check = "90 positive timing measurements"
            elif file == "scope.txt":
                assert artifact.read_text().strip() == "Passed 10 controlled scope checks."
                check = "10 regression checks"
            else:
                assert artifact.read_bytes() == git("show", SOURCE + ":" + PREFIX + name + "/" + file)
                check = "byte-exact pinned Git output"
            report[file] = {"check": check, "sha256": digest(artifact)}
    return report


def main(root):
    root.mkdir(parents=True, exist_ok=False)
    lab = Controller(root / "research_lab")
    with tempfile.TemporaryDirectory() as temp:
        staging = Path(temp)
        inventory = []
        inputs = {}
        for name in STEPS:
            stage = staging / name
            inputs[name] = prepare(name, stage)
            inventory.append(dict(episode_id=name, subject_id=name, record_id=name,
                                  observed_at="2026-11-18T00:00:00Z", available_at="2026-11-18T00:00:00Z",
                                  observation={"input_hashes": {f: digest(stage / f) for f in inputs[name]}},
                                  action=None, outcome={"status": "unknown", "verified": False}))
        # This constructed release contains only our generated filename/digest inventory.
        spec = dict(dataset_id="csm-replay-inventory", source_id="constructed/csm-inventory",
                    source_version=SOURCE, adapter="operations-jsonl", license_id="generated-factual-digests",
                    provenance_uri="https://github.com/FPC-effortless/cdl-attention-experiment",
                    ai_use_allowed=True, redistribution_allowed=True, privacy_reviewed=True, split_salt=SOURCE)
        write_json(root / "inventory-spec.json", spec)
        (root / "inventory.jsonl").write_bytes(b"".join(canonical(row) for row in inventory))
        release(root / "inventory.jsonl", root / "inventory-spec.json", root / "forge")
        pin = digest(root / "forge/manifest.json")
        import_forge(lab, root / "forge", pin)
        for row in inventory:
            name = row["episode_id"]
            stage = staging / name
            assert all(digest(stage / f) == h for f, h in row["observation"]["input_hashes"].items())
            artifacts = [f for _, primary, secondary in STEPS[name] for f in (primary, secondary) if f]
            recipe = dict(schema="trainlab.experiment.v1", name=name, description="Replay pinned CSM outputs",
                          execution_purpose="historical_replay", command=["python", "replay.py", name],
                          seed=0, timeout_seconds=600, source_kind="CONSTRUCTED", split_identity=SOURCE + "/" + name,
                          inputs=inputs[name], artifacts=artifacts, metrics={"steps_completed": "max"},
                          budget={"declared_recipe_steps": len(STEPS[name])})
            write_json(stage / "recipe.json", recipe)
            run = run_experiment(stage / "recipe.json", root / "runs")
            directory = root / "runs" / run["id"]
            if run["status"] != "succeeded":
                raise RuntimeError(f"{name}: inspect {directory}")
            assert verify_run(directory)["valid"]
            import_trainlab(lab, directory)
            checks = verify_outputs(name, directory / "output")
            evidence = EvidenceRecord(
                evidence_type="csm.historical_replay", outcome="OBSERVED", visibility="public",
                claim="Pinned CSM historical outputs reproduced under TrainLab.",
                subjects=[dict(kind="run", subject_id=run["id"], content_sha256=digest(directory / "run.json")),
                          dict(kind="inventory", subject_id="csm-replay-inventory", content_sha256=pin)],
                producer=dict(producer_id="csm-replay-verifier", producer_version="1", content_sha256=digest(Path(__file__))),
                artifacts=[dict(artifact_id=name+"."+f, content_sha256=v["sha256"]) for f,v in checks.items()],
                observed_at=datetime.fromisoformat(run["finished_at"]),
                provenance=dict(source=SOURCE, ecosystem=ECOSYSTEM, checks=checks, qualification="NOT_QUALIFIED"))
            envelope = evidence.model_dump(mode="json")
            write_json(root / (name + ".evidence.json"), envelope)
            put_node(lab, name, "finding", envelope)
            link(lab, name, "derived_from", "trainlab." + run["id"])
            link(lab, name, "derived_from", "forge." + pin[:24])
            print(name, run["id"], flush=True)
        write_json(root / "graph.json", retrieve(lab))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
