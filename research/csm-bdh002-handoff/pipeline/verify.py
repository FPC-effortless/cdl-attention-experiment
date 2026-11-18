"""Recheck a saved CSM pipeline run from the pinned Git reference."""
from pathlib import Path
import sys

from forge_data.dataset import Dataset
from investigation_world.evidence import EvidenceRecord
from research_lab.core import Controller
from research_lab.evidence import retrieve
from trainlab.runner import digest, read_json, verify_run

from pipeline import ECOSYSTEM, SOURCE, STEPS, verify_outputs


def verify(root):
    if not __debug__:
        raise RuntimeError("Run with Python assertions enabled")
    inventory_pin = digest(root / "forge/manifest.json")
    Dataset(root / "forge", manifest_sha256=inventory_pin)
    graph = retrieve(Controller(root / "research_lab"))
    nodes = {node["id"]: node["body"] for node in graph["nodes"]}
    links = {(link["source"], link["relation"], link["target"]) for link in graph["links"]}
    seen = set()
    for directory in sorted((root / "runs").iterdir()):
        run = read_json(directory / "run.json")
        name = run["name"]
        assert name in STEPS and name not in seen
        seen.add(name)
        assert verify_run(directory)["valid"] and run["status"] == "succeeded"
        checks = verify_outputs(name, directory / "output")
        envelope = read_json(root / (name + ".evidence.json"))
        evidence = EvidenceRecord.model_validate(envelope)
        assert evidence.provenance["checks"] == checks
        assert evidence.provenance["source"] == SOURCE and evidence.provenance["ecosystem"] == ECOSYSTEM
        assert evidence.producer.content_sha256 == digest(Path(__file__).with_name("pipeline.py"))
        assert {s.kind: s.content_sha256 for s in evidence.subjects} == {
            "inventory": inventory_pin, "run": digest(directory / "run.json")}
        assert {a.artifact_id: a.content_sha256 for a in evidence.artifacts} == {
            name + "." + f: v["sha256"] for f, v in checks.items()}
        assert nodes[name] == envelope
        for target in ("trainlab." + run["id"], "forge." + inventory_pin[:24]):
            assert (name, "derived_from", target) in links
    assert seen == set(STEPS)
    print("Verified six historical runs, pinned outputs, evidence bindings and lab graph.")


if __name__ == "__main__":
    verify(Path(sys.argv[1]).resolve())
