"""Check the published two-skill receipt, score aggregation, and Research Lab graph."""
import csv
from collections import Counter
from itertools import product
from pathlib import Path
import sys

from investigation_world.evidence import EvidenceRecord
from research_lab.core import Controller
from research_lab.evidence import retrieve
from trainlab.runner import digest, read_json, verify_run


def verify(root):
    directory, = (root/"runs").iterdir()
    record = read_json(directory/"run.json")
    assert verify_run(directory)["valid"] and record["claim_eligibility"] == "DECLARED_CONDITIONS_ONLY"
    assert record["status"] == "succeeded"
    assert read_json(directory/"execution_adequacy.json")["status"] == "MET"
    totals = Counter()
    with (directory/"output/scores.csv").open() as stream:
        rows = csv.DictReader(stream)
        for target, row in zip(product(range(6),repeat=6), rows, strict=True):
            assert row["goal"] == "".join(map(str,target))
            assert row["learned"] == row[row["selected"]]
            assert row["longest"] == row[row["longest_selected"]]
            assert int(row["hindsight"]) == min(int(row[k]) for k in ("primitive","swap","rotate"))
            totals.update({k:int(row[k]) for k in record["metrics"]})
    assert totals == record["metrics"]
    assert read_json(directory/"experiment.json") == read_json(Path(__file__).with_name("recipe.json"))
    raw = read_json(root/"evidence.json")
    evidence = EvidenceRecord.model_validate(raw)
    assert evidence.subjects[0].content_sha256 == digest(directory/"run.json")
    assert {a.artifact_id:a.content_sha256 for a in evidence.artifacts} == record["artifacts"]
    assert evidence.provenance["metrics"] == record["metrics"]
    graph = retrieve(Controller(root/"research_lab"))
    assert {n["id"]:n["body"] for n in graph["nodes"]}["skill006.transfer"] == raw
    assert any(edge["source"] == "skill006.transfer" and edge["target"] == "trainlab."+record["id"]
               for edge in graph["links"])
    print("Validated the research receipt, 46,656 goal rows, totals, evidence and lab finding.")


if __name__ == "__main__":
    verify(Path(sys.argv[1]))
