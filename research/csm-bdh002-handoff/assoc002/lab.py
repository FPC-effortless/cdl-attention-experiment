"""Run the ASSOC-002 development panel and retain local lab evidence."""
import json
from datetime import datetime
from pathlib import Path
import shutil
import sys
import tempfile

from investigation_world.evidence import EvidenceRecord
from research_lab.adapters import import_trainlab
from research_lab.core import Controller
from research_lab.evidence import link, put_node
from trainlab.runner import digest, run_experiment, verify_run, write_json

here = Path(__file__).resolve().parent
root = Path(sys.argv[1]).resolve()
root.mkdir(parents=True, exist_ok=False)
lab = Controller(root / "lab")
put_node(lab,"assoc002.protocol","experiment",{"protocol":(here/"README.md").read_text()})
with tempfile.TemporaryDirectory() as tmp:
    stage = Path(tmp)
    for file in ("panel.py","README.md"):
        shutil.copy(here/file,stage)
    recipe = dict(schema="trainlab.experiment.v1",name="ASSOC-002 development",
                  description="Characterize selective lookup on constructed noisy association packets",
                  execution_purpose="engineering_diagnostic", command=["python","panel.py"], seed=0,
                  timeout_seconds=120,source_kind="CONSTRUCTED",split_identity="disjoint-stores-v1",
                  inputs=["panel.py","README.md"],artifacts=["panel.csv"],metrics={"panels":"max"},
                  budget={"evaluation_packets":4608})
    write_json(stage/"recipe.json",recipe)
    run = run_experiment(stage/"recipe.json",root/"runs")
directory = root/"runs"/run["id"]
if run["status"] != "succeeded" or not verify_run(directory)["valid"]:
    raise RuntimeError(str(directory))
import_trainlab(lab,directory)
evidence = EvidenceRecord(evidence_type="csm.associative_memory_diagnostic",outcome="OBSERVED",
    visibility="public",claim="Constructed noisy recall panel executed on three independently generated stores per load.",
    subjects=[dict(kind="run",subject_id=run["id"],content_sha256=digest(directory/"run.json"))],
    producer=dict(producer_id="assoc002-local",producer_version="1",content_sha256=digest(here/"lab.py")),
    artifacts=[dict(artifact_id=k,content_sha256=v) for k,v in run["artifacts"].items()],
    observed_at=datetime.fromisoformat(run["finished_at"]),provenance={"qualification":"NOT_QUALIFIED"})
body=evidence.model_dump(mode="json")
write_json(root/"evidence.json",body)
put_node(lab,"assoc002.panel","finding",body)
link(lab,"assoc002.panel","tests","assoc002.protocol")
link(lab,"assoc002.panel","derived_from","trainlab."+run["id"])
