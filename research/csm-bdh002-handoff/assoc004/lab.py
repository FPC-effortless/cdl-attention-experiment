"""Run the ASSOC-004 development panel and retain local lab evidence."""
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
put_node(lab,"assoc004.protocol","experiment",{"protocol":(here/"README.md").read_text()})
with tempfile.TemporaryDirectory() as tmp:
    stage = Path(tmp)
    shutil.copy(here.parent/"assoc002/panel.py",stage)
    for file in ("retrieve.py","README.md"):
        shutil.copy(here/file,stage)
    recipe = dict(schema="trainlab.experiment.v1",name="ASSOC-004 development",
                  description="Compare redundant bank routing with matched exact-dot cap",
                  execution_purpose="engineering_diagnostic", command=["python","retrieve.py"], seed=0,
                  timeout_seconds=120,source_kind="CONSTRUCTED",split_identity="disjoint-stores-v1",
                  inputs=["retrieve.py","panel.py","README.md"],artifacts=["scores.csv","cues.csv.gz"],metrics={"panels":"max"},
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
    producer=dict(producer_id="assoc004-local",producer_version="1",content_sha256=digest(here/"lab.py")),
    artifacts=[dict(artifact_id=k,content_sha256=v) for k,v in run["artifacts"].items()],
    observed_at=datetime.fromisoformat(run["finished_at"]),provenance={"qualification":"NOT_QUALIFIED"})
body=evidence.model_dump(mode="json")
write_json(root/"evidence.json",body)
put_node(lab,"assoc004.panel","finding",body)
link(lab,"assoc004.panel","tests","assoc004.protocol")
link(lab,"assoc004.panel","derived_from","trainlab."+run["id"])
