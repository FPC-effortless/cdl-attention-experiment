"""Register the protocol and run the development panel in a new local lab."""
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
protocol = here / "README.md"
# Pin the protocol document independently of later results notes.
put_node(lab, "skill006.protocol", "experiment", {"document":protocol.read_text(),"sha256":digest(protocol)})
with tempfile.TemporaryDirectory() as tmp:
    stage = Path(tmp)
    for file in (here/"diagnostic.py",here.parent/"skill001/pilot.py",here.parent/"skill002/run.py"):
        shutil.copy(file,stage)
    recipe = dict(schema="trainlab.experiment.v1", execution_purpose="engineering_diagnostic",
                  name="skill006-development", description="Check constructed two-skill benchmark headroom",
                  command=["python","diagnostic.py"], seed=0, timeout_seconds=60, source_kind="CONSTRUCTED",
                  split_identity="development-four-record-exhaustive", inputs=["diagnostic.py","pilot.py","run.py"],
                  artifacts=["panel.csv","programs.json"], metrics={k:"min" for k in ("primitive","swap","rotate","hindsight")},
                  budget={"development_goals":256})
    write_json(stage/"recipe.json",recipe)
    run = run_experiment(stage/"recipe.json",root/"runs")
directory = root/"runs"/run["id"]
assert run["status"]=="succeeded" and verify_run(directory)["valid"]
import_trainlab(lab,directory)
evidence = EvidenceRecord(evidence_type="csm.development_panel",outcome="OBSERVED",visibility="public",
    claim="Four-record constructed benchmark executed and checked for headroom.",
    subjects=[dict(kind="run",subject_id=run["id"],content_sha256=digest(directory/"run.json"))],
    producer=dict(producer_id="skill006-development",producer_version="1",content_sha256=digest(here/"lab.py")),
    artifacts=[dict(artifact_id=k,content_sha256=v) for k,v in run["artifacts"].items()],
    observed_at=datetime.fromisoformat(run["finished_at"]),provenance={"qualification":"NOT_QUALIFIED","protocol":digest(protocol)})
body=evidence.model_dump(mode="json")
write_json(root/"evidence.json",body)
put_node(lab,"skill006.development","finding",body)
link(lab,"skill006.development","tests","skill006.protocol")
link(lab,"skill006.development","derived_from","trainlab."+run["id"])
print(run["metrics"])
