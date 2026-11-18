"""Freeze an admitted SKILL-006 recipe or execute it in a fresh Research Lab OS."""
import argparse
from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from investigation_world.evidence import EvidenceRecord
from research_lab.adapters import import_trainlab
from research_lab.core import Controller
from research_lab.evidence import link, put_node
from trainlab.runner import digest, read_json, run_experiment, verify_run, write_json

from admission import prepare

HERE = Path(__file__).resolve().parent
FILES = ("research.py", "admission.py", "diagnostic.py", "README.md", "REVIEW.md", "development.zip")
FROZEN = ("frozen.json", "recipe.json", "preflight.json", "adequacy.json")
METRICS = ("primitive", "swap", "rotate", "learned", "longest", "hindsight")


def sources(stage):
    for name in FILES:
        shutil.copyfile(HERE / name, stage / name)
    for study, name in (("skill001", "pilot.py"), ("skill002", "run.py")):
        shutil.copyfile(HERE.parent / study / name, stage / name)
    shutil.copyfile(HERE.parent / "requirements.txt", stage / "requirements.txt")


def freeze(stage):
    subprocess.run([sys.executable, "research.py", "fit", "frozen.json"], cwd=stage, check=True)
    recipe = dict(schema="trainlab.experiment.v1", name="csm-skill006-transfer",
                  execution_purpose="research", description="Frozen two-skill selector on complete six-record roster",
                  command=["python", "research.py", "evaluate", "frozen.json"], seed=0, timeout_seconds=1200,
                  source_kind="CONSTRUCTED", split_identity="train-all-3-4-5/evaluate-all-6",
                  inputs=[*FILES,"pilot.py","run.py","requirements.txt","frozen.json"],
                  artifacts=["summary.json","scores.csv","observations.json"],
                  metrics={metric:"min" for metric in METRICS},
                  budget={"training_goals":3408,"evaluation_goals":46656,"fits":1})
    review = dict(author="CSM-SKILL-006 research implementer",
                  reviewer="Codex scientific_review and pipeline_review (separate same-session subagents)",
                  basis="See pinned REVIEW.md for independent inspection of the four domains and conditions.")
    prepare(stage, recipe, review)
    for name in FROZEN:
        shutil.copyfile(stage / name, HERE / name)


def execute(stage, root):
    root.mkdir(parents=True, exist_ok=False)
    for name in FROZEN:
        shutil.copyfile(HERE / name, stage / name)
    lab = Controller(root / "research_lab")
    put_node(lab, "skill006.protocol", "experiment", {"recipe_sha256":digest(stage/"recipe.json"),
                                                     "review_sha256":digest(stage/"REVIEW.md")})
    record = run_experiment(stage/"recipe.json", root/"runs")
    directory = root/"runs"/record["id"]
    if record["status"] != "succeeded" or not verify_run(directory)["valid"]:
        raise RuntimeError(f"Research execution failed: inspect {directory}")
    assert record["claim_eligibility"] == "DECLARED_CONDITIONS_ONLY"
    import_trainlab(lab, directory)
    metrics = record["metrics"]
    passed = metrics["learned"] < min(metrics[k] for k in ("primitive", "swap", "rotate"))
    evidence = EvidenceRecord(evidence_type="csm.two_skill_transfer",outcome="OBSERVED",visibility="public",
        claim=f"The frozen router {'passed' if passed else 'missed'} the strict weighted-cost gate.",
        subjects=[dict(kind="run", subject_id=record["id"],content_sha256=digest(directory/"run.json"))],
        producer=dict(producer_id="csm-skill006",producer_version="1",content_sha256=digest(Path(__file__))),
        artifacts=[dict(artifact_id=k,content_sha256=v) for k,v in record["artifacts"].items()],
        observed_at=datetime.fromisoformat(record["finished_at"]),
        provenance={"qualification":"NOT_QUALIFIED","metrics":metrics})
    result=evidence.model_dump(mode="json")
    write_json(root/"evidence.json",result)
    put_node(lab,"skill006.transfer","finding",result)
    link(lab,"skill006.transfer","tests","skill006.protocol")
    link(lab,"skill006.transfer","derived_from","trainlab."+record["id"])
    print(record["id"],record["claim_eligibility"],metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode",choices=("freeze","run"))
    parser.add_argument("output",nargs="?",type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temporary:
        stage = Path(temporary)
        sources(stage)
        if args.mode == "freeze":
            freeze(stage)
        elif args.output is None:
            parser.error("run requires a fresh output directory")
        else:
            execute(stage,args.output.resolve())
