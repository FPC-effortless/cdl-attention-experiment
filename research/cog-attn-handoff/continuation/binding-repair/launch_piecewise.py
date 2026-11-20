"""Register frozen recipes in Research Lab OS and run through TrainLab."""
import json
import sys
from pathlib import Path

from research_lab.core import Controller
from research_lab.evidence import put_node
from research_lab.execution import run_worker

ROOT=Path(__file__).resolve().parent
RECIPES=ROOT/"recipe"
lab=Controller(ROOT/"ecosystem/research_lab")
lab.program("binding-repair","Test role and verb transfer and recover from correlated training bias",3600,0)
lab.resource("binding-cpu","cpu",1,0)
suites={"bias_piecewise":("Can anchors recover from an unknown input-dependent offset?",
            ["Piecewise nuisance models recover substantially more rules.","Independent holdback catches some misspecified rules."])}
for suite,(question,predictions) in suites.items():
    for arm in ("candidate","control"):
        if suite=="binding":
            modes=("local","sequence") if arm=="candidate" else ("pointer","transformer")
        else:
            modes=()
        recipe=dict(schema="trainlab.experiment.v1",execution_purpose="engineering_diagnostic",
                    name=f"{suite}-{arm}",description=question,
                    command=[sys.executable,"entry_piecewise.py",suite,arm],
                    seed=521,timeout_seconds=240,source_kind="CONSTRUCTED",
                    split_identity=f"binding-repair-{suite}-v1",
                    inputs=["entry.py","binding.py","language.py","bias.py","epistemic.py","bias_piecewise.py","entry_piecewise.py"],
                    artifacts=["results.json","results.sha256",*[m+".npz" for m in modes]],
                    metrics={"score":"max"},budget={"seeds":3})
        (RECIPES/f"{suite}-{arm}.json").write_text(json.dumps(recipe,indent=2))
    protocol=dict(schema="research_lab.protocol.v1",id=f"binding-repair-{suite}",program="binding-repair",
                  author="codex",question=question,hypothesis=predictions[0],alternatives=["The control transfers equally well.","The benefit requires supplied cue and noise classes."],
                  predictions=predictions,controls=["Shared data and seed, with joint baseline and matched ablations."],
                  ablations=["See frozen program protocol."],metric="score",direction="max",minimum_effect=0,
                  seeds=[521,547,569],resource="binding-cpu",slots=1,max_attempts=1,
                  stopping_rule="Run all six declared jobs once.",
                  scope="ENGINEERING_ONLY",priority=1,source_commit="eec439492407b0aa26922c23b5ae40d95289f4eb",
                  next_test="Replicate with independently written commands and an external anchor channel.",
                  limitations=["Constructed inventory and finite-program fixtures."],
                  candidate=f"{suite}-candidate.json",control=f"{suite}-control.json",
                  replication_tolerance=0,replay_artifacts=["results.sha256"])
    (RECIPES/f"{suite}-protocol.json").write_text(json.dumps(protocol,indent=2))
    lab.register(protocol,RECIPES)
    put_node(lab,f"binding.repair.{suite}.protocol","experiment",protocol)
lab.worker("bias-piecewise-local","experiment","binding-cpu")
run_worker(lab,"bias-piecewise-local")
(ROOT/"lab-status-piecewise.json").write_text(json.dumps(lab.status(),indent=2,default=str))
