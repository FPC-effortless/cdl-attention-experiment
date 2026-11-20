import json
import sys
from pathlib import Path
from research_lab.core import Controller
from research_lab.execution import run_worker
from research_lab.evidence import put_node

ROOT=Path(__file__).resolve().parent
RECIPES=ROOT/"recipe"
lab=Controller(ROOT/"ecosystem/research_lab")
lab.program("weakness-repairs","Repair grounding, audit observability and durable state",3600,0)
lab.resource("repair-cpu","cpu",1,0)
specs={
 "language":("Does transition-only contextual pointer training ground held-out language?",
             "Support transitions improve novel language grounding and the Transformer comparison meets the competence gate.",
             ["Held syntax exposes role mistakes.", "Support helps novel verbs and may not generalize to new templates."]),
 "epistemic":("Which silent shifts are detectable with a fixed anchor budget?",
              "Independent intervention anchors bound delay where event checks lack a shift signal.",
              ["Event-only misses silent shifts.","Shared bias defeats redundant passive anchors and biased operator validation."]),
 "storage":("Do atomic guards and retained histories preserve concurrent state?",
            "Transactional validation rejects stale bindings and bounded bitemporal compaction preserves supported queries.",
            ["Split check/commit commits stale writes.","Retained history reduces disk footprint at an explicit query horizon."])
}
for suite,(question,hypothesis,predictions) in specs.items():
    for arm in ("candidate","control"):
        cfg=dict(schema="trainlab.experiment.v1",execution_purpose="engineering_diagnostic",
                 name=f"{suite}-{arm}",description=question,
                 command=[sys.executable,"entry.py",suite,arm],seed=321,timeout_seconds=240,
                 source_kind="CONSTRUCTED",split_identity=f"weakness-{suite}-v1",inputs=["entry.py",f"{suite}.py"],
                 artifacts=["results.json","result.sha256"]+(["base.npz","adapted.npz"] if suite=="language" else []),
                 metrics={"score":"max"},budget={"data_seeds":3})
        (RECIPES/f"{suite}-{arm}.json").write_text(json.dumps(cfg,indent=2))
    protocol=dict(schema="research_lab.protocol.v1",id=f"weakness-{suite}",program="weakness-repairs",
                  author="codex",question=question,hypothesis=hypothesis,alternatives=["Apparent gains depend on supplied structure.","The control remains competitive."],
                  predictions=predictions,controls=["Same frozen seeds and declared data."],
                  ablations=["See the program's protocol and executable experiment."],
                  metric="score",direction="max",minimum_effect=0,seeds=[321,337,353],
                  resource="repair-cpu",slots=1,max_attempts=1,stopping_rule="Execute all six declared jobs once.",
                  scope="ENGINEERING_ONLY",priority=1,source_commit="eec439492407b0aa26922c23b5ae40d95289f4eb",
                  next_test="Replicate on a fresh external environment after the frozen gate.",
                  limitations=["Constructed fixtures and same-host evaluation."],
                  candidate=f"{suite}-candidate.json",control=f"{suite}-control.json",replication_tolerance=0,
                  replay_artifacts=["result.sha256"])
    (RECIPES/f"{suite}-protocol.json").write_text(json.dumps(protocol,indent=2))
    lab.register(protocol,RECIPES)
    put_node(lab,f"repair.{suite}.protocol","experiment",protocol)
lab.worker("repair-local","experiment","repair-cpu")
run_worker(lab,"repair-local")
(ROOT/"lab-status.json").write_text(json.dumps(lab.status(),indent=2,default=str))
