import json
import sys
from pathlib import Path
from research_lab.core import Controller
from research_lab.evidence import put_node
from research_lab.execution import run_worker

ROOT=Path(__file__).resolve().parent
RECIPES=ROOT/"recipe"
lab=Controller(ROOT/"ecosystem/research_lab")
lab.program("delta-authority-048","Verify turn updates, acquisition attestations and retrained module composition",3600,0)
lab.resource("module-cpu","cpu",1,0)
for suite,question in (("bridge_scope","Can a learned scope selector make the retrained CSM route qualify?"),):
    source=["entry_scope.py","bridge.py","scope.py",suite+".py"]
    if suite=="bridge_scope":
        source+=["state_store.py","tac_identity.py"]
        source+=[str(p.relative_to(RECIPES)) for folder in ("original_csm","original_lrc","lrc_vb001_comparison")
                 for p in sorted((RECIPES/folder).rglob("*")) if p.is_file() and "__pycache__" not in p.parts]
    for arm in ("candidate","control"):
        cfg=dict(schema="trainlab.experiment.v1",execution_purpose="engineering_diagnostic",
                 name=f"{suite}-{arm}",description=question,
                 command=[sys.executable,"entry_scope.py",suite,arm],seed=1181,timeout_seconds=240,
                 source_kind="CONSTRUCTED",split_identity=f"delta-authority-{suite}-v1",inputs=source,
                 artifacts=["results.json","results.sha256"]+(["tac.pt","csm.npz"] if suite=="bridge_scope" else [])+(["scope.npz"] if arm=="candidate" else []),
                 metrics={"score":"max"},budget={"seeds":3})
        (RECIPES/f"{suite}-{arm}.json").write_text(json.dumps(cfg,indent=2))
    protocol=dict(schema="research_lab.protocol.v1",id=f"delta-authority-{suite}-verified",program="delta-authority-048",
                  author="codex",question=question,hypothesis="The provenance or learned module intervention improves the declared test.",
                  alternatives=["Acquisition dependence is unobservable.","Held syntax remains out of distribution."],
                  predictions=["Supported test results exceed the control.","Adversarial controls preserve visible failure modes."],
                  controls=["Same inputs, seeds and typed effect oracle."],
                  ablations=["See the frozen research protocol and fixture."],metric="score",direction="max",minimum_effect=0,
                  seeds=[1181,1201,1217],resource="module-cpu",slots=1,max_attempts=1,
                  stopping_rule="Run all six declared jobs once.",
                  scope="ENGINEERING_ONLY",priority=1,source_commit="eec439492407b0aa26922c23b5ae40d95289f4eb",
                  next_test="Evaluate on an independently acquired external authority and open-language task.",
                  limitations=["Simulated source issuance and typed bridge limited to declared operators."],
                  candidate=f"{suite}-candidate.json",control=f"{suite}-control.json",
                  replication_tolerance=0,replay_artifacts=["results.sha256"])
    (RECIPES/f"{suite}-protocol.json").write_text(json.dumps(protocol,indent=2))
    lab.register(protocol,RECIPES)
    put_node(lab,f"delta.authority.{suite}.verified.protocol","experiment",protocol)
lab.worker("scope-verified-local","experiment","module-cpu")
run_worker(lab,"scope-verified-local")
(ROOT/"lab-status-scope-verified.json").write_text(json.dumps(lab.status(),indent=2,default=str))
