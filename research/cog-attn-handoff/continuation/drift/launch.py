import json
import sys
from pathlib import Path
from research_lab.core import Controller
from research_lab.execution import run_worker
from research_lab.evidence import put_node

root=Path(__file__).resolve().parent
recipes=root/'recipe'
lab=Controller(root/'ecosystem/research_lab')
lab.program("wr-continuation","Test provenance-dependent invalidation in CompanyWorld",900,0)
lab.resource("wr-cpu","cpu",1,0)
for arm in ("candidate","control"):
    cfg=dict(schema="trainlab.experiment.v1",execution_purpose="engineering_diagnostic",
             name="Drift "+arm,description="Constructed cache invalidation diagnostic",
             command=[sys.executable,"drift_trial.py",arm], seed=181,timeout_seconds=60,
             source_kind="CONSTRUCTED",split_identity="Drift-authority-v1",
             inputs=["drift_trial.py","workload.py","grounding.py", *sorted(x.name for x in recipes.glob("*.npz")), *sorted(x.name for x in recipes.glob("data-*.json"))],artifacts=["results.json","responses.sha256"],metrics={"score":"max"},
             budget={"fitted_models":0,"events_per_cell":1536})
    (recipes/(arm+".json")).write_text(json.dumps(cfg,indent=2))
protocol=dict(schema="research_lab.protocol.v1",id="Drift-authority",program="wr-continuation",
              author="codex",question="When may an exact-command confirmation survive a change of meaning or authority?",
              hypothesis="Tracing the selector and selected provider epochs plus commit validation preserves all gold reads with fewer confirmations than global invalidation.",
              alternatives=["Uniform epoch invalidation is equally selective.","Use-time validation suffices during a mid-write change."],
              predictions=["No incorrect commits with full traced guard.","Precheck leaks raced writes; omitted selector leaks authority shifts."],
              controls=["Exact reuse, global invalidation and always-confirm with race retry."],
              ablations=["Omit selector; omit commit validation; change inactive source."],
              metric="score",direction="max",minimum_effect=0,seeds=[181,193,211],
              resource="wr-cpu",slots=1,max_attempts=1,stopping_rule="Six jobs, all three traces at 10, 50 and 90 percent writes.",
              scope="ENGINEERING_ONLY",priority=1,source_commit="eec439492407b0aa26922c23b5ae40d95289f4eb",
              next_test="Replace the simulated clarification source and measure remote version-read costs.",
              limitations=["Constructed public cases, frozen learned action model, simulated truthful clarification, single-process in-memory transactions and version access."],
              candidate="candidate.json",control="control.json",replication_tolerance=0,
              replay_artifacts=["responses.sha256"])
(recipes/'protocol.json').write_text(json.dumps(protocol,indent=2))
lab.register(protocol,recipes)
put_node(lab,"drift.authority-protocol","experiment",protocol)
lab.worker("drift-local","experiment","wr-cpu")
run_worker(lab,"drift-local")
(root/'lab-status.json').write_text(json.dumps(lab.status(),indent=2,default=str))
