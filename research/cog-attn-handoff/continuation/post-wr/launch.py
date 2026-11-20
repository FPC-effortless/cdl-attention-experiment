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
             name="PostWR "+arm,description="Constructed cache invalidation diagnostic",
             command=[sys.executable,"trial.py",arm], seed=181,timeout_seconds=180,
             source_kind="CONSTRUCTED",split_identity="PostWR-authority-v1",
             inputs=["trial.py","attention.py","workload.py","grounding.py", *sorted(x.name for x in recipes.glob("*.npz")), *sorted(x.name for x in recipes.glob("data-*.json"))],artifacts=["results.json","responses.sha256"],metrics={"score":"max"},
             budget={"fitted_models":0,"events_per_cell":1536})
    (recipes/(arm+".json")).write_text(json.dumps(cfg,indent=2))
protocol=dict(schema="research_lab.protocol.v1",id="PostWR-authority",program="wr-continuation",
              author="codex",question="Can transactional verification contain grounding errors at measured service latency, and how does a trained KV decoder compare?",
              hypothesis="A stricter stage gate improves downstream accuracy while increasing verification work.",
              alternatives=["High-confidence errors pass both thresholds.","Distilled decoder accuracy differs materially at the shared threshold."],
              predictions=["More requests at the higher gate.","Confirm-all recovers gold state."],
              controls=["Same state executor, lazy read path, semantic memoization, truthful local process and seeded workloads."],
              ablations=["Blocking 0.80; staged 0.995; confirm-all WR upper bound."],
              metric="score",direction="max",minimum_effect=0,seeds=[181,193,211],
              resource="wr-cpu",slots=1,max_attempts=1,stopping_rule="Six jobs, three timing repetitions per cell.",
              scope="ENGINEERING_ONLY",priority=1,source_commit="eec439492407b0aa26922c23b5ae40d95289f4eb",
              next_test="Replace the simulated clarification source and measure remote version-read costs.",
              limitations=["Constructed public cases, frozen learned action model, simulated truthful clarification, single-process in-memory transactions and version access."],
              candidate="candidate.json",control="control.json",replication_tolerance=100000,
              replay_artifacts=["responses.sha256"])
(recipes/'protocol.json').write_text(json.dumps(protocol,indent=2))
lab.register(protocol,recipes)
put_node(lab,"postwr.authority-protocol","experiment",protocol)
lab.worker("postwr-local","experiment","wr-cpu")
run_worker(lab,"postwr-local")
(root/'lab-status.json').write_text(json.dumps(lab.status(),indent=2,default=str))
