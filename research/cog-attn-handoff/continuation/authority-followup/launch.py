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
             name="WR008 "+arm,description="Constructed cache invalidation diagnostic",
             command=[sys.executable,"cache_trial.py",arm], seed=101,timeout_seconds=60,
             source_kind="CONSTRUCTED",split_identity="WR008-authority-v1",
             inputs=["cache_trial.py"],artifacts=["predictions.json"],metrics={"score":"max"},
             budget={"fitted_models":0,"cases_per_seed":48})
    (recipes/(arm+".json")).write_text(json.dumps(cfg,indent=2))
protocol=dict(schema="research_lab.protocol.v1",id="WR008-authority",program="wr-continuation",
              author="codex",question="Which updates invalidate a cached evidence-backed answer?",
              hypothesis="Resolving the current first applicable source before reuse recovers insertions and precedence changes.",
              alternatives=["Identity alone suffices.","All public changes require a refresh."],
              predictions=["Supersession and reordered authorities refresh with the candidate.","Projection-only updates reuse answers."],
              controls=["Citation-only guard and freshly evaluated reference"],
              ablations=["Hidden override with identical public payload; lower priority and unrelated insertions"],
              metric="score",direction="max",minimum_effect=.20,seeds=[101,113,127],
              resource="wr-cpu",slots=1,max_attempts=1,stopping_rule="Run all six arm-seed cells once.",
              scope="ENGINEERING_ONLY",priority=1,source_commit="eec439492407b0aa26922c23b5ae40d95289f4eb",
              next_test="Measure the cost of maintaining source precedence versus fresh resolution.",
              limitations=["Constructed public cases, fixed reference solver and a seeded cache."],
              candidate="candidate.json",control="control.json",replication_tolerance=0,
              replay_artifacts=["predictions.json"])
(recipes/'protocol.json').write_text(json.dumps(protocol,indent=2))
lab.register(protocol,recipes)
put_node(lab,"wr008.authority-protocol","experiment",protocol)
lab.worker("wr008-local","experiment","wr-cpu")
run_worker(lab,"wr008-local")
(root/'lab-status.json').write_text(json.dumps(lab.status(),indent=2,default=str))
