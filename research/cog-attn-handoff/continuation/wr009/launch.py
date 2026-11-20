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
             name="WR009 "+arm,description="Constructed cache invalidation diagnostic",
             command=[sys.executable,"workload.py",arm], seed=131,timeout_seconds=60,
             source_kind="CONSTRUCTED",split_identity="WR009-authority-v1",
             inputs=["workload.py"],artifacts=["results.json","responses.sha256"],metrics={"score":"max"},
             budget={"fitted_models":0,"events_per_cell":2048})
    (recipes/(arm+".json")).write_text(json.dumps(cfg,indent=2))
protocol=dict(schema="research_lab.protocol.v1",id="WR009-authority",program="wr-continuation",
              author="codex",question="Does synchronous write-side invalidation improve mixed-workload throughput?",
              hypothesis="The indexed cache wins at 10 percent writes with 128 distractors and pays maintenance costs at high write frequency.",
              alternatives=["Fresh resolution is cheaper throughout.","Read savings persist even with mostly writes."],
              predictions=["Primary throughput increases by at least 1000 operations per second.","Relative benefit falls with increasing write frequency."],
              controls=["Fresh public solver on the identical trace; independent verification of each saved response"],
              ablations=["1, 10, 50 and 90 percent write probabilities; zero or 128 extra records"],
              metric="score",direction="max",minimum_effect=1000,seeds=[131,149,167],
              resource="wr-cpu",slots=1,max_attempts=1,stopping_rule="Six jobs, five timing repetitions per cell.",
              scope="ENGINEERING_ONLY",priority=1,source_commit="eec439492407b0aa26922c23b5ae40d95289f4eb",
              next_test="Test delayed or missing write delivery.",
              limitations=["Constructed public cases, fixed reference solver and synchronous writes."],
              candidate="candidate.json",control="control.json",replication_tolerance=100000,
              replay_artifacts=["responses.sha256"])
(recipes/'protocol.json').write_text(json.dumps(protocol,indent=2))
lab.register(protocol,recipes)
put_node(lab,"wr009.authority-protocol","experiment",protocol)
lab.worker("wr009-local","experiment","wr-cpu")
run_worker(lab,"wr009-local")
(root/'lab-status.json').write_text(json.dumps(lab.status(),indent=2,default=str))
