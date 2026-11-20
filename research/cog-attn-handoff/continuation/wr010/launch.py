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
             name="WR010 "+arm,description="Constructed cache invalidation diagnostic",
             command=[sys.executable,"workload.py",arm], seed=181,timeout_seconds=60,
             source_kind="CONSTRUCTED",split_identity="WR010-authority-v1",
             inputs=["workload.py","grounding.py", *sorted(x.name for x in recipes.glob("*.npz")), *sorted(x.name for x in recipes.glob("data-*.json"))],artifacts=["results.json","responses.sha256"],metrics={"score":"max"},
             budget={"fitted_models":0,"events_per_cell":1536})
    (recipes/(arm+".json")).write_text(json.dumps(cfg,indent=2))
protocol=dict(schema="research_lab.protocol.v1",id="WR010-authority",program="wr-continuation",
              author="codex",question="Does synchronous write-side invalidation improve mixed-workload throughput?",
              hypothesis="Lazy invalidation outperforms semantic-only caching at 10 percent writes without changing learned competence.",
              alternatives=["Fresh resolution is cheaper throughout.","Read savings persist even with mostly writes."],
              predictions=["Primary throughput increases by at least 1000 operations per second.","Lost or late notifications can return stale claims. A synchronously observed object version should restore agreement."],
              controls=["Same checkpoint fresh, semantic-only, eager and unassisted; matched clarification for all assisted arms."],
              ablations=["1, 10, 50 and 90 percent writes; 8-step delay and 25 percent message loss; authoritative-version rescue."],
              metric="score",direction="max",minimum_effect=1000,seeds=[181,193,211],
              resource="wr-cpu",slots=1,max_attempts=1,stopping_rule="Six jobs, three timing repetitions per cell.",
              scope="ENGINEERING_ONLY",priority=1,source_commit="eec439492407b0aa26922c23b5ae40d95289f4eb",
              next_test="Replace the simulated clarification source and measure remote version-read costs.",
              limitations=["Constructed public cases, frozen learned action model, simulated truthful clarification, single-process in-memory transactions and version access."],
              candidate="candidate.json",control="control.json",replication_tolerance=100000,
              replay_artifacts=["responses.sha256"])
(recipes/'protocol.json').write_text(json.dumps(protocol,indent=2))
lab.register(protocol,recipes)
put_node(lab,"wr010.authority-protocol","experiment",protocol)
lab.worker("wr010-local","experiment","wr-cpu")
run_worker(lab,"wr010-local")
(root/'lab-status.json').write_text(json.dumps(lab.status(),indent=2,default=str))
