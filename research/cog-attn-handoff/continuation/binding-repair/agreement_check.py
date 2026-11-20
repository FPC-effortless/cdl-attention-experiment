import json
import sys
from collections import defaultdict
from pathlib import Path

from research_lab.core import Controller
from research_lab.evidence import put_node
from trainlab.runner import verify_run

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"recipe"))
from admission import admitted_action
from binding import COHORTS,rows,words_for_seed

LAB=ROOT/"ecosystem/research_lab"
actions=[(op,src,dst) for op in range(4) for src in range(3) for dst in range(3) if src!=dst]
summary=defaultdict(lambda:dict(n=0,admitted=0,correct=0))
for path in (LAB/"runs").glob("*/output/results.json"):
    run=path.parents[1]
    config=json.loads((run/"experiment.json").read_text())
    if config["command"][-2:]!=["binding","candidate"]:
        continue
    assert verify_run(run)["valid"]
    seed=config["seed"]
    local,sequence=json.loads(path.read_text())
    for i,(first,second) in enumerate(zip(local["rows"],sequence["rows"],strict=True)):
        name=COHORTS[i]
        counts=summary[name]
        actual=rows(seed+7100+i,words_for_seed(seed),512,name)
        for a,p,b,q,row in zip(first["predictions"],first["confidence"],second["predictions"],second["confidence"],actual,strict=True):
            commit=admitted_action(a,p,b,q)
            counts["n"]+=1
            if commit is not None:
                counts["admitted"]+=1
                if name!="ambiguous":
                    counts["correct"]+=actions[commit]==tuple(row[4:])
assert all(c["n"]==1536 for c in summary.values()) and len(summary)==6
(ROOT/"agreement.json").write_text(json.dumps(summary,indent=2))
put_node(Controller(LAB),"binding.repair.agreement-replay","evidence",
         dict(scope="ENGINEERING_ONLY",design="post-hoc saved-output comparison",summary=dict(summary)))
