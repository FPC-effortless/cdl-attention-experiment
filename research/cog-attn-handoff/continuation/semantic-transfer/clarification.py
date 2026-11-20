"""Offline replay of one-role clarification using fully specified holdout cases."""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from trainlab.runner import verify_run

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"recipe"))
from nli_transfer import fixture

ACTIONS=[(o,s,d) for o in range(4) for s in range(3) for d in range(3) if s!=d]
pools=defaultdict(lambda:dict(n=0,raw=0,one=0,full=0,questions=[0]*3, fixed_slot_correct=[0]*3))
for path in (ROOT/"ecosystem/research_lab/runs").glob("*/output/results.json"):
    run=path.parents[1]
    config=json.loads((run/"experiment.json").read_text())
    if config["command"][-2]!="nli":
        continue
    assert verify_run(run)["valid"]
    _,_,_,held=fixture(config["seed"])
    for item in json.loads(path.read_text()):
        for cohort in ("implicit","correction"):
            row=item["rows"][cohort]
            report=pools[(item["mode"],cohort)]
            for q,posterior in zip(held[cohort],row["posterior"],strict=True):
                truth=tuple(q[4:])
                assert abs(sum(posterior)-1)<1e-5
                uncertainty=[]
                for slot in range(3):
                    marg=[sum(p for a,p in zip(ACTIONS,posterior,strict=True) if a[slot]==x)
                          for x in range(4 if slot==0 else 3)]
                    uncertainty.append(-sum(p*math.log(p) for p in marg if p>0))
                slot=max(range(3),key=lambda j:uncertainty[j])
                unassisted=max(range(len(ACTIONS)),key=posterior.__getitem__)
                possible=[i for i,a in enumerate(ACTIONS) if a[slot]==truth[slot]]
                clarified=max(possible,key=posterior.__getitem__)
                report["n"]+=1
                report["raw"]+=ACTIONS[unassisted]==truth
                report["one"]+=ACTIONS[clarified]==truth
                report["full"]+=1
                report["questions"][slot]+=1
                for fixed in range(3):
                    alternatives=[i for i,a in enumerate(ACTIONS) if a[fixed]==truth[fixed]]
                    fixed_best=max(alternatives,key=posterior.__getitem__)
                    report["fixed_slot_correct"][fixed]+=ACTIONS[fixed_best]==truth
assert len(pools)==4 and all(d["n"]==384 for d in pools.values())
summary=[dict(mode=m,cohort=c,**v) for (m,c),v in sorted(pools.items())]
(ROOT/"clarification.json").write_text(json.dumps(summary,indent=2))
for row in summary:
    print(row)
