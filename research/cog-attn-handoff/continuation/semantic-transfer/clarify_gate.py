"""Exploratory selective release after a single truthful slot response."""
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
GRID=(.50,.7,.8,.9,.95,.98,.99,.995,.999,.9999,1.01)


def resolved(raw, row):
    output=[]
    for item,p in zip(raw,row["posterior"],strict=True):
        truth=tuple(item[4:])
        entropy=[]
        for slot,n in enumerate((4,3,3)):
            masses=[sum(prob for a,prob in zip(ACTIONS,p,strict=True) if a[slot]==i) for i in range(n)]
            entropy.append(-sum(q*math.log(q) for q in masses if q>0))
        slot=max(range(3),key=entropy.__getitem__)
        choices=[i for i,a in enumerate(ACTIONS) if a[slot]==truth[slot]]
        selected=max(choices,key=p.__getitem__)
        # A response from outside the model's support is routed to full confirmation.
        mass=sum(p[i] for i in choices)
        output.append((p[selected]/mass if mass else 0.,ACTIONS[selected]==truth))
    return output


def totals(items,threshold):
    selected=[exact for confidence,exact in items if confidence>=threshold]
    return len(selected),sum(selected)


summary=[]
lab=ROOT/"ecosystem/research_lab"
for path in sorted((lab/"runs").glob("*/output/results.json")):
    run=path.parents[1]
    config=json.loads((run/"experiment.json").read_text())
    if config["command"][-2]!="nli":
        continue
    assert verify_run(run)["valid"]
    seed=config["seed"]
    _,_,cal,held=fixture(seed)
    for model in json.loads(path.read_text()):
        calibration=resolved(cal["clear"],model["calibration"]["clear"])
        choices=[]
        for threshold in GRID:
            n,good=totals(calibration,threshold)
            if (n-good)*100<=n:
                choices.append((n,threshold))
        _,threshold=max(choices)
        for key in ("implicit","correction"):
            n,good=totals(resolved(held[key],model["rows"][key]),threshold)
            summary.append(dict(seed=seed,mode=model["mode"],cohort=key,threshold=threshold,accepted=n,correct=good))
assert len(summary)==12
out=ROOT/"clarify-gate.json"
out.write_text(json.dumps(summary,indent=2))
aggregate=defaultdict(lambda:[0,0])
for r in summary:
    aggregate[r["mode"],r["cohort"]][0]+=r["accepted"]
    aggregate[r["mode"],r["cohort"]][1]+=r["correct"]
print(dict(aggregate))
