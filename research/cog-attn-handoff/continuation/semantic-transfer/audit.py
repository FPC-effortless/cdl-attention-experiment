"""Recompute exact actions, effects and frozen admission thresholds."""
import hashlib
import json
import sys
from pathlib import Path

from research_lab.core import Controller
from research_lab.evidence import put_node
from trainlab.runner import verify_run

ROOT=Path(__file__).resolve().parent
LAB=ROOT/"ecosystem/research_lab"
sys.path.insert(0,str(ROOT/"recipe"))
from binding import words_for_seed
from paraphrases import fixture
from nli_transfer import fixture as nli_fixture

ACTIONS=[(o,s,d) for o in range(4) for s in range(3) for d in range(3) if s!=d]
GRID=(0.,.5,.8,.9,.95,.98,.99,.995,.999,.9999,1.01)


def admitted(row,a,c):
    return [p>=a and q>=c for p,q in zip(row["confidence"],row["clear"],strict=True)]


def replay(raw,result,has_label):
    assert len(raw)==len(result["pred"])
    exact,effect,role,operation=[],[],[],[]
    for datum,idx in zip(raw,result["pred"],strict=True):
        op,src,dst=ACTIONS[idx]
        state=datum[2].tolist()
        a,b=state[src],state[dst]
        state[dst]=[a,(a+b)%8,(b-a)%8,max(a,b)][op]
        exact.append(ACTIONS[idx]==tuple(datum[4:]))
        effect.append(state==datum[3].tolist())
        role.append((src,dst)==tuple(datum[5:]))
        operation.append(op==datum[4])
    if has_label:
        assert exact==result["correct"] and effect==result["effect"]
    else:
        assert result["correct"] is None and result["effect"] is None
    return exact,effect,role,operation


summary=[]
seen=set()
for path in sorted((LAB/"runs").glob("*/output/results.json")):
    run=path.parents[1]
    assert verify_run(run)["valid"]
    config=json.loads((run/"experiment.json").read_text())
    seed=config["seed"]
    nli=config["command"][-2]=="nli"
    if nli:
        _,_,cal,test=nli_fixture(seed)
    else:
        _,_,_,cal,test=fixture(seed,words_for_seed(seed))
    for model in json.loads(path.read_text()):
        m=model["mode"]
        assert (seed,m) not in seen
        seen.add((seed,m))
        for k,row in model["calibration"].items():
            replay(cal[k],row,k!="ambiguous")
        clear,bad=(model["calibration"][k] for k in ("clear","ambiguous"))
        feasible=[]
        for a in GRID:
            for c in GRID:
                good=admitted(clear,a,c)
                chosen=sum(good)
                wrong=sum(g and not r for g,r in zip(good,clear["correct"],strict=True))
                risks=sum(admitted(bad,a,c))
                if 100*wrong<=chosen and 100*risks<=len(bad["pred"]):
                    feasible.append((chosen,a,c))
        count,a,c=max(feasible)
        assert (count,a,c)==tuple(model["gate"][k] for k in ("cal_coverage","action","clarity"))
        for cohort,row in model["rows"].items():
            labeled=cohort!="ambiguous"
            exact,effect,role,operation=replay(test[cohort],row,labeled)
            data=dict(seed=seed,model=m,cohort=cohort,total=len(exact),
                      exact=sum(exact) if labeled else None,effect=sum(effect) if labeled else None,
                      role=sum(role) if labeled else None,operation=sum(operation) if labeled else None,
                      gate=model["gate"])
            for name,g in [("fixed",admitted(row,.8,0)),("calibrated",admitted(row,a,c))]:
                data[name]=dict(accepted=sum(g),correct=sum(f and p for f,p in zip(g,exact,strict=True)) if labeled else None)
            summary.append(data)
assert seen==(
    {(seed,model) for seed in (601,619,643) for model in ("pretrained","random","local")} |
    {(seed,model) for seed in (677,701,733) for model in ("pair","separate")})
assert len(summary)==54
out=ROOT/"audit.json"
out.write_text(json.dumps(summary,indent=2))
put_node(Controller(LAB),"semantic.transfer.audited","evidence",
         dict(scope="ENGINEERING_ONLY",sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
              holdout_cases=9*4*512+6*3*128,calibration_cases=9*768+6*256))
for model in ("pretrained","random","local","pair","separate"):
    for cohort in (("implicit","correction","ambiguous") if model in ("pair","separate") else
                   ("layout","implicit","correction","ambiguous")):
        rs=[r for r in summary if r["model"]==model and r["cohort"]==cohort]
        print(model,cohort,[sum(r[k] or 0 for r in rs) for k in ("exact","effect","role","operation")],
              {g:[sum(r[g][k] or 0 for r in rs) for k in ("accepted","correct")] for g in ("fixed","calibrated")})
