"""Independent follow-up rescoring for language, candidate filtering and sealed cells."""
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
from implicit import data_for
from bias_piecewise import conditions

actions=[(op,src,dst) for op in range(4) for src in range(3) for dst in range(3) if src!=dst]
results={"implicit":[],"bias_piecewise":[]}
seen=set()
for path in (LAB/"runs").glob("*/output/results.json"):
    config=json.loads((path.parents[1]/"experiment.json").read_text())
    suite,arm=config["command"][-2:]
    if suite not in results:
        continue
    assert verify_run(path.parents[1])["valid"]
    seed=config["seed"]
    seen.add((seed,suite,arm))
    data=json.loads(path.read_text())
    if suite=="implicit":
        _,_,_,tests=data_for(seed,arm)
        for item in data:
            expected=tests[item["cohort"]]
            exact=effect=op_correct=role_correct=admitted=precision=0
            for pred,confidence,sample in zip(item["predictions"],item["confidence"],expected,strict=True):
                operation,src,dst=actions[pred]
                state=sample[2].tolist()
                a,b=state[src],state[dst]
                state[dst]=[a,(a+b)%8,(b-a)%8,max(a,b)][operation]
                correct=actions[pred]==tuple(sample[4:])
                exact+=correct
                effect+=state==sample[3].tolist()
                op_correct+=operation==sample[4]
                role_correct+=(src,dst)==sample[5:]
                admitted+=confidence>=.8
                precision+=correct and confidence>=.8
            assert admitted==item["accepted"]
            if item["cohort"]!="ambiguous":
                assert (exact,effect,op_correct,role_correct,precision)==tuple(
                    item[k] for k in ("exact","effect","operation","role","correct_accepted"))
            results[suite].append(dict(seed=seed,arm=arm,**{k:v for k,v in item.items() if k not in ("predictions","confidence")}))
    else:
        programs,switches,fixtures=conditions(seed)
        matrix=programs.tolist()
        allowed=switches.tolist() if arm=="candidate" else [[0]*25]
        cases={}
        for kind,trial,truth,measured,train,pool,validation,test in fixtures:
            measured,truth=measured.tolist(),truth.tolist()
            survivors=[]
            for program in matrix:
                west_east=[(measured[k]-program[k])%5 for k in train]
                if any(all(len({m for k,m in zip(train,west_east) if switch[k]==side})<=1 for side in (0,1))
                       for switch in allowed):
                    survivors.append(program)
            cases[kind,trial]=(truth,pool,validation,test,survivors)
        for item in data["rows"]:
            truth,pool,validation,test,survivors=cases[item["scenario"],item["trial"]]
            queried=set()
            for key,label in item["queries"]:
                assert key in pool and key not in queried and label==truth[key]
                queried.add(key)
            filtered=[p for p in survivors if all(p[k]==v for k,v in item["queries"])]
            remain=[k for k in pool if k not in queried]+test
            unique=len(set(tuple(p[k] for k in remain) for p in filtered))==1
            values=[filtered[0][k] for k in test] if unique else None
            valid=unique and all(p[k]==truth[k] for p in filtered for k in validation)
            correct=unique and values==[truth[k] for k in test]
            assert item["count"]==len(filtered) and item["prediction"]==values
            assert item["validation"]==valid and item["correct"]==correct
            results[suite].append(dict(seed=seed,arm=arm,**item))
assert len(seen)==12
assert len(results["implicit"])==30 and len(results["bias_piecewise"])==4320
out=ROOT/"audit-followups.json"
out.write_text(json.dumps(results,indent=2))
put_node(Controller(LAB),"binding.repair.followups-audited","evidence",
         dict(scope="ENGINEERING_ONLY",sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
              implicit_predictions=15360,bias_evaluations=4320))
for arm in ("control","candidate"):
    for c in ("joint","implicit","shared_wording","distant_wording","ambiguous"):
        group=[r for r in results["implicit"] if r["arm"]==arm and r["cohort"]==c]
        print(arm,c,[(k,sum(r[k] or 0 for r in group)) for k in ("exact","effect","accepted","correct_accepted")])
    for c in ("constant","piecewise","checkerboard"):
        for budget in (1,2,4):
            group=[r for r in results["bias_piecewise"] if r["arm"]==arm and r["scenario"]==c and r["budget"]==budget]
            print(arm,c,budget,
                  sum(r["correct"] for r in group),sum(r["prediction"] is not None and not r["correct"] for r in group),
                  sum(r["validation"] and r["correct"] for r in group),sum(r["validation"] and not r["correct"] for r in group))
