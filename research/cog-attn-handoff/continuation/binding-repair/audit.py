"""Verify TrainLab receipts and rescore against independent arithmetic/state replay."""
import ast
import hashlib
import json
import random
import sys
from pathlib import Path

from research_lab.core import Controller
from research_lab.evidence import put_node
from trainlab.runner import verify_run

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"recipe"))
from binding import COHORTS, rows, words_for_seed
from epistemic import grammar

LAB=ROOT/"ecosystem/research_lab"
actions=[(op,src,dst) for op in range(4) for src in range(3) for dst in range(3) if src!=dst]


def saved_language(seed,data):
    records=[]
    for model in data:
        for index,result in enumerate(model["rows"]):
            cohort=COHORTS[index]
            source=rows(seed+7100+index,words_for_seed(seed),512,cohort)
            exact=effect=roles=operators=accepted_correct=0
            keep=[c>=.8 for c in result["confidence"]]
            for sample,prediction,admit in zip(source,result["predictions"],keep,strict=True):
                state=list(map(int,sample[2]))
                op,src,dst=actions[prediction]
                a,b=state[src],state[dst]
                state[dst]=[a,(a+b)%8,(b-a)%8,max(a,b)][op]
                correct=actions[prediction]==tuple(sample[4:])
                exact+=correct
                effect+=state==sample[3].tolist()
                roles+=(src,dst)==sample[5:]
                operators+=op==sample[4]
                accepted_correct+=admit and correct
            assert sum(keep)==result["accepted"]
            if cohort!="ambiguous":
                assert (exact,effect,roles,operators,accepted_correct)==tuple(
                    result[k] for k in ("exact","effect","role","operation","correct_accepted"))
            records.append(dict(seed=seed,mode=model["mode"],**{k:v for k,v in result.items() if k not in ("confidence","predictions")}))
    return records


def evaluate_expr(expression,x,y):
    if isinstance(expression,ast.Name):
        return {"x":x,"y":y}[expression.id]
    if isinstance(expression,ast.Constant):
        return expression.value
    lhs=evaluate_expr(expression.left,x,y)
    rhs=evaluate_expr(expression.right,x,y)
    return (lhs+rhs if isinstance(expression.op,ast.Add) else lhs*rhs)%5


def saved_bias(seed,data,arm):
    programs,_=grammar()
    decoded=[]
    for signature,expr in programs:
        tree=ast.parse(expr,mode="eval").body
        eval_values=[evaluate_expr(tree,x,y) for x in range(5) for y in range(5)]
        assert signature==tuple(eval_values)
        decoded.append(eval_values)
    rng=random.Random(seed)
    states={}
    for kind in ("constant","input_dependent","random"):
        for trial in range(80):
            truth=rng.choice(decoded) if kind!="random" else [rng.randrange(5) for _ in range(25)]
            offset=rng.randrange(5)
            order=rng.sample(range(25),25)
            train,pool,test=order[:8],order[8:17],order[17:]
            rng.sample(pool,len(pool))
            measured=[(value+offset+int(kind=="input_dependent" and point%5>=3))%5
                      for point,value in enumerate(truth)]
            candidates=[p for p in decoded if len({(measured[k]-p[k])%5 for k in train})==1 and
                        (arm=="candidate" or all(measured[k]==p[k] for k in train))]
            states[kind,trial]=(truth,measured,pool,test,candidates)
    records=[]
    for row in data["rows"]:
        truth,measured,pool,test,candidates=states[row["scenario"],row["trial"]]
        assert row["initial"]==len(candidates)
        feedback=truth if row["channel"]=="independent" else measured
        seen=set()
        for point,value in row["queries"]:
            assert point in pool and point not in seen and feedback[point]==value
            seen.add(point)
        assert len(seen)<=row["budget"]
        survivors=[p for p in candidates if all(p[k]==v for k,v in row["queries"])]
        assert len(survivors)==row["count"]
        remainder=[k for k in pool if k not in seen]+test
        same=len({tuple(p[k] for k in remainder) for p in survivors})==1
        actual=[survivors[0][k] for k in test] if same else None
        assert actual==row["prediction"]
        assert row["correct"]==(actual is not None and actual==[truth[k] for k in test])
        records.append(dict(seed=seed,**row))
    return records


language,bias=[],[]
jobs=set()
for path in (LAB/"runs").glob("*/output/results.json"):
    run=path.parents[1]
    assert verify_run(run)["valid"]
    config=json.loads((run/"experiment.json").read_text())
    suite,arm=config["command"][-2:]
    seed=config["seed"]
    jobs.add((suite,arm,seed))
    payload=json.loads(path.read_text())
    if suite=="binding":
        language.extend(saved_language(seed,payload))
    else:
        bias.extend(saved_bias(seed,payload,arm))
assert len(jobs)==12
assert len(language)==72 and len(bias)==21600
summary={"language":language,"bias":bias}
out=ROOT/"audit.json"
out.write_text(json.dumps(summary,indent=2))
put_node(Controller(LAB),"binding.repair.audited","evidence",
         dict(scope="ENGINEERING_ONLY",language_predictions=len(language)*512,bias_evaluations=len(bias),
              sha256=hashlib.sha256(out.read_bytes()).hexdigest()))
for mode in ("local","sequence","pointer","transformer"):
    for cohort in COHORTS:
        group=[r for r in language if (r["mode"],r["cohort"])==(mode,cohort)]
        print(mode,cohort,[sum(r[k] or 0 for r in group)
                            for k in ("exact","effect","role","operation","accepted","correct_accepted")])
for filters in [("constant","independent"),("constant","compromised"),
                ("input_dependent","independent"),("random","independent")]:
    for mode in ("active","random","uncorrected"):
        print(filters,mode,[(n,sum(r["correct"] for r in bias if
                                       (r["scenario"],r["channel"],r["policy"],r["budget"])==(*filters,mode,n)),
                                sum(r["prediction"] is not None and not r["correct"] for r in bias if
                                       (r["scenario"],r["channel"],r["policy"],r["budget"])==(*filters,mode,n)))
                               for n in (0,1,2,4,8)])
