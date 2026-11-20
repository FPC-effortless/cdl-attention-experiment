"""Audit all requests, support for committed bindings, and change to world state."""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

from research_lab.core import Controller
from research_lab.evidence import put_node
from trainlab.runner import verify_run

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"recipe"))
from fresh_scores import dataset
from policy import PROFILES,reply_tape,CURRENT
from reserve import tape_for,PROFILES as RESERVE_PROFILES

LAB=ROOT/"ecosystem/research_lab"
SEEDS=(751,773,797)


def state_effect(state,action):
    if action is None:
        return state
    operator,source,dest=action
    value=state[source]
    current=state[dest]
    if operator==1:
        value=(value+current)%8
    elif operator==2:
        value=(current-value)%8
    elif operator==3:
        value=max(value,current)
    return [value if i==dest else previous for i,previous in enumerate(state)]


def check_quorum(item,events,profile,mode,prior):
    roots=mode in ("quorum_roots","reserve")
    action=item["action"]
    assert len(events)<=8
    if not action:
        return
    assert item["reason"]=="quorum"
    assert all(not (x["epoch"]==CURRENT and x["status"]=="unresolved") for x in events)
    assert sum(p for i,p in enumerate(prior) if i//6==action[0])>=.99
    for slot in (1,2):
        valid=[e for e in events if e["slot"]==slot and e["epoch"]==CURRENT and e["status"]=="answer"]
        key="root" if roots else "service"
        assert len({e[key] for e in valid})==len(valid)
        assert len({e[key] for e in valid if e["value"]==action[slot]})>=2


sources={}
for seed in SEEDS:
    ref=json.loads((ROOT/f"recipe/scores-{seed}.json").read_text())
    folder=ROOT/"recipe"
    assert hashlib.sha256((folder/f"scores-{seed}.json").read_bytes()).hexdigest()==(folder/f"scores-{seed}.sha256").read_text()
    _,_,held=dataset(seed)
    expected=[(purpose,index,item) for purpose,examples in held.items() for index,item in enumerate(examples)]
    for bank,(purpose,index,item) in zip(ref["items"],expected,strict=True):
        assert bank["cohort"]==purpose and bank["index"]==index
        assert bank["state"]==item[2].tolist()
        assert bank["truth"]==(None if purpose=="ambiguous" else list(item[4:]))
        assert bank["words"]==" ".join(item[0]) and bank["entities"]==item[1]
        assert abs(sum(bank["posterior"])-1)<1e-5
    sources[seed]=ref["items"]

summary=defaultdict(lambda:dict(clear=0,ambiguous=0,admitted=0,wrong=0,effect_error=0,ambiguous_admitted=0,
                                requests_clear=0,requests_ambiguous=0))
jobs=set()
verified=0
for path in sorted((LAB/"runs").glob("*/output/results.json")):
    run=path.parents[1]
    assert verify_run(run)["valid"]
    config=json.loads((run/"experiment.json").read_text())
    seed=config["seed"]
    suite,arm=config["command"][-2:]
    assert (seed,suite,arm) not in jobs
    jobs.add((seed,suite,arm))
    payload=json.loads(path.read_text())
    assert payload["seed"]==seed
    n=0
    seen=set()
    for item in payload["rows"]:
        mode,profile,index=item["mode"],item["profile"],item["i"]
        assert (mode,profile,index) not in seen
        seen.add((mode,profile,index))
        n+=1
        bank=sources[seed][index]
        assert item["cohort"]==bank["cohort"]
        truth=bank["truth"]
        out=item["action"]
        if out is not None:
            assert out[0] in range(4) and all(v in range(3) for v in out[1:]) and out[1]!=out[2]
        expected=(tape_for(seed,index,profile,truth) if suite=="reserve" else reply_tape(seed,index,profile,truth))
        have={(r["service"],r["slot"]):r for r in expected}
        distinct=set()
        for event in item["events"]:
            binding=(event["service"],event["slot"])
            assert binding not in distinct
            distinct.add(binding)
            assert event==have[binding]
            if mode=="reserve":
                assert event["service"] in (4,5,6)
        if mode.startswith("quorum") or mode in ("reserve","public"):
            check_quorum(item,item["events"],profile,mode,bank["posterior"])
        assert item["requests"]==(3 if mode=="verified" else len(item["events"]))
        assert item["wrong"]==(out is not None and out!=truth)
        actual=state_effect(bank["state"],out)
        expected=state_effect(bank["state"],truth)
        effect_error=out is not None and actual!=expected
        assert item["effect_error"]==effect_error
        record=summary[suite,mode,profile]
        if truth is None:
            record["ambiguous"]+=1
            record["ambiguous_admitted"]+=out is not None
            record["requests_ambiguous"]+=item["requests"]
        else:
            record["clear"]+=1
            record["admitted"]+=out is not None
            record["wrong"]+=item["wrong"]
            record["effect_error"]+=item["effect_error"]
            record["requests_clear"]+=item["requests"]
        verified+=1
    n_modes=(2 if arm=="candidate" else 5) if suite=="staged" else 1
    assert n==384*n_modes*len(PROFILES if suite=="staged" else RESERVE_PROFILES)
assert len(jobs)==12 and verified==39168
records=[]
for (suite,mode,profile),counts in sorted(summary.items()):
    assert counts["clear"]==768 and counts["ambiguous"]==384
    precision=1-counts["wrong"]/counts["admitted"] if counts["admitted"] else None
    coverage=counts["admitted"]/counts["clear"]
    risk=counts["ambiguous_admitted"]/counts["ambiguous"]
    qualified=coverage>=.70 and precision is not None and precision>=.99 and risk<=.01
    records.append(dict(suite=suite,mode=mode,profile=profile,**counts,precision=precision,coverage=coverage,
                        ambiguous_risk=risk,qualified=qualified))
out=ROOT/"audit.json"
out.write_text(json.dumps(records,indent=2))
put_node(Controller(LAB),"staged.clarification.audit","evidence",
         dict(scope="ENGINEERING_ONLY",replayed=verified,
              audited_sha256=hashlib.sha256(out.read_bytes()).hexdigest()))
for row in records:
    print(row['suite'],row['mode'],row['profile'],row['admitted'],row['wrong'],row['effect_error'],row['ambiguous_admitted'],
          round(row['requests_clear']/row['clear'],3),row['qualified'])
