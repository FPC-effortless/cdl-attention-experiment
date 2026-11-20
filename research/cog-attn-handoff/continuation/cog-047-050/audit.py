"""Check source identities, policy outcomes, paired gates and exact state transitions."""
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from research_lab.core import Controller
from research_lab.evidence import put_node
from trainlab.runner import verify_run

ROOT=Path(__file__).resolve().parent
LAB=ROOT/"ecosystem/research_lab"
sys.path.insert(0,str(ROOT/"recipe"))
from authentic import examples
from fusion import data_for, ordered_tokens, original_nli, vocab_lexicon
from binding import words_for_seed

LENGTHS=(128,512,2048,8192,32768)


def apply(action,state):
    o,s,d=action
    s,d=int(s),int(d)
    a,b=state[s],state[d]
    values=state.copy()
    if o==0:
        values[d]=a
    elif o==1:
        values[d]=(b+a)%8
    elif o==2:
        values[d]=(b-a)%8
    else:
        values[d]=max(a,b)
    return values


def parse_fusion():
    report=json.loads((ROOT/"fusion.json").read_text())
    result=defaultdict(lambda:[0,0,0])
    for row in report["rows"]:
        extra=words_for_seed(row["seed"])
        permitted=vocab_lexicon(original_nli(row["seed"]),extra)
        derived=ordered_tokens(row["tokens"],row["names"],permitted)
        assert (list(derived) if derived is not None else None)==row["local_contract"]
        if row["split"]!=2:
            continue
        chosen=row["local_contract"]
        if chosen is None and row["risk_score"]>=report["threshold"]:
            chosen=row["blended"]
        a=result[row["group"]]
        a[0]+=1
        a[1]+=chosen is not None
        a[2]+=chosen is not None and chosen==row["label"]
    assert dict(result)==report["results"]["calibrated"]
    return {k:v for k,v in report.items() if k!="rows"}


def replay_natural():
    artifact=json.loads((ROOT/"online.json").read_text())
    stats=[]
    for cell in artifact["streambench"]:
        rows=cell["steps"]
        assert len(rows)==48 and all(s["i"]==i and s["epoch"]==i//12 for i,s in enumerate(rows))
        expected=data_for(cell["seed"],2)
        pattern=[(example,group) for group in ("typed","implicit","correction","ambiguous")
                 for example in expected[group][:2]]
        history=(pattern+pattern[:4])*4
        actual={name:1 for row,_ in history for name in row[1]}
        target=actual.copy()
        for step,(frame,group) in zip(rows,history,strict=True):
            names=frame[1]
            gold=None if group=="ambiguous" else list(frame[4:])
            assert step["truth"]==gold and step["cohort"]==group
            if step["committed"] is not None:
                after=apply(step["committed"],[actual[n] for n in names])
                actual.update(zip(names,after))
            if step["epoch"]!=2 and gold is not None:
                after=apply(gold,[target[n] for n in names])
                target.update(zip(names,after))
            assert step["exact_state"]==(actual==target)
        exact=sum(s["exact_state"] for s in rows)
        committed=sum(s["committed"] is not None for s in rows)
        wrong=sum(s["committed"] is not None and s["committed"]!=s["truth"] for s in rows)
        assert wrong==0
        stats.append(dict(**{k:v for k,v in cell.items() if k!="steps"},committed=committed,exact=exact,
                          autopred=sum(s["auto"] is not None for s in rows),
                          cache_hits=sum(s["cache_hit"] for s in rows),wrong=wrong))
    assert len(stats)==6
    return stats


def validate_workload(cell):
    state=[1,4,6]
    intended=state.copy()
    counters={k:0 for k in ("raw_exact","committed","exact","match","stale","deferred")}
    for i,s in enumerate(cell["steps"]):
        assert (s["i"],s["epoch"])==(i,i//12)
        assert s["race"]==(i%12==0 and i!=0)
        action=s["action"]
        if action is not None:
            assert action[1]!=action[2]
            if s["profile"]=="missing":
                raise AssertionError("commit without independent root")
            state=apply(action,state)
        if s["profile"]!="missing":
            intended=apply(s["gold"],intended)
        assert state==s["state"] and intended==s["expected"]
        counters["raw_exact"]+=s["raw"]==s["gold"]
        counters["committed"]+=action is not None
        counters["exact"]+=action==s["gold"]
        counters["match"]+=state==intended
        counters["stale"]+=s["race"]
        counters["deferred"]+=action is None
    assert cell["total_ns"]==cell["fit_ns"]+cell["cold_ns"]+12*cell["elapsed_ns"]
    assert sum(cell["stages_ns"].values())<=cell["elapsed_ns"]
    
    return dict(**{k:v for k,v in cell.items() if k!="steps"},**counters)


def public_intents(artifact):
    results={}
    for split,out in artifact["result"].items():
        expected=list(examples(split))
        assert out["oracle_replay"]==dict(cases=len(expected),exact=len(expected))
        rows=out["predictions"]
        observed={}
        for actual,meta in zip(rows,(e for e in expected if e[2] is not None),strict=True):
            session,turn,_,target,full=meta
            assert (session,turn,target)==(actual["session"],actual["turn"],actual["truth"])
            if actual["action"]!="hold":
                observed[session]=actual["action"]
            truth=full.get("hotel-area",[""])[0]
            assert (observed.get(session,""),truth)==(actual["predicted"],actual["state"])
        exact=sum(r["predicted"]==r["state"] for r in rows)
        assert exact==out["exact"]
        results[split]={k:v for k,v in out.items() if k!="predictions"}
        results[split]["total"]=len(rows)
    return dict(seed=artifact["seed"],mode=artifact["mode"],result=results,fit_ns=artifact["fit_ns"])


fusion=parse_fusion()
natural=replay_natural()
modules=json.loads((ROOT/"modules.json").read_text())
for x in modules["lrc"]:
    checkpoint=ROOT/f"recipe/original_lrc/repair_{x['seed']}.pt"
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==x["sha256"]
    assert x["exact"]==sum(a==b for a,b in zip(x["pred"],x["truth"],strict=True))
    assert x["held_paths"]==sum(a==b for a,b in zip(x["selected"],x["gold_support"],strict=True))
for x in modules["csm"]:
    assert x["correct"]==sum(a==b for a,b in zip(x["predicted"],x["truth"],strict=True))
lineage=json.loads((ROOT/"lineage.json").read_text())
assert len(lineage)==40
for row in lineage:
    expected=(0,1) if row["profile"]=="unrelated" else (
        (1,0) if row["profile"]=="concealed" or (row["profile"]=="shared" and row["mode"]=="names") else None)
    assert row["admitted"]==(list(expected) if expected is not None else None)

costs,dialogues=[],[]
jobs=set()
for path in sorted((LAB/"runs").glob("*/output/results.json")):
    run=path.parents[1]
    assert verify_run(run)["valid"]
    spec=json.loads((run/"experiment.json").read_text())
    suite,arm=spec["command"][-2:]
    assert (suite,arm,spec["seed"]) not in jobs
    jobs.add((suite,arm,spec["seed"]))
    data=json.loads(path.read_text())
    if suite=="authentic":
        dialogues.append(public_intents(data))
    else:
        costs.extend(dict(series=suite,**validate_workload(cell)) for cell in data)
assert len(jobs)==24 and len(costs)==270 and len(dialogues)==6
summary=[]
for series in ("continual","continual_v2","continual_v3"):
    for length in LENGTHS:
        for charge in (0.,2.):
            scope=[r for r in costs if (r["series"],r["length"],r["charge_ms"])==(series,length,charge)]
            base={r["seed"]:r for r in scope if r["mode"]=="local"}
            for mode in ("full","scoped"):
                compare=[r for r in scope if r["mode"]==mode]
                successful=[r for r in compare if r["raw_exact"]/48>=.95 and base[r["seed"]]["raw_exact"]/48>=.95
                            and abs(r["raw_exact"]-base[r["seed"]]["raw_exact"])/48<=.01 and r["kv_parity"]<=1e-5
                            and r["exact"]==base[r["seed"]]["exact"]==36 and r["match"]==base[r["seed"]]["match"]==48]
                assert len(compare)==3
                summary.append(dict(series=series,mode=mode,length=length,charge_ms=charge,
                                    qualified=len(successful)==3,raw=[r["raw_exact"] for r in compare],
                                    max_parity=max(r["kv_parity"] for r in compare),
                                    baseline_cost_ms=statistics.median(r["total_ns"]/1e6 for r in compare),
                                    local_cost_ms=statistics.median(r["total_ns"]/1e6 for r in base.values()),
                                    speedup=(statistics.median(r["total_ns"]/base[r["seed"]]["total_ns"]
                                               for r in compare) if len(successful)==3 else None),
                                    online_ratio=(statistics.median(r["elapsed_ns"]/base[r["seed"]]["elapsed_ns"]
                                                      for r in compare) if len(successful)==3 else None)))
repair=json.loads((ROOT/"dialogue-repair.json").read_text())
labels=json.loads((ROOT/"recipe/dialogue-expanded.json").read_text())["test"]
dialogue_repair={}
for mode,cell in repair.items():
    dialogue_repair[mode]={"threshold":cell["threshold"],"train_n":cell["train_n"]}
    for group in ("raw","guarded"):
        results=cell[group]
        state={}
        correct=attempts=updates=0
        for label,row in zip(labels,results["rows"],strict=True):
            assert (label["dialogue"],label["turn"])==(row["dialogue"],row["turn"])
            if row["action"]!="hold":
                state[row["dialogue"]]=row["action"]
                attempts+=1
                updates+=row["action"]==label["action"]
            assert row["state"]==state.get(row["dialogue"],"")
            correct+=row["state"]==label["target"]
        assert (correct,attempts,updates)==(results["state_matches"],results["update_attempts"],results["correct_updates"])
        dialogue_repair[mode][group]={k:v for k,v in results.items() if k!="rows"}
kv=json.loads((ROOT/"kv-audit.json").read_text())
assert len(kv)==30 and all(r["max_abs"]<=1e-5 for r in kv)
artifact=dict(fusion=fusion,dialogue_repair=dialogue_repair,kv_replay=kv,natural=natural,modules=modules,lineage=lineage,dialogues=dialogues,cost_cells=costs,cost_summary=summary)
out=ROOT/"audit.json"
out.write_text(json.dumps(artifact,indent=2))
put_node(Controller(LAB),"cog.qualification.audit-v3","evidence",
         dict(scope="ENGINEERING_ONLY",jobs=len(jobs),cost_cases=len(costs)*48,
              sha256=hashlib.sha256(out.read_bytes()).hexdigest()))
for row in summary:
    print(row)
