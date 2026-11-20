"""Recompute artifacts against generated inputs and a Python linear-scan history oracle."""
import hashlib
import json
import statistics
import sys
from pathlib import Path

from research_lab.core import Controller
from research_lab.evidence import put_node
from trainlab.runner import verify_run

root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/"recipe"))
from storage import history_rows
from language import COHORTS,samples

lab=root/"ecosystem/research_lab"
language,shifts,programs,races,histories=[],[],[],[],[]
for path in sorted((lab/"runs").glob("*/output/results.json")):
    directory=path.parents[1]
    assert verify_run(directory)["valid"]
    cfg=json.loads((directory/"experiment.json").read_text())
    suite,arm=cfg["command"][-2:]
    seed=cfg["seed"]
    payload=json.loads(path.read_text())
    if suite.startswith("language"):
        for stage in payload:
            for i,cohort in enumerate(COHORTS):
                row=stage["rows"][i]
                gold=samples(seed+7000+i,384,cohort)
                actions=[(o,s,d) for o in range(4) for s in range(3) for d in range(3) if s!=d]
                success=effects=0
                assert len(gold)==len(row["candidates"])
                for datum,choice,expected,flag in zip(gold,row["candidates"],row["gold"],row["effect_ok"]):
                    values,target=datum[2:4]
                    assert actions[expected]==tuple(datum[4:])
                    op,s,d=actions[choice]
                    a,b=int(values[s]),int(values[d])
                    made=values.copy()
                    made[d]=[a,(a+b)%8,(b-a)%8,max(a,b)][op]
                    effect=bool((made==target).all())
                    assert effect==flag
                    success+=choice==expected
                    effects+=effect
                if cohort!="unsupported":
                    assert (success,effects)==(row["action"],row["effects"])
                language.append(dict(seed=seed,suite=suite,arm=arm,stage=stage["stage"],
                                     memory_kib=stage["peak_rss_kib"],weights=stage["weight_bytes"],
                                     fit_cpu_ns=stage["fit_cpu_ns"],
                                     **{k:v for k,v in row.items() if k not in ("effect_ok","gold","candidates")}))
    elif suite=="epistemic":
        for row in payload["shifts"]:
            for observation in row["worlds"]:
                if row["mode"]=="event_only":
                    assert observation["lag"] is None and observation["wrong"]==1024-observation["change"]
                elif observation["lag"] is not None:
                    assert (observation["change"]+observation["lag"])%32==0
                shifts.append(dict(seed=seed,mode=row["mode"],channel=row["channel"],**observation))
        if payload["discovery"]:
            programs.extend(dict(seed=seed,**r) for r in payload["discovery"]["rows"])
    else:
        race=payload["race"]
        journal=race["journal"]
        assert {j[0] for j in journal}=={f"{worker}:{i}" for worker in range(4) for i in range(100)}
        assert race["value"]==race["pre_replay"]==sum(j[1] for j in journal)
        stale=sum(j[2:]!=[1,1] for j in journal)
        expected=0 if arm=="candidate" else 4 if suite=="storage_refresh" else 400
        assert stale==expected
        assert all(j[1]==j[2]+1 for j in journal)
        races.append(dict(seed=seed,suite=suite,arm=arm,stale=stale,writes=len(journal),
                          value=race["value"],retry=sum(w[1] for w in race["worker_stats"]),
                          worker_peak_max_kib=max(w[-1] for w in race["worker_stats"])))
        for h in payload["history"]:
            if suite=="storage_refresh":
                continue
            records,boundary,queries=history_rows(seed,h["size"])
            for query,answer in zip(queries,h["answers"],strict=True):
                k,vt,tx=query
                winning=max(((v,t,result) for key,v,t,result in records if key==k and v<=vt and t<=tx),default=None)
                assert answer==(None if winning is None else winning[2]),(seed,query,answer,winning)
            histories.append(dict(seed=seed,**{k:v for k,v in h.items() if k!="answers"},queries=len(queries)))
assert (len(language),len(shifts),len(programs),len(races),len(histories))==(120,1152,720,12,18)
summary=dict(language=language,shifts=shifts,programs=programs,races=races,history=histories)
(root/"audit.json").write_text(json.dumps(summary,indent=2))
put_node(Controller(lab),"repair.audited-results","evidence",dict(scope="ENGINEERING_ONLY",
         results_sha256=hashlib.sha256((root/"audit.json").read_bytes()).hexdigest(),
         audited=dict(language=len(language),shift_worlds=len(shifts),program_trials=len(programs),
                      race_jobs=len(races),history_queries=sum(h["queries"] for h in histories))))
for stage in ("base","adapted"):
    for cohort in COHORTS:
        for arm in ("candidate","control"):
            rr=[r for r in language if (r["arm"],r["stage"],r["cohort"],r["suite"])==(arm,stage,cohort,"language")]
            print(stage,cohort,arm,[sum(r[k] or 0 for r in rr) for k in ("action","effects")],
                  [sum(r["confidence0.8"][i] for r in rr) for i in (0,1)],
                  [sum(r["gate0.8"][i] for r in rr) for i in (0,1)])
for mode in ("event_only","periodic","intervention"):
    for channel in ("independent","shared"):
        rr=[r for r in shifts if (r["mode"],r["channel"])==(mode,channel)]
        found=[r["lag"] for r in rr if r["lag"] is not None]
        print(mode,channel,len(found),max(found,default=None),sum(r["wrong"] for r in rr),sum(r["anchors"] for r in rr))
for kind in ("compositional","random","biased"):
    for mode in ("shared_validation","independent_validation"):
        rr=[r for r in programs if (r["kind"],r["mode"])==(kind,mode)]
        print(kind,mode,sum(r["admit"] for r in rr),sum(r["correct"] for r in rr))
for size in (8192,65536):
    for mode in ("scan","index","compact"):
        rr=[r for r in histories if (r["size"],r["mode"])==(size,mode)]
        print(size,mode,{k:statistics.median(r[k] for r in rr) for k in ("rows","file_bytes","query_ns","build_ns","prep_ns","peak_parent_rss_kib")})
