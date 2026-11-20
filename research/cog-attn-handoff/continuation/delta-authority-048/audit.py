"""Audit held traces without trusting embedded correctness flags."""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

from research_lab.core import Controller
from research_lab.evidence import put_node
from trainlab.runner import verify_run

ROOT=Path(__file__).resolve().parent
LAB=ROOT/"ecosystem/research_lab"
sys.path.insert(0,str(ROOT/"recipe"))
from bridge import family_batch,csm_data,HELD,CHALLENGE
from original_lrc.bridge_data import bridge_dataset
from attestation import Authority,Verifier,fixture

valid_profiles={
    "labels":{"independent","unsigned_alias","modified_value","replay_nonce","wrong_origin_key","forged_grant",
              "enrolled_mirrors","copied_snapshot","equal_observations","hidden_common"},
    "signed":{"independent","enrolled_mirrors","copied_snapshot","equal_observations","hidden_common"},
    "scoped":{"independent","copied_snapshot","equal_observations","hidden_common"},
    "digest":{"independent","hidden_common"},
}

corrected_sha=hashlib.sha256((ROOT/"recipe/bridge_scope.py").read_bytes()).hexdigest()
manifests=[]
checks=defaultdict(lambda:[0,0,0])
routes=defaultdict(lambda:[0,0,0])
lrc=defaultdict(lambda:[0,0,0])
spot={}
receipts=set()
for path in sorted((LAB/"runs").glob("*/output/results.json")):
    run=path.parents[1]
    assert verify_run(run)["valid"]
    meta=json.loads((run/"experiment.json").read_text())
    study,arm=meta["command"][-2:]
    seed=meta["seed"]
    if study=="bridge_scope":
        receipt=json.loads((run/"run.json").read_text())
        if receipt["inputs"]["bridge_scope.py"]!=corrected_sha:
            continue
        manifests.append(dict(seed=seed,arm=arm,run_id=run.name,source_hashes=receipt["inputs"],
                             checkpoints={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in sorted(path.parent.glob("*")) if p.suffix in (".npz",".pt")}))
    assert (study,arm,seed) not in receipts
    receipts.add((study,arm,seed))
    out=json.loads(path.read_text())
    if study=="attestation":
        for c in out["results"]:
            a=checks[c["policy"],c["profile"]]
            a[0]+=1
            if c["admitted"] is not None:
                a[1]+=1
                expected=c["expected"] if c["profile"] in (
                    "independent","equal_observations","replay_nonce","wrong_origin_key","forged_grant") else c["expected"][::-1]
                assert c["admitted"]==expected
                a[2]+=c["admitted"]==c["expected"]
            assert (c["admitted"] is not None)==(c["profile"] in valid_profiles[c["policy"]])
        if arm=="candidate":
            for period,rows in out["spot"].items():
                p=int(period)
                for r in rows:
                    ticks=range(0,64,p) if p else ()
                    detection=next((i for i in ticks if i>=r["shift"]),None)
                    wrong=(64 if detection is None else detection)-r["shift"]
                    assert r["detected"]==detection and r["requests"]==len(ticks)
                    assert r["wrong_before_freeze"]==wrong
                spot[seed,period]=rows
    else:
        truths=[bridge_dataset(seed+551+i*313,64)["answer"].tolist() for i in range(4)]
        for cohort in out["profiles"]:
            profile=cohort["profile"]
            _,gold_family=family_batch(seed+890,256,"unseen" if profile=="unseen_family" else "held")
            _,_,gold_subject=csm_data(seed+327,256,CHALLENGE if profile=="challenge" else HELD)
            got=cohort["lrc"]
            a=lrc[study,arm,profile]
            a[0]+=got["trials"];a[1]+=got["held_answers"];a[2]+=got["held_paths"]
            for mode,items in cohort["results"].items():
                datum=routes[study,arm,profile,mode]
                for j,row in enumerate(items):
                    family=int(gold_family[j])
                    subject=int(gold_subject[j,0])
                    dest=truths[family][j]
                    value=1000*family+10*j+subject
                    assert row["target"]==[family,subject,dest,value]
                    written=row["admitted"]
                    if written:
                        assert row["value"]==1000*row["bank"]+10*j+row["subject"]
                    exact=written and [row["bank"],row["subject"],row["place"],row["value"]]==row["target"]
                    assert exact==row["exact"]
                    datum[0]+=1;datum[1]+=written;datum[2]+=exact
                    if mode=="unresolved":
                        assert not written and row["value"] is None
assert len(receipts)==18 and len(manifests)==6

# Pin representative evidence bytes and reject a cross-command or cross-epoch use.
a=Authority(1103)
request,replies,_=fixture(a,1103,1,"independent")
for changed in ({**request,"command":"other"},{**request,"era":4},{**request,"nonce":"next"}):
    assert Verifier(a.private.public_key()).authorize(replies,changed,"scoped") is None
vectors=[]
for profile in ("independent","enrolled_mirrors","copied_snapshot","unsigned_alias","hidden_common"):
    a=Authority(1103)
    request,replies,_=fixture(a,1103,1,profile)
    vectors.append(dict(profile=profile,request=request,ca=a.private.public_key().public_bytes_raw().hex(),replies=[
        dict(service=r["service"],grant=r["grant"].__dict__,body=r["body"],grant_sig=r["grant_sig"].hex(),
             response_sig=r["response_sig"].hex()) for r in replies]))
(ROOT/"vectors.json").write_text(json.dumps(vectors,indent=2))

delta=json.loads((ROOT/"delta.json").read_text())
corpus=json.loads((ROOT/"recipe/turns.json").read_text())
delta_report={}
for mode,method in delta["results"].items():
    data=method["test"]
    observed={}
    total=correct=match=wrong=0
    for src,row in zip(corpus["test"],data["details"],strict=True):
        assert (src["dialogue"],src["turn"],src["delta"],src["state"])==(
            row["dialogue"],row["turn"],row["truth"],row["annotated"])
        action=row["action"]
        if action!="hold":
            name=row["dialogue"]
            assert observed.get(name,"")!=action
            observed[name]=action
            total+=1
            correct+=action==src["delta"]
            wrong+=action!=src["delta"]
        state=observed.get(row["dialogue"],"")
        assert state==row["value"]
        match+=state==src["state"]
    assert (total,correct,wrong,match)==(data["admitted"],data["correct"],data["wrong"],data["state_match"])
    delta_report[mode]={k:{a:b for a,b in method[k].items() if a!="details"} for k in ("calibration","test","unfiltered")}
    delta_report[mode]["gate"]=method["gate"]
assertions=json.loads((ROOT/"assertions.json").read_text())
for group in ("held","raw"):
    observed={}
    admitted=correct=0
    for src,r in zip(corpus["test"],assertions[group]["rows"],strict=True):
        assert (src["dialogue"],src["turn"],src["delta"],src["state"])==(r["dialogue"],r["turn"],r["truth"],r["target"])
        action=r["admitted"]
        if action is not None:
            observed[r["dialogue"]]=action
            admitted+=1
            correct+=action==src["delta"]
        assert observed.get(r["dialogue"],"")==r["predicted"]
    assert (admitted,correct)==(assertions[group]["admitted"],assertions[group]["correct"])
for arm in ("candidate","control"):
    for profile in ("supported","challenge","unseen_family"):
        assert lrc["bridge_scope",arm,profile]==[3072,3072,3072]
        assert routes["bridge_scope",arm,profile,"unresolved"]==[768,0,0]
for profile in ("supported","challenge"):
    assert routes["bridge_scope","candidate",profile,"integrated"]==[768,768,768]
    for mode in ("rotate_tac","rotate_lrc","rotate_csm"):
        assert routes["bridge_scope","candidate",profile,mode][2]==0
assert routes["bridge_scope","candidate","unseen_family","integrated"]==[768,0,0]
(ROOT/"module-manifest.json").write_text(json.dumps(manifests,indent=2))
assertion_summary={k:{p:v for p,v in group.items() if p!="rows"} if isinstance(group,dict) else group
                   for k,group in assertions.items()}
summary=dict(delta=delta_report,assertions=assertion_summary,receipts=len(receipts),
             authority=[dict(policy=p,profile=s,counts=v) for (p,s),v in sorted(checks.items())],
             bridge=[dict(study=s,arm=p,cohort=c,mode=m,counts=v) for (s,p,c,m),v in sorted(routes.items())],
             lrc=[dict(study=s,arm=p,cohort=c,counts=v) for (s,p,c),v in sorted(lrc.items())],
             spot={str(p):[r for (seed,interval),items in spot.items() if interval==str(p) for r in items]
                   for p in (0,4,8,16)})
path=ROOT/"audit.json"
path.write_text(json.dumps(summary,indent=2))
put_node(Controller(LAB),"delta.authority.qualified-audit","evidence",
         dict(scope="ENGINEERING_ONLY",jobs=len(receipts),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
              dialogue_turns=len(corpus["test"])))
for row in summary["bridge"]:
    if row["mode"] in ("integrated","unresolved"):
        print(row)
print(summary["lrc"])
for name,result in summary["delta"].items():
    print(name,result)
