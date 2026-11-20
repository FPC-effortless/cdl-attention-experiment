"""Controlled source corruption overlay on real comments and their recorded outcomes."""
import hashlib
import json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent/'results'
rows=json.loads((R/'67003/issue-predictions.json').read_text())
rows=[r for r in rows if r[0]=='pooled_flat']
delays=[r[-1] for r in json.loads((R/'confirmation.json').read_text())['rows'] if r[-1]>0]
# Rewards and costs are explicit experimental assumptions (action utility units).
def score(r,threshold,policy):
    _,split,issue,k,y,p,_=r
    rng=np.random.default_rng(int(hashlib.sha256(str(k).encode()).hexdigest()[:12],16))
    shared=bool(rng.random()<.25)
    answers=[int(y^(rng.random()<.08)) if rng.random()>.15 else None for _ in range(3)]
    if shared: answers[1]=answers[0]
    delay=float(rng.choice(delays))
    if policy=='defer':return False,0,0
    if policy=='direct':return p>=threshold,0,0
    if policy=='utility':ask=p>=threshold
    elif policy=='entropy':ask=min(p,1-p)>=threshold
    else:ask=True
    if not ask:return False,0,0
    votes=[v for i,v in enumerate(answers) if not(policy!='pooled' and shared and i==1)]
    return votes.count(1)>=2,3,delay
def metrics(samples,t,policy):
    arr=[score(r,t,policy) for r in samples]
    tp=sum(bool(a) and r[4] for (a,_,_),r in zip(arr,samples));fp=sum(bool(a) and not r[4] for (a,_,_),r in zip(arr,samples))
    requests=sum(n for a,n,d in arr);seconds=sum(d for a,n,d in arr)
    return dict(n=len(samples),tp=tp,fp=fp,recall=tp/sum(r[4] for r in samples),requests=requests,
                charged_seconds=seconds,utility=tp-25*fp-.05*requests-.0001*seconds)
out={}
for policy in ('utility','entropy','pooled','verify_all','direct','defer'):
    valid=[r for r in rows if r[1]=='validation']
    options=[(metrics(valid,t,policy)['utility'],t) for t in ([.015,.03,.05,.10,.2,.4,.6,.8,.95,1.01] if policy in ('utility','entropy','direct') else [0])]
    t=max(options)[1]
    out[policy]={'threshold':t,**{s:metrics([r for r in rows if r[1]==s],t,policy) for s in ('test','external')}}
(R/'uncertainty.json').write_text(json.dumps(out,indent=2))
