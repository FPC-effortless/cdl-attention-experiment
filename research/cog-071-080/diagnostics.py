"""Scoped transfers of the original acquisition and authority policies."""
import json,random
from pathlib import Path
from time import perf_counter_ns
import numpy as np
from causal import env,INPUTS,Learner
from transfer import Careful
from vendor.p067 import policies
from vendor.gateway068 import Authority,admissible

ROOT=Path(__file__).resolve().parent
def sensors(seed):
    rng=random.Random(seed);training=[]
    for _ in range(400):
        rule=rng.randrange(4);bit=rng.randrange(2)
        # All observational channels see co-varying flags. Terminal utility depends on a held intervention.
        x=int(env(rule,(bit,bit))); y=int(env(rule,(0,1)))
        training.append((x,{'direct':x,'mirror':x,'replay':x},(int(not y),y,0)))
    model=policies.fit(training);model.update(ask_threshold=.8,ask_floor=1.)
    out=[]
    for i in range(64):
        new=rng.choice([1,2,3]);bit=rng.randrange(2);x=int(env(new,(bit,bit)))
        active=Careful('active');active.audit(lambda f:env(new,f),i%2==0)
        missing=active.decision((0,1)) is None
        for mode in ('passive','adaptive','diagnostic'):
            if mode=='diagnostic':
                action=active.decision((0,1));requests=active.probes
            else:
                source=policies.query(model,x,mode,0,dict(policies.QUOTA),8,{})
                action=policies.decide(model,x,source,x if source else None,0)[2]
                requests=int(source is not None)
            truth=env(new,(0,1))
            out.append([i,mode,i%2==0,requests,action,truth,missing])
    return out

def packed(seed):
    rng=np.random.default_rng(seed)
    # Requests inside a packet come from distinct sessions and are already coavailable.
    batches=[]
    for b in range(48):
        pack=[]
        for j in range(8):
            x=rng.normal(size=(8,3)); h={str(i):[(t,[float(v)]) for t,v in enumerate(x[i])] for i in range(8)}
            events=[Authority(str(i),'issues',2,1,(1,bool(i%4))) for i in range(8)]
            pack.append((events,h))
        batches.append(pack)
    w=np.array([.6,-.4,.8]);out={}
    for mode in random.Random(seed).sample(['folded','packed'],2):
        start=perf_counter_ns();answer=[]
        for pack in batches:
            x=[];groups=[]
            for e,h in pack:
                selected=admissible(e,'issues',2,h)
                lanes=sorted(selected)
                arr=np.array([selected[i] for i in lanes]).reshape(len(lanes),3)
                if not len(arr):raise AssertionError('no eligible lanes')
                if mode=='folded':answer.append(lanes[int((arr@w).argmax())])
                else:x.append(arr);groups.append(lanes)
            if mode=='packed':
                scores=np.stack(x)@w
                answer.extend(lanes[int(s.argmax())] for lanes,s in zip(groups,scores))
        out[mode]=dict(elapsed_ns=perf_counter_ns()-start,answer=answer)
    assert out['folded']['answer']==out['packed']['answer']
    return out
if __name__=='__main__':
    result={s:dict(causal=sensors(s),packed=packed(s)) for s in (8111,8123,8147)}
    (ROOT/'results/diagnostics.json').write_text(json.dumps(result))
