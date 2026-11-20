"""Stress the observation prior and mine reusable subpaths from successful behavior."""
from collections import Counter,deque
import json
from pathlib import Path
from time import perf_counter_ns
import numpy as np
from panel import Panel,probe,plan,rollout

def sensors(seed):
    rng=np.random.default_rng(seed+170000)
    metrics=[]
    for width in (3,5,8):
        for noise in (0,.05,.15):
            for correlated in (False,True):
                # Object count, masks, partitions and inert pixels are hidden from the learners.
                pix=rng.permutation(np.repeat(np.arange(width+1),4))
                scores=[];modes=[]
                for a in range(width):
                    changes=[]
                    for _ in range(128):
                        s=rng.integers(0,2,width+1);out=s.copy();out[a]^=1
                        if correlated and a in (0,1):out[1-a]^=1
                        x=s[pix]^(rng.random(len(pix))<noise)
                        y=out[pix]^(rng.random(len(pix))<noise)
                        changes.append(x!=y)
                    scores.append(np.mean(changes,axis=0))
                    modes.append(Counter(tuple(np.flatnonzero(z)) for z in changes).most_common(1)[0][0])
                scores=np.array(scores)
                peak=scores.max(axis=0)
                # Two actuators with indistinguishable effects leave the identity unresolved.
                winner=scores.argmax(axis=0)
                gap=peak-np.sort(scores,axis=0)[-2]
                inferred=np.where(peak<.5,width,np.where(gap>.18,winner,-1))
                exact=sum(set(mask)==set(np.flatnonzero(pix==a)) for a,mask in enumerate(modes))
                recovered=sum(np.array_equal(inferred==a,pix==a) for a in range(width))
                metrics.append(dict(width=width,noise=noise,correlated=correlated,mode=int(exact),contrast=int(recovered),
                                    unknown_pixels=int((inferred<0).sum())))
    return metrics

def skills(seed):
    rng=np.random.default_rng(seed+180000)
    env=Panel(rng); model=probe(env,rng,True)
    train=set()
    while len(train)<160:train.add(tuple(map(int,rng.choice(32,2,replace=False))))
    support=Counter()
    for s,g in sorted(train):
        path=plan(model,s,g)
        assert rollout(env,s,path)==g
        support.update({path[i:i+k] for k in (2,3,4) for i in range(len(path)-k+1)})
    macros=[seq for seq,n in support.most_common(12) if n>=3]
    options=[(a,) for a in range(5)]+macros
    def apply(s,p):
        for a in p:s=model.graph[s][a]
        return s
    compiled=[[apply(s,p) for p in options] for s in range(32)]
    def solve(s,g):
        queue=deque([(s,())]); seen={s}
        while queue:
            state,path=queue.popleft()
            if state==g:return path
            for idx,nxt in enumerate(compiled[state]):
                if nxt not in seen:
                    seen.add(nxt);queue.append((nxt,path+(idx,)))
    rows=[]
    for s in range(32):
        for g in range(32):
            if s==g or (s,g) in train:continue
            t=perf_counter_ns(); baseline=plan(model,s,g);base=perf_counter_ns()-t
            t=perf_counter_ns(); choice=solve(s,g); path=sum((options[i] for i in choice),());elapsed=perf_counter_ns()-t
            rows.append([len(baseline),len(choice),len(path),rollout(env,s,path)==g,any(i>=5 for i in choice),base,elapsed])
    return dict(macros=macros,distinct_train=len(train),held=rows)
if __name__=='__main__':
    dest=Path(__file__).resolve().parent/'results'
    (dest/'discovery-skills.json').write_text(json.dumps({s:{'sensors':sensors(s),'skills':skills(s)} for s in (5101,5107,5113)}))
