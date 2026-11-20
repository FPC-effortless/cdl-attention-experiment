"""Mechanism controls on the sequential switchboard stream."""
from collections import deque
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import SGDClassifier
from panel import N, Panel
HERE=Path(__file__).resolve().parent

def run(seed):
    rng=np.random.default_rng(seed)
    neurons=SGDClassifier(loss='log_loss',random_state=seed)
    memory={}
    recent=deque(maxlen=64)
    matrix=[]
    worlds=[]
    def features(e,s,a):
        # Same input for online, associative and episodic controls.
        return np.r_[e.frame(s),np.eye(N)[a],np.outer(np.eye(N)[a],e.frame(s)).ravel()]
    for epoch in range(12):
        e=Panel(rng);worlds.append(e)
        xs=[];ys=[]
        for _ in range(512):
            s=int(rng.integers(32));a=int(rng.integers(5));y=int(e.step(s,a)!=s)
            f=features(e,s,a);xs.append(f);ys.append(y)
            k=tuple(f[:20]),a
            memory[epoch,k]=y;memory[None,k]=y;recent.append(((epoch,k),y))
        neurons.partial_fit(xs,ys,classes=[0,1])
        row=[]
        for earlier,w in enumerate(worlds):
            inputs=[(s,a) for s in range(32) for a in range(5)]
            x=np.array([features(w,s,a) for s,a in inputs])
            y=np.array([w.step(s,a)!=s for s,a in inputs])
            keys=[(tuple(f[:20]),a) for f,(s,a) in zip(x,inputs)]
            row.append([sum(memory.get((earlier,k),-1)==v for k,v in zip(keys,y)),
                        sum(memory.get((None,k),-1)==v for k,v in zip(keys,y)),
                        sum(dict(recent).get((earlier,k),-1)==v for k,v in zip(keys,y)),
                        int((neurons.predict(x)==y).sum())])
        matrix.append(row)
    return {'metrics':['versioned_exact_association','unversioned_exact_association','bounded_episodic','online_delta_rule'],
            'opportunities_per_world':160,'matrix':matrix,'associative_keys':len(memory)}
if __name__=='__main__':
    (HERE/'results/continual-controls.json').write_text(json.dumps({s:run(s) for s in (5101,5107,5113)},default=lambda v:v.item()))
