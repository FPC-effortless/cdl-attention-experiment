"""Prospective balanced eight-key workload; original #119 unchanged."""
import numpy as np
import torch
from space import BASE
from knowledge import (START,MASK,QUERY,KEY,VAL,OP,WRITE,HOLD,RESET,TARGET,DIST,END,EMPTY,make8,maps)

# (action, mutation addresses query target?) exact quotas per group of eight.
BUCKETS=((WRITE,True),(RESET,True),(HOLD,True),(HOLD,False),
         (WRITE,False),(WRITE,False),(RESET,False),(RESET,False))

def generate(rng,allowed,n,held_syntax=False):
    assert n>0 and n%8==0
    allowed=np.asarray(allowed,dtype=np.int64)
    mask=np.zeros(4**8,dtype=bool);mask[allowed]=True
    q=rng.integers(8,size=n);other=(q+rng.integers(1,8,size=n))%8
    b=np.tile(np.arange(8),n//8)
    # Independently shuffle each eight-example group; preserve exact quotas.
    b=b.reshape(-1,8);b=np.take_along_axis(b,np.argsort(rng.random(b.shape),axis=1),axis=1).ravel()
    action=np.array([BUCKETS[j][0] for j in b]);target=np.array([BUCKETS[j][1] for j in b]);mut=np.where(target,q,other)
    ids=rng.choice(allowed,n);post=ids.copy();values=np.full(n,MASK,dtype=np.int64)
    for i in np.flatnonzero(action==WRITE):
        place=4**(7-int(mut[i]))
        for _ in range(10000):
            old=int(ids[i]//place%4);opts=ids[i]+(np.arange(4)-old)*place;ok=opts[(np.arange(4)!=old)&mask[opts]]
            if len(ok):
                post[i]=rng.choice(ok);values[i]=int(post[i]//place%4)+VAL;break
            ids[i]=rng.choice(allowed)
        else:raise RuntimeError('no feasible same-split WRITE after retries')
    post[action!=WRITE]=ids[action!=WRITE]
    change=np.stack((action,mut+KEY,values),axis=-1)
    order=np.argsort(rng.random((n,8)),axis=1);op=rng.integers(3,size=(n,2));syntax=rng.integers(2,4,size=n) if held_syntax else rng.integers(2,size=n)
    x,y,cut=make8(ids,q,op,order,other,syntax,change=change)
    assert cut==28 and x.shape==(n,36) and mask[ids].all() and mask[post].all()
    return x,y,dict(ids=ids,post=post,q=q,other=other,op=op,order=order,syntax=syntax,change=change,bucket=b)

def draw8_balanced(rng,allowed,n,fixed=False,mixed=False):
    if fixed or mixed:raise ValueError('Balanced generator supports prospective random-full regime only')
    x,y,_=generate(rng,allowed,n);return x,y

def independent_answer(x):
    """Reference interpreter parses input tokens; does not consume metadata or gold labels."""
    out=[]
    for row in x.tolist():
        assert row[0]==START and row[-1]==END and len(row)==36
        state={};events=row[1:-8];query=row[-8:]
        for i in range(0,len(events),3):
            a,k,v=events[i:i+3];assert KEY<=k<KEY+8
            if a==WRITE:assert VAL<=v<VAL+4;state[k]=v-VAL
            elif a==HOLD:assert v==MASK
            elif a==RESET:assert v==MASK;state[k]=EMPTY
            else:raise AssertionError('unrecognized event')
        k=query[query.index(TARGET)+1];value=state[k];assert query[query.index(DIST)+1]!=k
        if value!=EMPTY:
            for t in query[-3:-1]:
                assert t in (OP,OP+1,OP+2);value=(value+{OP:0,OP+1:1,OP+2:-1}[t])%4
        out.append(value+VAL)
    return torch.tensor(out,dtype=torch.long)

def counterfactuals(x,y,meta,allowed):
    """Create same-split paired examples on prefix-dependent rows only."""
    mask=np.zeros(4**8,dtype=bool);mask[allowed]=True
    change=meta['change'];q=meta['q'];eligible=~(((change[:,0]==WRITE)|(change[:,0]==RESET))&(change[:,1]==q+KEY))
    rows=[];newids=[]
    for i in np.flatnonzero(eligible):
        id0=meta['ids'][i];place=4**(7-int(q[i]));old=int(id0//place%4);opts=id0+(np.arange(4)-old)*place
        ok=(np.arange(4)!=old)&mask[opts]
        if change[i,0]==WRITE:
            # Same distractor update must remain in this split for both worlds.
            mp=4**(7-int(change[i,1]-KEY));before=int(id0//mp%4);v=int(change[i,2]-VAL);post=opts+(v-before)*mp;ok&=mask[post]
        if ok.any():rows.append(i);newids.append(opts[np.flatnonzero(ok)[0]])
    rows=np.asarray(rows,dtype=np.int64);newids=np.asarray(newids,dtype=np.int64)
    assert len(rows)>0 and mask[newids].all()
    x2,y2,_=make8(newids,q[rows],meta['op'][rows],meta['order'][rows],meta['other'][rows],meta['syntax'][rows],change=change[rows])
    assert torch.equal(x[rows,-11:],x2[:,-11:]) and torch.all(y[rows]!=y2) and torch.equal(y2,independent_answer(x2))
    return rows,x2,y2,int(eligible.sum())
