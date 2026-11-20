"""Frozen synthetic scenario generation and observed-only executive receipts."""
from collections import Counter
from dataclasses import replace
from hashlib import sha256
from itertools import product
from .domain import World, initial, step, done
from .planning import plan

ROUTES=((0,1,3),(0,2,1),(1,1,2),(1,3,0),(2,0,3),(2,3,2),(3,2,0),(3,3,1))
PANELS={"train":range(0,6),"dev_composition":range(6,8),"dev_binding":range(8,10)}

def split(sig):
    return sha256(bytes(sig)).digest()[0]%10

def cases():
    rows={}
    for panel,buckets in PANELS.items():
        rows[panel]=[]
        for sid,sig in enumerate(product(range(len(ROUTES)),repeat=3)):
            if split(sig) not in buckets: continue
            items=tuple(f'{panel}-{sid}-{i}' for i in range(3))
            wanted=tuple(2+((x+i)%2) for i,x in enumerate(sig))
            w=World(tuple(ROUTES[x] for x in sig),wanted,items)
            focus=sid%3
            rows[panel].append((sig,'success',focus,w))
            rows[panel].append((sig,'initial_hold',focus,replace(w,blocked=focus)))
            rows[panel].append((sig,'revision',focus,replace(w,revise_after=1+sid%3)))
    train={x[0] for x in rows['train']}
    for panel in ('dev_composition','dev_binding'):
        assert train.isdisjoint(x[0] for x in rows[panel])
        assert all(any(s[i]==x[0][i] for s in train) for x in rows[panel] for i in range(3))
    return rows

def execute(w,focus,options=(),max_pops=4096):
    state=initial(w); version=0; blocked=w.blocked;spent=0
    receipts=[]; events=[]; expansions=0; probes=0; attempts=0; issued=0
    budget=False;planned_options=0;finished_options=0
    while True:
        result=plan(w,state,blocked,options,max(0,max_pops-expansions))
        expansions+=result.popped
        probes+=result.attempted_primitives
        attempts+=result.attempted_options
        budget=budget or result.exhausted_budget
        if result.path is None: break
        epoch=version
        invalidated=False
        planned_options+=len(result.option_ends)
        for ordinal,a in enumerate(result.path,1):
            if w.revise_after is not None and issued==w.revise_after and not events:
                version+=1
                blocked=focus
                events.append((issued,version,focus))
                invalidated=True
                break
            assert epoch==version
            updated=step(w,state,a,blocked)
            assert updated is not None, (state,a,blocked)
            state,cost=updated
            spent+=cost;issued+=1
            receipts.append((epoch,a,state,cost))
            finished_options+=ordinal in result.option_ends
        if not invalidated: break
    status="completed" if done(w,state) else ("unknown" if budget else "blocked")
    return {"status":status,"version":version,"state":state,"receipts":receipts,
            "events":events,"spent":spent,"expanded":expansions,
            "transition_attempts":probes,"option_attempts":attempts,"focus":focus,
            "planned_options":planned_options,"finished_options":finished_options,
            "budget":budget}

def flat_fields(w,r):
    return (('request',r['focus']),('receipt',{'completed':0,'blocked':1,'unknown':2}[r['status']]),
            ('version',r['version']),('progress',r['state'][r['focus']]),
            ('total',w.wanted[r['focus']]))

