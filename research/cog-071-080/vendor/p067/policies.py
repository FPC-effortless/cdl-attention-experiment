"""Small empirical policy; fitting is a separate CLI stage."""
from collections import defaultdict,Counter
from .world import COST,ACQ_SLOTS

SOURCES=('direct','replay','mirror')
ARMS=('adaptive','fixed_evsi','learned_uncertainty','passive','always_ask','random_matched','query_ignore')
MATCHED=('adaptive','fixed_evsi','learned_uncertainty','random_matched','query_ignore')
QUOTA={'direct':2,'replay':1}

def key(passive,s=None,v=None):
    return str(passive) if s is None else f'{passive}:{s}:{v}'

def fit(examples):
    table=defaultdict(lambda:[0,0,0,0,0,0,0])
    hist=defaultdict(Counter)
    for passive,obs,reward in examples:
        maximum=max(reward)
        ks=[key(passive)]
        for s in SOURCES:
            v=obs[s];ks.append(key(passive,s,v));hist[key(passive)+':'+s][str(v)]+=1
        for k in ks:
            x=table[k];x[0]+=1
            for a,y in enumerate(reward):
                x[a+1]+=y-reward[0]
                x[a+4]+=int(maximum-y<=0)
    return {'table':dict(table),'hist':{k:dict(v) for k,v in hist.items()}}

def summary(model,p,s=None,v=None):
    g=model['table'].get(key(p),[0]*7)
    q=model['table'].get(key(p,s,v),g)
    if s is not None and sum(q)==0:q=g
    n=q[0]
    vals=[x/(n+2) for x in q[1:4]]
    confidence=[(x+1)/(n+2) for x in q[4:7]]
    return vals,confidence

def evsi(model,p,s):
    hist=model['hist'].get(key(p)+':'+s,{})
    norm=sum(hist.values())
    if not norm:return 0
    post=sum(n*max(summary(model,p,s,None if k=='None' else int(k))[0]) for k,n in hist.items())/norm
    return post-max(summary(model,p)[0])

def decide(model,p,s,v,tau):
    vals,conf=summary(model,p,s,v)
    choice=max((0,1,2),key=lambda a:(vals[a],-a))
    if conf[choice]<tau:
        return 'UNKNOWN',conf[choice],choice
    return choice,conf[choice],choice

def query(model,p,arm,slot,quota,upticks,schedule):
    avail=[k for k,v in quota.items() if v>0]
    if arm=='passive' or not avail:return None
    if arm=='always_ask':
        affordable=[s for s in avail if COST[s]<=upticks]
        return max(affordable,key=lambda s:(evsi(model,p,s)/COST[s],s)) if affordable else None
    if slot>=ACQ_SLOTS:return None
    if arm in ('random_matched','fixed_evsi','query_ignore'):return schedule.get(slot)
    force=sum(quota.values())>=ACQ_SLOTS-slot
    if arm=='learned_uncertainty':
        _,conf=summary(model,p)
        if force or max(conf)<model['ask_threshold']:
            return min(avail,key=lambda s:(COST[s],s))
        return None
    ranked=sorted(avail,key=lambda s:(-evsi(model,p,s)/COST[s],s))
    pick=ranked[0]
    return pick if force or evsi(model,p,pick)>model['ask_floor']*COST[pick] else None
