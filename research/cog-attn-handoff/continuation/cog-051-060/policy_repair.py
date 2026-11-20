"""Demand-compile a reverse shortest-path policy when an observed goal recurs."""
from collections import Counter, OrderedDict, defaultdict, deque
import json
from pathlib import Path
from time import perf_counter_ns
import numpy as np
from panel import Panel,probe,plan,rollout

def reverse_policy(model,goal):
    parents=defaultdict(list)
    for s,children in model.graph.items():
        for a,d in enumerate(children):parents[d].append((s,a))
    queue=deque([goal]);policy={goal:()}
    while queue:
        dest=queue.popleft()
        for s,a in parents[dest]:
            if s not in policy:
                policy[s]=(a,)+policy[dest];queue.append(s)
    return policy

def run(seed,hot):
    rng=np.random.default_rng(seed);freq=Counter();cache=OrderedDict();rows=[]
    for i in range(1536):
        if i%512==0:
            e=Panel(rng);model=probe(e,rng,True);freq.clear();cache.clear();epoch=i
        s=int(rng.integers(32));g=int(rng.integers(4 if hot and rng.random()<.85 else 32))
        t=perf_counter_ns();base=plan(model,s,g);uncompiled=perf_counter_ns()-t
        t=perf_counter_ns();freq[g]+=1;built=False;hit=g in cache
        if not hit and freq[g]>=3:
            if len(cache)==8:cache.popitem(last=False)
            cache[g]=(epoch,reverse_policy(model,g));built=True
        if g in cache:
            version,policy=cache[g];assert version==epoch
            cache.move_to_end(g);p=policy[s]
        else:p=plan(model,s,g)
        elapsed=perf_counter_ns()-t
        rows.append([g,s,built,hit,len(p)==len(base) and rollout(e,s,p)==g,elapsed,uncompiled])
    return rows
if __name__=='__main__':
    (Path(__file__).resolve().parent/'results/policy-repair.json').write_text(json.dumps({s:{str(h):run(s,h) for h in (False,True)} for s in (55019,55021,55043)}))
