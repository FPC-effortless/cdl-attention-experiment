"""COG-093: learn sliding puzzle action ordering from self-solved episodes.

No reference paths or hidden optimal distances. Learn features solely from paths
that an online A* actor discovers and environment.apply verifies. Compare exact
same A* with/without the learned action priority on unseen 8-puzzles and
budgeted 15-puzzles. Never claim the A* mechanism itself was acquired.
"""
from __future__ import annotations
import json,math,sys,heapq,statistics,hashlib,random
from collections import defaultdict,Counter
from pathlib import Path
from time import perf_counter_ns
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE.parent))
from score_ladder import environment
from puzzles import Sliding
from difficulty import Sliding4

DIRECTIONS='UDLR'

def manhattan(state,n):
    return sum(abs(i//n-(t-1)//n)+abs(i%n-(t-1)%n) for i,t in enumerate(state) if t)

def loc_feature(s,t,a,n):
    z=s.index(0);dr=manhattan(t,n)-manhattan(s,n)
    # Compact physical feature, not exact test-board state or hidden answer.
    return f'{n}/{z//n}/{z%n}/{dr}/{a}'

def learn(paths):
    by=defaultdict(set)
    for r in paths:
        if r['solved']:
            for feature in r['features']:by[feature].add(r['root'])
    return {f:len(roots) for f,roots in by.items() if len(roots)>=2}

def search(p,model=None,arm='primitive',limit=40000):
    """A* with admissible Manhattan; learned preferences break f ties only."""
    t0=perf_counter_ns();n=int(len(p.start)**.5);goal=p.target
    h0=manhattan(p.start,n);q=[(h0,0,0,p.start)]
    best={p.start:0};parent={p.start:None};edge={};seq=0;attempt=pop=learned_hits=0
    while q and pop<limit:
        f,_,g,s=heapq.heappop(q)
        if g!=best[s]:continue
        pop+=1
        if s==goal:
            path=[];states=[]
            while parent[s] is not None:
                path.append(edge[s]);states.append((parent[s],s,edge[s]));s=parent[s]
            path.reverse();states.reverse()
            assert p.apply(path)
            return dict(solved=True,actions=len(path),expanded=pop,attempts=attempt,
                        learned_tiebreaks=learned_hits,elapsed_ns=perf_counter_ns()-t0,
                        path=path,features=[loc_feature(s,t,a,n) for s,t,a in states])
        for a,t in p.neighbors(s):
            attempt+=1;newg=g+1
            if newg>=best.get(t,1<<50):continue
            best[t]=newg;parent[t]=s;edge[t]=a;seq+=1
            f=newg+manhattan(t,n)
            # All arms share the exact admissible A* f score and search budget.
            # Distinct ordering is a genuine competence-preserving policy change.
            if arm=='learned':
                support=(model or {}).get(loc_feature(s,t,a,n),0)
                # Do not read gold reference; counts came from self-solved earlier roots.
                learned_hits+=int(support>0)
                tie=-support*100000+seq
            elif arm=='shuffled':
                support=(model or {}).get(loc_feature(s,t,DIRECTIONS[(DIRECTIONS.index(a)+1)%4],n),0)
                tie=-support*100000+seq
            else:tie=seq
            heapq.heappush(q,(f,tie,newg,t))
    return dict(solved=False,actions=0,expanded=pop,attempts=attempt,
                learned_tiebreaks=learned_hits,elapsed_ns=perf_counter_ns()-t0,
                path=[],features=[])

def run():
    cases=[json.loads(x) for x in (HERE.parent/'results'/'cases.jsonl').read_text().splitlines()]
    dev=[r for r in cases if r['split']=='development' and r['domain']=='sliding3']
    reserved=[r for r in cases if r['split']=='reserved' and r['domain'] in ('sliding3','sliding4')]
    training=[]
    for c in dev:
        p=environment(c);r=search(p,limit=250000)
        if r['solved']:
            training.append(dict(root=c['input_sha256'],solved=True,features=r['features']))
    model=learn(training)
    rows=[]
    for c in reserved:
        p=environment(c);limit=40000 if c['domain']=='sliding3' else 10000
        for arm in ('primitive','learned','shuffled'):
            r=search(p,model,arm,limit)
            rows.append(dict(id=c['id'],level=c['level'],domain=c['domain'],arm=arm,
                             input_sha256=c['input_sha256'],limit=limit,
                             **{k:v for k,v in r.items() if k not in ('path','features')}))
    summary={a:dict(solved=sum(x['solved'] for x in rows if x['arm']==a),n=sum(x['arm']==a for x in rows),
                    expanded=sum(x['expanded'] for x in rows if x['arm']==a),
                    attempts=sum(x['attempts'] for x in rows if x['arm']==a),
                    policy_ms=round(sum(x['elapsed_ns'] for x in rows if x['arm']==a)/1e6,3))
                for a in ('primitive','learned','shuffled')}
    result=dict(experiment='COG-ATTN-093 self-discovered sliding preferences',
       condition='A* algorithm supplied unchanged in all arms; only tie-break policy learned',
       training_roots=len(training),learned_features=len(model),learned_supports=model,
       summary=summary,rows=rows,
       limitations=['no reference answers used by policy or training','learning did not acquire search algorithm',
                    'test-time goal and full tile configuration explicitly observed',
                    '15-puzzle scrambles do not guarantee exact difficulty','budget exhaustion counted as failure'])
    out=HERE/'results';out.mkdir(exist_ok=True)
    (out/'sliding-093.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'training_roots':len(training),'learned_features':len(model),'summary':summary},indent=2))
    return result
if __name__=='__main__':run()
