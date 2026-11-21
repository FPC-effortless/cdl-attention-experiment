"""Admissible partial-goal distance as a state-conditioned search skill."""
from collections import deque
from dataclasses import replace
from heapq import heappush,heappop
import json,random,sys
from time import perf_counter_ns
from tasks import Task,replay
from panel import ROOT,distance
from skills import predict

SEEDS={'fit':(9113,9127,9133),'dev':(9203,9209),'eval':(9311,9319,9323,9329,9337,9341)}
def cohort(part):
    for seed in SEEDS[part]:
        for i in range(10):
            rng=random.Random(f'subgoal/{seed}/{i}')
            n=4+i%3;p=4 if part=='eval' and i%2 else 3
            links=tuple((x,y) for x in range(p) for y in range(p) if x!=y and (i%3==0 or abs(x-y)==1))
            s=tuple(rng.randrange(p) for _ in range(n));g=s
            while s==g:g=tuple(rng.randrange(p) for _ in range(n))
            yield Task(f'{part}/{seed}/{i}','hanoi',s,g,pegs=p,links=links)

def data(t,s):
    k=min(3,len(s)-1)
    return [len(s),t.pegs,len(t.links),sum(a!=b for a,b in zip(s,t.target)),
            sum(a!=b for a,b in zip(s[:k],t.target[:k])),len(set(s[:k])),len(set(t.target[:k]))]

def abstract(t,k):
    simple=replace(t,start=t.target[:k],target=t.target[:k])
    d={simple.target:0};q=deque(d);work=0
    while q:
        s=q.popleft()
        for _,v in simple.neighbors(s):
            work+=1
            if v not in d:d[v]=d[s]+1;q.append(v)
    return d,work

def solve_bound(t,arm,tree=None,tau=0):
    start=perf_counter_ns();k=min(3,len(t.start)-1)
    table=None;build=lookups=queries=0
    def h(s):
        nonlocal table,build,lookups,queries
        if arm=='zero':return 0
        cheap=sum(a!=b for a,b in zip(s,t.target))
        if arm=='gate':
            queries+=1
            if predict(tree,data(t,s))<=tau:return cheap
        if arm=='cheap':return cheap
        if table is None:table,build=abstract(t,k)
        lookups+=1
        return max(cheap,table[s[:k]])
    gscore={t.start:0};q=[(h(t.start),0,0,t.start)];i=work=pops=0;parent={}
    while q:
        _,g,_,s=heappop(q)
        if g!=gscore[s]:continue
        pops+=1
        if s==t.target:break
        for a,n in t.neighbors(s):
            work+=1
            if g+1<gscore.get(n,10**12):
                i+=1;gscore[n]=g+1;parent[n]=s,a
                heappush(q,(g+1+h(n),g+1,i,n))
    path=[];s=t.target
    while s!=t.start:s,a=parent[s];path.append(a)
    path.reverse();assert replay(t,path)
    return dict(steps=len(path),actions=path,work=work,build=build,total=work+build,lookups=lookups,
                queries=queries,pops=pops,ns=perf_counter_ns()-start)
def fit():
    from sklearn.tree import DecisionTreeRegressor
    rows=[]
    for t in cohort('fit'):
        path=solve_bound(t,'cheap')['actions'];s=t.start
        for j in range(len(path)):
            if j in (0,len(path)//2,max(0,len(path)-3)):
                u=replace(t,start=s);a=solve_bound(u,'cheap');b=solve_bound(u,'always')
                assert a['steps']==b['steps']
                rows.append([t.id,data(t,s),a['total'],b['total']])
            s=dict(t.neighbors(s))[path[j]]
    reg=DecisionTreeRegressor(max_depth=3,min_samples_leaf=5,random_state=91)
    reg.fit([r[1] for r in rows],[r[2]-r[3] for r in rows]);tr=reg.tree_
    def node(i):
        return dict(leaf=float(tr.value[i].item())) if tr.children_left[i]<0 else dict(
            feature=int(tr.feature[i]),threshold=float(tr.threshold[i]),left=node(tr.children_left[i]),right=node(tr.children_right[i]))
    tree=node(0);profile={z:0 for z in (0,10,50,10**9)}
    for t in cohort('dev'):
        ts=list(profile);random.Random(t.id).shuffle(ts)
        for tau in ts:profile[tau]+=solve_bound(t,'gate',tree,tau)['ns']
    # Disable bypasses the selector entirely.
    cheap=sum(solve_bound(t,'cheap')['ns'] for t in cohort('dev'))
    tau=min(profile,key=profile.get) if min(profile.values())<cheap else None
    (ROOT/'results/subgoal-model.json').write_text(json.dumps(dict(tree=tree,training=rows,profile=profile,cheap=cheap,tau=tau)))
def score():
    m=json.loads((ROOT/'results/subgoal-model.json').read_text());out=[]
    for t in cohort('eval'):
        ref=distance(t);arms=['zero','cheap','always','gate'];rng=random.Random(t.id);by={}
        for _ in range(3):
            rng.shuffle(arms)
            for arm in arms:
                r=solve_bound(t,'cheap' if arm=='gate' and m['tau'] is None else arm,m['tree'],m['tau'])
                assert ref==r['steps']
                if arm not in by:by[arm]=r;by[arm]['times']=[]
                by[arm]['times'].append(r['ns'])
        for r in by.values():r['ns']=sorted(r['times'])[1]
        out.append(dict(id=t.id,digest=t.digest(),pegs=t.pegs,arms=by))
    return out
if __name__=='__main__':
    if sys.argv[1]=='fit':fit()
    else:(ROOT/'results/subgoal.json').write_text(json.dumps(score(),separators=(',',':'))+'\n')
