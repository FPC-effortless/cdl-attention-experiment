"""COG-093b: fused feature extraction; identical A* correctness and tie preferences.

Precomputation uses only the frozen training feature table from COG-093, not
held-out outcomes. Measures complete online CPU search for matched policies.
"""
from __future__ import annotations
import heapq,json,random,statistics,sys
from pathlib import Path
from time import perf_counter_ns
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from sliding_093 import environment,manhattan,search,learn,DIRECTIONS
from sliding_093 import loc_feature

def fast(p,weights=None,arm='primitive',limit=40000):
    start=perf_counter_ns();n=int(len(p.start)**.5);goal=p.target;weights=weights or {}
    # weights already checked on >=2 training roots, and converted once.
    q=[(manhattan(p.start,n),0,0,p.start)]
    best={p.start:0};parent={p.start:None};edge={};seq=0;attempt=pop=hits=0
    while q and pop<limit:
        f,_,g,s=heapq.heappop(q)
        if g!=best[s]:continue
        pop+=1
        if s==goal:
            path=[]
            while parent[s] is not None:
                path.append(edge[s]);s=parent[s]
            path.reverse();assert p.apply(path)
            return dict(solved=True,actions=len(path),expanded=pop,attempts=attempt,
                        learned_tiebreaks=hits,elapsed_ns=perf_counter_ns()-start)
        current_h=f-g;blank=s.index(0)
        for a,t in p.neighbors(s):
            attempt+=1;ng=g+1
            if ng>=best.get(t,1<<50):continue
            best[t]=ng;parent[t]=s;edge[t]=a;seq+=1
            next_h=manhattan(t,n)
            if arm=='learned':
                v=weights.get((n,blank,next_h-current_h,a),0)
                hits+=int(v>0)
                priority=seq-v*100000
            elif arm=='shuffled':
                a_shift=DIRECTIONS[(DIRECTIONS.index(a)+1)%4]
                priority=seq-weights.get((n,blank,next_h-current_h,a_shift),0)*100000
            else:priority=seq
            heapq.heappush(q,(ng+next_h,priority,ng,t))
    return dict(solved=False,actions=0,expanded=pop,attempts=attempt,
                learned_tiebreaks=hits,elapsed_ns=perf_counter_ns()-start)

def run():
    cases=[json.loads(x) for x in (HERE.parent/'results'/'cases.jsonl').read_text().splitlines()]
    dev=[c for c in cases if c['domain']=='sliding3' and c['split']=='development']
    model_train=[];train_t0=perf_counter_ns()
    for c in dev:
        r=search(environment(c),limit=250000)
        if r['solved']:model_train.append(dict(root=c['input_sha256'],solved=True,features=r['features']))
    model=learn(model_train)
    weights={tuple([int(parts[0]),int(parts[1])*int(parts[0])+int(parts[2]),int(parts[3]),parts[4]]):v
             for k,v in model.items() for parts in [k.split('/')]}
    training_ns=perf_counter_ns()-train_t0
    held=[c for c in cases if c['domain'] in ('sliding3','sliding4') and c['split']=='reserved']
    checks=[];timings=[]
    # Matched and randomized order, compute stable decisions independently of timing.
    for rep in range(3):
        for c in held:
            p=environment(c);limit=40000 if c['domain']=='sliding3' else 10000
            order=['primitive','learned','shuffled'];random.Random(rep*100+c['level']*11).shuffle(order)
            for arm in order:
                r=fast(p,weights,arm,limit)
                if rep==0:checks.append(dict(id=c['id'],domain=c['domain'],level=c['level'],arm=arm,
                                              **{k:v for k,v in r.items() if k!='elapsed_ns'}))
                timings.append(dict(rep=rep,id=c['id'],arm=arm,elapsed_ns=r['elapsed_ns']))
    summary={arm:dict(n=sum(r['arm']==arm for r in checks),solved=sum(r['solved'] for r in checks if r['arm']==arm),
                      expanded=sum(r['expanded'] for r in checks if r['arm']==arm),
                      attempts=sum(r['attempts'] for r in checks if r['arm']==arm),
                      total_ms_median=round(statistics.median(sum(r['elapsed_ns'] for r in timings if r['rep']==rep and r['arm']==arm)/1e6 for rep in range(3)),3))
             for arm in ('primitive','learned','shuffled')}
    previous=json.loads((HERE/'results/sliding-093.json').read_text())
    for arm in summary:
        prior=previous['summary'][arm]
        assert all(summary[arm][k]==prior[k] for k in ('n','solved','expanded','attempts')),(arm,summary[arm],prior)
    result=dict(experiment='COG-093b fused offline-learned tie priority in exact A*',
                model_precomputation_from_training_only=True,train_ms=round(training_ns/1e6,3),
                matched_evaluation_repetitions=3,summary=summary,rows=checks,timings=timings,
                controls=['primitive exact A*','same-frequency rotated-action learned table'],
                limitations=['scoped algorithmic optimization, not an independently learned search algorithm',
                             'memory, training and compilation charged separately',
                             'CPU variability and three repeats insufficient for robust hardware claims'])
    (HERE/'results/sliding-093b.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'train_ms':result['train_ms'],'summary':summary},indent=2))
    return result
if __name__=='__main__':run()
