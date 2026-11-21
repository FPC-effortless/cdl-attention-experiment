"""R1 persistent Hanoi pattern database, no fit; run then independently score."""
from collections import deque
from heapq import heappop, heappush
from hashlib import sha256
from pathlib import Path
from time import perf_counter_ns
import itertools
import json
import os
import random
import statistics
import sys
import tracemalloc
from tasks import Task, replay
from panel import distance, edges

SEEDS = (9419, 9431, 9433, 9437, 9461, 9463)
ARMS = ('cheap', 'tower', 'cold3', 'warm3', 'warm5')
HORIZONS = (1, 4, 8, 12)
ROOT = Path(__file__).resolve().parent

def cases():
    for seed, topology, regime in itertools.product(
        SEEDS, ('line3', 'ring4', 'complete4'), ('shared', 'mixed')
    ):
        key = f'{seed}/{topology}/{regime}'
        rng = random.Random('r1/'+key)
        pegs = 3 if topology == 'line3' else 4
        permutations = list(itertools.permutations(range(pegs)))
        def links(perm, second):
            pairs = [(a, b) for a in range(pegs) for b in range(pegs) if a != b and
                     (topology == 'complete4' or abs(a-b)==1 or
                      (pegs==4 and {a,b}=={0,3}))]
            if second and topology == 'complete4':
                pairs = [(a,b) for a,b in pairs if {a,b}!={0,1}]
            return tuple(sorted((perm[a], perm[b]) for a,b in pairs))
        perm = rng.choice(permutations)
        old = links(perm, False)
        new = rng.choice(sorted({links(x, True) for x in permutations} - {old}))
        prefix = tuple(rng.randrange(pegs) for _ in range(3))
        goals = set()
        tasks = []
        for i in range(12):
            target = ()
            while not target or target in goals:
                target = (prefix if regime=='shared' else tuple(rng.randrange(pegs) for _ in range(3))) + tuple(
                    rng.randrange(pegs) for _ in range(3))
            goals.add(target)
            start = target
            while start == target:
                start = tuple(rng.randrange(pegs) for _ in range(6))
            tasks.append(Task(f'{key}/{i}', 'hanoi', start, target, pegs=pegs,
                              links=old if i<6 else new))
        yield key, seed, regime, tasks

def solve(task, table, tower=False):
    width = len(next(iter(table))) if table else 0
    h = lambda s: max(task.lower(s) if tower else sum(a!=b for a,b in zip(s,task.target)),
                      table[s[:width]] if table else 0)
    q=[(h(task.start),0,0,task.start)]
    gscore={task.start:0}
    parent={}
    i=work=0
    while q:
        _, g, _, s = heappop(q)
        if g != gscore[s]:
            continue
        if s == task.target:
            break
        for move, nxt in task.neighbors(s):
            work += 1
            if g+1 < gscore.get(nxt, 10**12):
                i+=1
                gscore[nxt]=g+1
                parent[nxt]=s,move
                heappush(q,(g+1+h(nxt),g+1,i,nxt))
    s=task.target
    path=[]
    while s != task.start:
        s,move=parent[s]
        path.append(move)
    path.reverse()
    if not replay(task,path):
        raise ValueError('invalid emitted plan')
    return path, work

def build(k, task):
    from dataclasses import replace
    root = task.target[:k]
    abstract = replace(task, start=root, target=root)
    q=deque([root])
    d={root:0}
    work=0
    while q:
        s=q.popleft()
        for _, nxt in abstract.neighbors(s):
            work+=1
            if nxt not in d:
                d[nxt]=d[s]+1
                q.append(nxt)
    return d,work

def session(tasks,arm):
    cache={}
    rows=[]
    prev=None
    t0=perf_counter_ns()
    for task in tasks:
        # Observe public task topology; discard all old-topology tables.
        if task.links != prev:
            cache.clear()
            prev=task.links
        k=5 if arm=='warm5' else 3
        table=None
        cost=0
        hit=False
        if arm not in ('cheap','tower'):
            key=(task.pegs,task.links,k,task.target[:k])
            hit=key in cache
            if hit:
                table=cache[key]
            else:
                table,cost=build(k,task)
                if arm.startswith('warm'):
                    cache[key]=table
        path, work=solve(task,table,tower=arm=='tower')
        elapsed=perf_counter_ns()-t0
        rows.append(dict(actions=path,work=work,build=cost,hit=hit,ns=elapsed,
                         entries=sum(map(len,cache.values()))))
    return rows

def measure(out):
    rows=[]
    for key,seed,regime,tasks in cases():
        order=list(ARMS)
        rng=random.Random(key+'/ordering')
        record={arm:[] for arm in ARMS}
        for _ in range(7):
            rng.shuffle(order)
            for arm in order:
                record[arm].append(session(tasks,arm))
        peaks={}
        for arm in order:
            tracemalloc.start()
            session(tasks,arm)
            peaks[arm]=tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
        rows.append(dict(key=key,seed=seed,regime=regime,
                         fixtures=[t.digest() for t in tasks],arms=record,peaks=peaks))
    out.mkdir(parents=True,exist_ok=True)
    (out/'measurements.json').write_text(json.dumps(rows,separators=(',',':'))+'\n')
    (out/'metrics.json').write_text(json.dumps({'sessions':len(rows)})+'\n')

def summary(inp,out):
    records=json.loads(inp.read_text())
    seen=set()
    for (key,seed,regime,tasks),r in zip(cases(),records,strict=True):
        assert (key,seed,regime)==(r['key'],r['seed'],r['regime'])
        assert set(r['arms'])==set(ARMS) and set(r['peaks'])==set(ARMS)
        assert [t.digest() for t in tasks]==r['fixtures']
        for i,t in enumerate(tasks):
            assert t.digest() not in seen
            seen.add(t.digest())
            optimum=distance(t)
            for batches in r['arms'].values():
                assert len(batches)==7 and all(len(b)==12 for b in batches)
                for b in batches:
                    s=t.start
                    assert len(b[i]['actions'])==optimum
                    for move in b[i]['actions']:
                        s=dict(edges(t,s))[tuple(move)]
                    assert s==t.target
                    assert b[i]['work']==batches[0][i]['work']
                    assert b[i]['build']==batches[0][i]['build']
        for batches in r['arms'].values():
            for b in batches:
                assert not b[0]['hit'] and not b[6]['hit']
    result={'verified_tasks':len(seen),'per_stratum':{}}
    for regime in ('shared','mixed'):
        subset=[r for r in records if r['regime']==regime]
        scores={}
        for h in HORIZONS:
            scores[str(h)]={}
            for arm in ARMS:
                def ns(selection,a):
                    return sum(statistics.median(b[h-1]['ns'] for b in r['arms'][a])
                               for r in selection)
                blocks=[(ns([r for r in subset if r['seed']==seed],'cheap'),
                         ns([r for r in subset if r['seed']==seed],arm)) for seed in SEEDS]
                rng=random.Random(94571)
                ratios=[]
                for _ in range(5000):
                    sample=rng.choices(blocks,k=len(blocks))
                    ratios.append(sum(x for x,_ in sample)/sum(y for _,y in sample))
                ratios.sort()
                steps=[step for r in subset for step in r['arms'][arm][0][:h]]
                scores[str(h)][arm]=dict(ns=ns(subset,arm),search=sum(s['work'] for s in steps),
                    build=sum(s['build'] for s in steps),hits=sum(s['hit'] for s in steps),
                    peak_python_bytes_median=statistics.median(r['peaks'][arm] for r in subset),
                    max_entries=max(s['entries'] for s in steps),ci=[ratios[124],ratios[4874]])
        result['per_stratum'][regime]=scores
    out.mkdir(parents=True,exist_ok=True)
    (out/'amortization_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'metrics.json').write_text(json.dumps({'verified_tasks':len(seen)})+'\n')

if __name__=='__main__':
    stage=sys.argv[1]
    output=Path(os.environ['TRAINLAB_OUTPUT_DIR'])
    if stage=='measure': measure(output)
    elif stage=='score': summary(Path(sys.argv[2]),output)
    else: raise SystemExit('unknown stage')
