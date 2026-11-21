"""Read-only audit of archived R1 measurements after the producer's timed run."""
import collections
import json
import pathlib
import statistics
import sys

from amortization import cases

work=pathlib.Path(sys.argv[1])
rows=json.loads((work/"measurements.json").read_bytes())

def successors(state, links):
    allowed=set(links)
    for i,p in enumerate(state):
        if p in state[:i]:
            continue
        for a,b in allowed:
            if a==p and b not in state[:i]:
                yield (a,b),state[:i]+(b,)+state[i+1:]

def bfs(start, goal, links):
    todo=collections.deque([start])
    best={start:0}
    while todo:
        state=todo.popleft()
        if state==goal:
            return best[state]
        for _,next_state in successors(state,links):
            if next_state not in best:
                best[next_state]=best[state]+1
                todo.append(next_state)
    raise AssertionError("disconnected")

stats={}
for (key,seed,regime,tasks),r in zip(cases(),rows,strict=True):
    assert (key,seed,regime)==(r['key'],r['seed'],r['regime'])
    for i,task in enumerate(tasks):
        optimal=bfs(task.start,task.target,task.links)
        for arm,buckets in r['arms'].items():
            for b in buckets:
                rec=b[i]
                state=task.start
                assert len(rec['actions'])==optimal
                for act in rec['actions']:
                    state=dict(successors(state,task.links))[tuple(act)]
                assert state==task.target
    for arm,buckets in r['arms'].items():
        rr=stats.setdefault((regime,arm),collections.Counter())
        rr['ns']+=statistics.median(b[-1]['ns'] for b in buckets)
        for step in buckets[0]:
            rr['work']+=step['work']
            rr['build']+=step['build']
            rr['hit']+=step['hit']

for (regime,arm),st in sorted(stats.items()):
    ref=stats[(regime,'cheap')]
    print(regime,arm,st['work']+st['build'],st['ns']/1_000_000,st['hit'],ref['ns']/st['ns'])
print('Independent primitive BFS and replay passed',len(rows)*12*5*7)
