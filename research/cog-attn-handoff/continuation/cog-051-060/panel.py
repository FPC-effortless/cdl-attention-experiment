"""Opaque switchboard with raw redundant sensors, changing dependencies and motor interventions."""
from collections import Counter, deque
from time import perf_counter_ns
import numpy as np
from sklearn.tree import DecisionTreeClassifier

N = 5


class Panel:
    def __init__(self,rng):
        self.pixels = rng.permutation(np.repeat(np.arange(N),4))
        self.order = rng.permutation(N)
        self.parent = {a:int(self.order[i-1]) if i else None for i,a in enumerate(self.order)}

    def frame(self,state):
        return ((state >> self.pixels) & 1).astype('uint8')

    def step(self,state,a):
        parent = self.parent[a]
        return state ^ (1<<a) if parent is None or (state>>parent)&1 else state


class Dynamics:
    def __init__(self,observations):
        self.masks, self.rules = {}, {}
        self.observations = observations
        for a,rows in observations.items():
            changes = Counter(tuple(np.flatnonzero(x!=z)) for x,z in rows if np.any(x!=z))
            self.masks[a] = changes.most_common(1)[0][0] if changes else ()
        for a,rows in observations.items():
            self.rules[a] = DecisionTreeClassifier(random_state=0).fit(
                [self.encode(x) for x,z in rows],[int(np.any(x!=z)) for x,z in rows])
        self.graph = {}
        for n in range(1<<len(observations)):
            x=[(n>>j)&1 for j in observations]
            self.graph[n]=[n^(int(self.rules[a].predict([x])[0])<<a) for a in observations]

    def encode(self,frame):
        return [int(frame[mask[0]]) if mask else 0 for mask in self.masks.values()]

    def state(self,frame):
        return sum(v<<i for i,v in enumerate(self.encode(frame)))

    def label(self,x,z):
        change=tuple(np.flatnonzero(x!=z))
        return next((a for a,m in self.masks.items() if m and m==change),None)


def probe(env,rng,varied):
    rows={}
    for a in range(N):
        states=rng.integers(0,1<<N,size=64) if varied else [0]*64
        rows[a]=[(env.frame(s),env.frame(env.step(int(s),a))) for s in states]
    return Dynamics(rows)


def plan(model,start,goal):
    q=deque([(start,())])
    seen={start}
    while q:
        state,path=q.popleft()
        if state==goal:
            return path
        for a,next_state in enumerate(model.graph[state]):
            if next_state not in seen:
                seen.add(next_state)
                q.append((next_state,path+(a,)))
    return None


def rollout(env,s,path):
    for a in path:
        s=env.step(s,a)
    return s


def run(seed):
    rng=np.random.default_rng(seed)
    env=Panel(rng)
    model=probe(env,rng,True)
    single=probe(env,rng,False)
    state_trials=[]
    def exact(m,e):
        matches=0
        for s in range(1<<N):
            frame=e.frame(s)
            for a in range(N):
                predicted=frame.copy()
                latent=m.state(frame)
                if m.graph[latent][a]!=latent:
                    predicted[list(m.masks[a])]^=1
                matches+=int(np.array_equal(predicted,e.frame(e.step(s,a))))
        return matches
    for _ in range(8):
        new=Panel(rng)
        state_trials.append([exact(probe(new,rng,True),new),exact(probe(new,rng,False),new)])
    cache={}
    warm=[]
    for _ in range(240):
        start,goal=map(int,rng.choice(1<<N,2,replace=False))
        path=plan(model,start,goal)
        if path and rollout(env,start,path)==goal:
            cache[start,goal]=path
    trials=[]
    for _ in range(512):
        s,g=map(int,rng.choice(1<<N,2,replace=False))
        t=perf_counter_ns()
        p=plan(model,s,g)
        direct=perf_counter_ns()-t
        t=perf_counter_ns()
        hit=(s,g) in cache
        reused=cache[s,g] if hit else plan(model,s,g)
        cached=perf_counter_ns()-t
        state=s
        steps=0
        while state!=g and steps<64:
            state=env.step(state,int(rng.integers(N)))
            steps+=1
        trace=s
        correct=True
        for a in reused:
            nxt=env.step(trace,a)
            correct &= model.graph[trace][a]==nxt
            trace=nxt
        trials.append([s,g,len(p),steps,state==g,trace==g,correct,hit,direct,cached])
    worlds=[env]+[Panel(rng) for _ in range(11)]
    stores=[probe(w,rng,True) for w in worlds]
    retained=[exact(m,w) for m,w in zip(stores,worlds)]
    overwrite=[exact(stores[-1],w) for w in worlds]
    after=worlds[1]
    stale=sum(rollout(after,s,p)!=g for (s,g),p in cache.items())
    revised=stores[1]
    fixed=sum(rollout(after,s,plan(revised,s,g))==g for s,g in cache)
    # Retest every previously learned graph after each acquisition.
    matrix=[[exact(stores[i],worlds[i]) for i in range(t+1)] for t in range(len(stores))]
    return dict(discovery=state_trials,per_world=N*(1<<N),trials=trials,
                retention=retained,overwrite=overwrite,retention_matrix=matrix,
                compiled=len(cache),stale_without_signal=stale,replanned_after_signal=fixed,
                rules=[[int(k),v.tree_.node_count] for k,v in model.rules.items()])
