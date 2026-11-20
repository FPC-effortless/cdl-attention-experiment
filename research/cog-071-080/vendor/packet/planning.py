"""Exact A* with optional learned or control multi-step edges."""
from dataclasses import dataclass
from heapq import heappush, heappop
from itertools import count
from collections import Counter, defaultdict
from .domain import initial, step, done, h

@dataclass
class Result:
    path: tuple | None
    cost: int | None
    popped: int
    attempted_primitives: int
    attempted_options: int
    exhausted_budget: bool
    option_ends: tuple = ()

def plan(w,s,blocked=None,options=(),max_pops=4096):
    if blocked is not None and s[blocked] < w.wanted[blocked]:
        return Result(None,None,0,0,0,False)
    stamp=count()
    q=[(h(w,s),0,next(stamp),0,s,(),())]
    best={s:0}
    pop=pr=op=0
    choices=tuple(options)+tuple((a,) for a in range(4+len(w.bays)))
    # Prefer deeper nodes on equal f; apply the same scheme to all arms.
    while q:
        _,_,_,g,s,path,ends=heappop(q)
        if best[s]!=g: continue
        if pop>=max_pops: return Result(None,None,pop,pr,op,True)
        pop+=1
        if done(w,s): return Result(path,g,pop,pr,op,False,ends)
        # Useless intermediate bay visits cannot improve this metric: direct
        # travel costs strictly less than any multi-move path of equal distance.
        useful={w.bays[i][s[i]] for i,n in enumerate(w.wanted)
                if i!=blocked and s[i]<n}
        # Options first only changes tie-order; all useful primitives remain.
        for seq in choices:
            if len(seq)==1 and seq[0]<4 and seq[0] not in useful:
                continue
            cur=s; gg=g; valid=True
            op+=len(seq)>1
            for a in seq:
                pr+=1
                z=step(w,cur,a,blocked)
                if z is None: valid=False;break
                cur,c=z;gg+=c
            if valid and gg<best.get(cur,10**12):
                best[cur]=gg
                newends=ends+(len(path)+len(seq),) if len(seq)>1 else ends
                heappush(q,(gg+h(w,cur),-gg,next(stamp),gg,cur,path+seq,newends))
    return Result(None,None,pop,pr,op,False)

def acquire(trace_rows,limit=16):
    """Compression proxy: frequency times saved choices minus dictionary cost.
    Only primitive successful traces and source-family identity are used.
    """
    counts=Counter()
    groups=defaultdict(set)
    for row in trace_rows:
        assert row['verified']
        path=tuple(row['path'])
        for n in (2,3,4):
            for i in range(len(path)-n+1):
                seq=path[i:i+n]
                counts[seq]+=1
                groups[seq].add(row['source_family'])
    eligible=[seq for seq in counts if len(groups[seq])>=3 and counts[seq]>=8]
    eligible.sort(key=lambda seq:(-((len(seq)-1)*counts[seq]-len(seq)),seq))
    return tuple(eligible[:limit])

