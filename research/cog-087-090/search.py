"""Exact A* with matched tie order and optionally certified branch contraction."""
from heapq import heappush,heappop
from time import perf_counter_ns
from tasks import replay
from skills import View,features,predict,suffix

def solve(t,arm='A',opts=(),model=None,tau=0,first=False,keep_states=False):
    begin=perf_counter_ns()
    view=View(t.cells) if t.kind=='maze' and arm in ('E','structural') else None
    best={t.start:0};parent={};stamp=0
    heap=[(t.h(t.start),0,0,t.start)]
    calls=attempts=generated=popped=queries=activated=certified=0
    def successors(s):
        nonlocal calls,generated
        calls+=1;n=t.neighbors(s);generated+=len(n)
        return dict(n)
    def offer(end,path):
        nonlocal stamp
        ng=g+len(path)
        if ng<best.get(end,10**12):
            best[end]=ng;parent[end]=(s,path);stamp+=1
            heappush(heap,(ng+t.h(end),ng,stamp,end))
    while heap:
        _,g,_,s=heappop(heap)
        if best[s]!=g:continue
        popped+=1
        if s==t.target:break
        n=successors(s);invoke=False;seq=()
        if arm in ('E','structural') and (not first or popped==1):
            seq=() if view else suffix(t,s,model['book'])
            if (view and model['corridor_support'] and len(n)>0) or seq:
                queries+=1
                if arm=='structural':invoke=True
                else:
                    x=features(t,s,n,seq)
                    invoke=predict(model['app_tree'],x)>=.5 and predict(model['tree'],x)>tau
        if invoke:
            activated+=1
            if view:
                for a,nxt in n.items():
                    r=view.corridor(s,a)
                    if r is None:
                        attempts+=1;offer(nxt,(a,));continue
                    # A goal in a corridor is reachable before its junction.
                    k=r.visited.index(t.target)+1 if t.target in r.visited else len(r.path)
                    attempts+=k
                    offer(r.visited[k-1],r.path[:k]);certified+=int(k>1)
                continue
            if len(seq)==t.lower(s):
                cur=s
                for a in seq:
                    attempts+=1
                    cur=successors(cur).get(a)
                    if cur is None:break
                if cur==t.target:
                    offer(cur,seq);certified+=1
                    # A certified completion dominates every other continuation from s.
                    continue
        for path in opts:
            cur=s
            for a in path:
                attempts+=1;cur=successors(cur).get(a)
                if cur is None:break
            if cur is not None:offer(cur,path)
        for a,nxt in n.items():
            attempts+=1
            if arm!='efficient':nxt=successors(s)[a]
            offer(nxt,(a,))
    else:
        raise AssertionError(f'unsolved fixture {t.id}')
    s=t.target;edges=[]
    while s!=t.start:
        s,path=parent[s];edges.append(path)
    actions=tuple(a for path in reversed(edges) for a in path)
    assert replay(t,actions),t.id
    checks=view.counts if view else {}
    result=dict(actions=actions,steps=len(actions),popped=popped,attempts=attempts,
                generated=generated,neighbor_calls=calls,selector_calls=queries,activated=activated,
                certified=certified,cell_reads=checks.get('reads',0),version_checks=checks.get('checks',0),
                cache_hits=checks.get('hits',0),replay_steps=len(actions))
    # Additional cell reads are priced separately from 086's primitive-attempt counter.
    result['charged']=attempts+result['cell_reads']+result['version_checks']+queries+len(actions)
    result['ns']=perf_counter_ns()-begin
    return result
