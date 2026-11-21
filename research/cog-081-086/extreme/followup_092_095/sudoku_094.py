"""COG-094: observe consequence of choosing Sudoku search operators.

No completed boards or cell answers are supplied in training or evaluation.
Primitives are row/column/subgrid legality and backtracking (engineered).
The learned controller only chooses between these search skills using measured
self-solve consequences, NOT a learned Sudoku constraint representation.
"""
from __future__ import annotations
import json,sys,statistics
from pathlib import Path
from collections import Counter
from time import perf_counter_ns
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE.parent))
from score_ladder import environment


def solve(p,policy='mrv',max_nodes=5000):
    start=perf_counter_ns(); nodes=0;dead=0;branches=0;actions=[];valid=False
    def dfs(s):
        nonlocal nodes,dead,branches
        if nodes>=max_nodes:return None
        nodes+=1
        # legal candidates come from explicit Sudoku constraints, not gold cells.
        slots=[(i,p.choices(s,i)) for i,v in enumerate(s) if v==0]
        if not slots:
            return [] if p.valid(s) else None
        if policy=='first': idx,opts=slots[0]
        elif policy=='mrv': idx,opts=min(slots,key=lambda item:(len(item[1]),item[0]))
        elif policy=='max': idx,opts=max(slots,key=lambda item:(len(item[1]),-item[0]))
        else:raise ValueError(policy)
        if not opts:dead+=1;return None
        if len(opts)>1:branches+=1
        for v in opts:
            nxt=list(s);nxt[idx]=v
            result=dfs(tuple(nxt))
            if result is not None:return [(idx,v)]+result
            if nodes>=max_nodes:return None
        return None
    path=dfs(p.board)
    success=path is not None and p.apply(path)
    return dict(solved=bool(success),expanded=nodes,dead_ends=dead,branch_points=branches,
                actions=len(path) if success else None,elapsed_ns=perf_counter_ns()-start)


def fit(cases):
    dev=[c for c in cases if c['domain']=='sudoku' and c['split']=='development']
    train=[]
    for c in dev:
        p=environment(c)
        for arm in ('first','mrv','max'):
            r=solve(p,arm)
            train.append(dict(id=c['id'],level=c['level'],n=p.n,arm=arm,**r))
    # Select from observed completed runs, not annotated solution labels.
    by={}
    for n in (4,9):
        pool=[r for r in train if r['n']==n]
        by[str(n)]=min(('first','mrv','max'),key=lambda arm:(
            -sum(x['solved'] for x in pool if x['arm']==arm),
            sum(x['expanded'] for x in pool if x['arm']==arm),arm))
    return by,train


def run():
    cases=[json.loads(x) for x in (HERE.parent/'results'/'cases.jsonl').read_text().splitlines()]
    policy,train=fit(cases)
    rows=[]
    for c in cases:
        if c['split']!='reserved' or c['domain']!='sudoku':continue
        p=environment(c)
        for arm in ('first','mrv','max','learned'):
            selected=policy[str(p.n)] if arm=='learned' else arm
            r=solve(p,selected)
            rows.append(dict(id=c['id'],level=c['level'],n=p.n,arm=arm,solver=selected,**r))
    summary={arm:dict(n=sum(r['arm']==arm for r in rows),
                       solved=sum(r['solved'] for r in rows if r['arm']==arm),
                       nodes=sum(r['expanded'] for r in rows if r['arm']==arm),
                       elapsed_ms=round(sum(r['elapsed_ns'] for r in rows if r['arm']==arm)/1e6,3))
             for arm in ('first','mrv','max','learned')}
    result=dict(experiment='COG-ATTN-094 outcome-selected Sudoku search operator',
                frozen_n_policies=policy,training_cases=20,training_trials=train,
                summary=summary,rows=rows,
                limitations=['legal candidates and backtracking supplied as search primitives',
                             'learned only selects amongst three given search operators',
                             'no completed Sudoku boards or gold answer labels given to policy',
                             'same puzzle generator family across development and reserved',
                             'cost of multiple training fits not amortized into runtime comparisons'])
    out=HERE/'results';out.mkdir(exist_ok=True)
    (out/'sudoku-094.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'frozen_policies':policy,'summary':summary},indent=2))
    return result
if __name__=='__main__':run()
