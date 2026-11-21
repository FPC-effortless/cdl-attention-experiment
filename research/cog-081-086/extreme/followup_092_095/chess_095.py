"""COG-095 restricted KQK/KRK: self-search-derived White move ordering.

This chess agent gets only legal moves and terminal mate observations, and is
charged every explored White state and candidate. All Black legal replies must
be covered by the resulting strategy. No reference mate table, answer label,
provided successful strategy, or force_mate oracle is used for search/training.

The algorithmic AND/OR search is GIVEN and only move-order policy is learned;
this does not test whether COG-ATTN learned chess rules or minimax itself.
"""
from __future__ import annotations
import json,sys,hashlib,math
from pathlib import Path
from collections import defaultdict,Counter
from time import perf_counter_ns
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE.parent))
from score_ladder import environment,score_chess
from difficulty import encode_puzzle
from puzzles import ChessEndgame

class Exhausted(Exception):pass

def move_feat(p,name,old,dest,child):
    # Legal Black replies are an environmental consequence of the White move.
    # No mate check in feature construction; terminal feedback in search only.
    replies=child.black_replies()
    return f'{p.kind}/{name}/{int(child.black_checked())}/{min(4,len(replies))}'

def search(p,depth,arm='primitive',model=None,node_budget=2000):
    begin=perf_counter_ns();nodes=0;attempts=0;terminal=0;cache={};model=model or {}
    def recurse(s,d):
        nonlocal nodes,attempts,terminal
        key=(s.wk,s.heavy,s.bk,s.kind,d)
        if key in cache:return cache[key]
        if nodes>=node_budget:raise Exhausted()
        nodes+=1
        opts=s.white_moves()
        # Features involve a complete 1-step observation of each legal move's
        # Black legal replies. Charged explicitly, even if later unexpanded.
        if arm in ('learned','shuffled'):
            scored=[]
            for ix,(name,old,new,child) in enumerate(opts):
                f=move_feat(s,name,old,new,child)
                if arm=='shuffled':
                    parts=f.split('/');parts[1]='K' if parts[1]!='K' else s.kind
                    f='/'.join(parts)
                scored.append((-model.get(f,0),ix,(name,old,new,child)))
            opts=[x[-1] for x in sorted(scored)]
        for name,old,new,child in opts:
            attempts+=1
            terminal+=1
            if child.mate():
                plan={encode_puzzle(s)['fen']:[name,old,new]}
                cache[key]=plan;return plan
            if d<=1:continue
            black=child.black_replies()
            if not black:continue   # stalemate is not a win
            plans=[];failed=False
            for bk in black:
                if bk==child.heavy:failed=True;break  # captured piece = draw
                nxt=ChessEndgame(child.wk,child.heavy,bk,child.kind)
                result=recurse(nxt,d-1)
                if result is None:failed=True;break
                plans.append(result)
            if not failed:
                plan={encode_puzzle(s)['fen']:[name,old,new]}
                for pp in plans:plan.update(pp)
                cache[key]=plan;return plan
        cache[key]=None;return None
    try:
        strategy=recurse(p,depth)
        exhausted=False
    except Exhausted:
        strategy=None;exhausted=True
    success=strategy is not None
    if success:assert score_chess(p,{'strategy':strategy},depth)
    return dict(solved=success,exhausted=exhausted,nodes=nodes,attempts=attempts,
                terminal_checks=terminal,elapsed_ns=perf_counter_ns()-begin,
                first_move=strategy[encode_puzzle(p)['fen']] if success else None,
                strategy_size=len(strategy) if success else 0)


def train(cases):
    dev=[x for x in cases if x['split']=='development' and x['domain']=='chess' and x['level']<=6]
    roots=defaultdict(set);runs=[]
    for c in dev:
        p=environment(c);depth=c['rules']['maximum_required_mate_in']
        r=search(p,depth,node_budget=2500)
        runs.append(dict(id=c['id'],depth=depth,solved=r['solved'],nodes=r['nodes']))
        if r['solved']:
            name,old,new=r['first_move'];child=next(ch for a,b,z,ch in p.white_moves() if (a,b,z)==(name,old,new))
            feat=move_feat(p,name,old,new,child)
            roots[feat].add(c['input_sha256'])
    model={f:len(srcs) for f,srcs in roots.items() if len(srcs)>=2}
    return model,runs


def run():
    cases=[json.loads(x) for x in (HERE.parent/'results'/'cases.jsonl').read_text().splitlines()]
    model,train_rows=train(cases)
    rows=[]
    for c in cases:
        if c['split']!='reserved' or c['domain']!='chess':continue
        p=environment(c);depth=c['rules']['maximum_required_mate_in']
        for arm in ('primitive','learned','shuffled'):
            r=search(p,depth,arm,model,node_budget=2500)
            rows.append(dict(id=c['id'],level=c['level'],arm=arm,depth=depth,
                             **{k:v for k,v in r.items() if k!='first_move'}))
    summary={arm:dict(solved=sum(x['solved'] for x in rows if x['arm']==arm),
                      exhausted=sum(x['exhausted'] for x in rows if x['arm']==arm),
                      n=sum(x['arm']==arm for x in rows),
                      nodes=sum(x['nodes'] for x in rows if x['arm']==arm),
                      attempts=sum(x['attempts'] for x in rows if x['arm']==arm),
                      policy_ms=round(sum(x['elapsed_ns'] for x in rows if x['arm']==arm)/1e6,3))
             for arm in ('primitive','learned','shuffled')}
    report=dict(experiment='COG-ATTN-095: opponent-robust KQK/KRK local learning diagnostic',
                training_scope='development levels 1-6; 12 positions; self-searched AND/OR forced-mate strategies',
                training_rows=train_rows,learned_features=model,
                node_budget=2500,summary=summary,rows=rows,
                limitations=['NOT full chess; KQK/KRK with White to move only',
                'AND/OR search and legal chess primitives explicitly programmed, not acquired',
                'training labels are self-search terminal mate outcomes, no evaluator-supplied winning moves',
                'only 20 reserved chess cases; roots from same constrained generator',
                'budget-exhausted searches fail; no provided mate oracle at inference'])
    out=HERE/'results';out.mkdir(exist_ok=True)
    (out/'chess-095.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'training':len(train_rows),'learned_features':model,'summary':summary},indent=2))
    return report
if __name__=='__main__':run()
