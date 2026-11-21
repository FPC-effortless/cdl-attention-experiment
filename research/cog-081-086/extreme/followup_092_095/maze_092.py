"""COG-092: preregistered local maze skill admission and online correction.

This is an experimental controller, NOT the full neural COG-ATTN network.
The agent observes only its present cell, legal neighbors, and goal, never the
complete map or benchmark reference path. Training receives successful episodes
from its own local search, not gold actions. All traversals including backtracks
count towards the budget. The world simulator validates movement.
"""
from __future__ import annotations
import json,random,hashlib,sys,math,statistics
from pathlib import Path
from collections import Counter
from time import perf_counter_ns
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(HERE.parent))
sys.path.insert(0,str(HERE.parent/'learning'))
from score_ladder import environment
from difficulty import make_maze_challenge,maze_changed,digest,encode_puzzle
from maze_transfer_091 import feature,manhattan,experience_model,collect_training,calibrate,explore,preregistered_budget,REVERSE,ACTION_ORDER

SEEDS=(9901,9907,9913,9923)
CAL_LEVELS=(6,7)


def navigate(m,model,alpha,threshold=2,freeze_fraction=1.0,correction=0.0,budget=None):
    """Only local neighbor access; learned action admitted conditionally.

    An explored branch is marked unproductive only upon physical backtrack.
    The online negative outcome is available without a gold/reference path.
    """
    budget=preregistered_budget(m.size) if budget is None else budget
    state=m.start;seen={state};stack=[state];parents=[];walk=[];bad=Counter();updates=0;gate_hits=0;observations=0
    t0=perf_counter_ns()
    while state!=m.target and len(walk)<budget:
        nbs=dict(m.neighbors(state)); observations+=1
        unknown=[(a,ns) for a,ns in nbs.items() if ns not in seen]
        if unknown:
            feat=feature(state,m.target,nbs)
            supports=model.get(feat,{})
            hero=min(unknown,key=lambda x:(manhattan(x[1],m.target),ACTION_ORDER.index(x[0])))
            # Effective source-root support for the proposed action; no support => default.
            candidates=sorted(unknown,key=lambda z:(manhattan(z[1],m.target)-alpha*math.log1p(supports.get(z[0],0))
                                                    +correction*bad[feat,z[0]],ACTION_ORDER.index(z[0])))
            learned=candidates[0]
            support=supports.get(learned[0],0)
            use_learned=(support>=threshold and len(walk)<budget*freeze_fraction)
            a,n=learned if use_learned else hero
            gate_hits+=int(use_learned)
            seen.add(n); stack.append(n);parents.append((feat,a));walk.append(a);state=n
        elif len(stack)>1:
            feat,a0=parents.pop();bad[feat,a0]+=1;updates+=1
            prior=stack[-2];a=REVERSE[a0]
            assert nbs[a]==prior
            stack.pop();state=prior;walk.append(a)
        else:break
    success=state==m.target
    if success:
        assert m.apply(walk) and m.apply([z[1] for z in parents])
    return dict(solved=success,actions=len(walk),visited=len(seen),online_corrections=updates,
                gate_decisions=gate_hits,local_observations=observations,elapsed_ns=perf_counter_ns()-t0,
                illegal_actions=0,success_route=[z[1] for z in parents] if success else [])


def load_cases():
    p=HERE.parent/'results'/'cases.jsonl'
    return [json.loads(s) for s in p.read_text().splitlines() if s]


def calibration(cases,model,alpha):
    dev=[x for x in cases if x['domain']=='maze' and x['split']=='development' and x['level'] in CAL_LEVELS]
    assert len(dev)==4
    candidates=[]
    for threshold in (2,3,4):
        for fraction in (0.25,0.6,1.0):
            for correction in (0.0,1.0):
                vals=[]
                for c in dev:
                    m=environment(c)
                    for factor in (1,2):
                        vals.append(navigate(m,model,alpha,threshold,fraction,correction,preregistered_budget(m.size)*factor))
                success=sum(v['solved'] for v in vals)
                actions=sum(v['actions'] for v in vals)
                candidates.append(dict(threshold=threshold,fraction=fraction,correction=correction,
                                       solved=success,n=len(vals),actions=actions))
    selected=min(candidates,key=lambda x:(-x['solved'],x['actions'],x['threshold'],x['fraction'],x['correction']))
    return {k:selected[k] for k in ('threshold','fraction','correction')},candidates


def run():
    cases=load_cases();traces,model,opts,t=collect_training(cases)
    alpha,cal=calibrate(cases,model)
    gate,grid=calibration(cases,model,alpha)
    known={r['input_sha256'] for r in cases if r['domain']=='maze'}
    rows=[]
    for seed in SEEDS:
        for level in range(1,11):
            maze,ref=make_maze_challenge(seed*100003+level*131,level)
            sha=digest(('maze',encode_puzzle(maze)));assert sha not in known
            known.add(sha)
            changed=None
            if level>=8:
                changed,_=maze_changed(maze,ref.path)
                assert changed is not None
            for label,m in [('original',maze)]+([('shift',changed)] if changed else []):
                for factor in (1,2):
                    b=preregistered_budget(m.size)*factor
                    for arm in ('heuristic','learned','gated','corrected'):
                        if arm in ('heuristic','learned'):
                            r=explore(m,mode=arm,model=model,alpha=alpha,budget=b)
                        else:
                            r=navigate(m,model,alpha,gate['threshold'],gate['fraction'],
                                       gate['correction'] if arm=='corrected' else 0.0,b)
                        rows.append(dict(seed=seed,level=level,world=label,factor=factor,arm=arm,
                                         input_sha256=sha,**{k:v for k,v in r.items()
                                                              if k not in ('walk','observations','verified_path','success_route')}))
    arms=['heuristic','learned','gated','corrected']
    summary={str(f):{arm:dict(n=sum(r['factor']==f and r['arm']==arm for r in rows),
                                solved=sum(r['solved'] for r in rows if r['factor']==f and r['arm']==arm),
                                actions=sum(r['actions'] for r in rows if r['factor']==f and r['arm']==arm),
                                policy_ms=round(sum(r['elapsed_ns'] for r in rows if r['factor']==f and r['arm']==arm)/1e6,3))
                       for arm in arms}for f in (1,2)}
    report=dict(experiment='COG-ATTN-092: conditional admission and local correction',
       source='actual 091 self-explored source-root model; new local policy, not full trained COG-ATTN',
       frozen_training_worlds=len(traces),alpha_091=alpha,admission=gate,development_calibration=grid,
       held_seeds=list(SEEDS),summary=summary,rows=rows,
       limitations=['heldout generated from same maze family','shift tasks correlated with each source maze',
                    'feature support comes from 10 solved training worlds, not an autonomous pretrained model',
                    'policy CPU timings host-dependent; no network/learning cost inside inference',
                    'reference BFS only used for OFFLINE task generation and shift construction, never policy execution'])
    p=HERE/'results';p.mkdir(exist_ok=True)
    (p/'maze-092.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'admission':gate,'summary':summary},indent=2))
    return report
if __name__=='__main__':run()
