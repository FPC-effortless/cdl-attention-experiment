"""COG-ATTN-091: experience-conditioned *local* maze navigation experiment.

This tests a new local-control bridge with the pinned COG-066 acquire() function
as a diagnostic, not a complete trained COG-ATTN architecture.

Training examples are discovered by an on-policy local explorer and verified by
maze.apply(), not supplied from gold paths. Test receives only each current
cell's legal neighbors, current coordinate and goal; it never reads the free-cell
map for action selection nor the reference solutions. Full-map BFS is a separate
post-hoc oracle measurement, not a test-time tool. The resulting policy is a
small interpretable frequency table, not a neural network.
"""
from __future__ import annotations
import argparse, hashlib, json, math, random, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter_ns

HERE=Path(__file__).resolve().parent
EXTREME=HERE.parent
ROOT=EXTREME.parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(EXTREME))
from score_ladder import environment
from bridge.cog066_source import acquire

ACTION_ORDER=('U','D','L','R')
REVERSE={'U':'D','D':'U','L':'R','R':'L'}


def sign(v):return (v>0)-(v<0)
def feature(state,goal,neighbors):
    # Local perception only. Mask reports observable wall/open passages.
    return (sign(goal[0]-state[0]),sign(goal[1]-state[1]),''.join(sorted(neighbors)))

def manhattan(s,g):return abs(s[0]-g[0])+abs(s[1]-g[1])


def experience_model(traces):
    """Source-root support, not the number of frames, determines admission."""
    by_root=defaultdict(lambda:defaultdict(set))
    for t in traces:
        assert t['verified']
        for f,a in t['observations']:
            by_root[f][a].add(t['source_family'])
    # At least 2 independent training worlds required for a reusable mapping.
    return {f:{a:len(roots) for a,roots in by_action.items() if len(roots)>=2}
            for f,by_action in by_root.items() if any(len(v)>=2 for v in by_action.values())}


def score_action(choice,goal,feat,model,alpha,mode,seed):
    a,n=choice
    if mode=='random':
        # Stable RNG is supplied per action externally; not this call.
        raise AssertionError('random ranked separately')
    penalty=manhattan(n,goal)
    supports=model.get(feat,{}) if mode in ('learned','shuffled') else {}
    if mode=='shuffled':
        # Reassign each learned action to a different physical action while
        # preserving the support mass and feature inventory.
        supports={ACTION_ORDER[(ACTION_ORDER.index(k)+1)%4]:v for k,v in supports.items()}
    bonus=math.log1p(supports.get(a,0)) if supports else 0.0
    return penalty-alpha*bonus, ACTION_ORDER.index(a)


def explore(maze,mode='heuristic',model=None,alpha=0.0,seed=0,budget=None):
    """Legitimate local sensor interface: only neighbors(current) are queried.

    The stack contains unvisited cells forming one active route. Backtracking
    consumes actual moves. The explorer has episodic visited-cell memory but no
    look-ahead access to the whole graph. A budget exhaustion is a failed task.
    """
    model=model or {}
    if budget is None:budget=max(80,8*maze.size+64)
    rng=random.Random(seed)
    state=maze.start
    visited={state};stack=[state];stack_actions=[];walk=[];observations=[]
    local_calls=0;possible=0
    begin=perf_counter_ns()
    while state!=maze.target and len(walk)<budget:
        options=dict(maze.neighbors(state));local_calls+=1
        unknown=[(a,n) for a,n in options.items() if n not in visited]
        if unknown:
            obs=feature(state,maze.target,options)
            if mode=='random':
                a,n=rng.choice(unknown)
            else:
                a,n=min(unknown,key=lambda x:score_action(x,maze.target,obs,model,alpha,mode,seed))
            visited.add(n);stack.append(n);stack_actions.append(a)
            observations.append((obs,a));walk.append(a);state=n
        elif len(stack)>1:
            previous=stack[-2];a=REVERSE[stack_actions[-1]]
            # Verify every actual backtrack against observable legal actions.
            assert options[a]==previous
            walk.append(a);stack.pop();stack_actions.pop();state=previous
        else:
            break
        possible+=len(unknown)
    elapsed=perf_counter_ns()-begin
    succeeded=state==maze.target
    assert len(walk)<=budget
    if succeeded:
        assert maze.apply(walk)
        assert maze.apply(stack_actions)
    return dict(solved=succeeded,actions=len(walk),route_len=len(stack_actions) if succeeded else None,
                budget=budget,visited=len(visited),local_observations=local_calls,
                elapsed_ns=elapsed,illegal_actions=0,verified_path=list(stack_actions) if succeeded else [],
                observations=observations,walk=list(walk))


def preregistered_budget(n):return max(80,8*n+64)

def collect_training(rows):
    # Fresh training worlds only. No test-world outcome or reference labels.
    selected=[r for r in rows if r['split']=='development' and r['domain']=='maze' and r['level']<=5]
    traces=[];began=perf_counter_ns()
    for case in selected:
        maze=environment(case)
        record=explore(maze,'heuristic',seed=int(hashlib.sha256(case['id'].encode()).hexdigest()[:8],16),
                       budget=2*maze.size*maze.size)
        assert record['solved'],('training failed',case['id'])
        # Reconstruct observations from the verified final (loop-erased) route.
        state=maze.start;observations=[]
        for a in record['verified_path']:
            options=dict(maze.neighbors(state))
            assert a in options
            observations.append((feature(state,maze.target,options),a))
            state=options[a]
        assert state==maze.target
        traces.append(dict(source_family=case['input_sha256'],path=tuple(record['verified_path']),
                           observations=observations,verified=True,
                           id=case['id'],training_actions=record['actions']))
    policy=experience_model(traces)
    # Reuse actual source COG-066 skill-acquisition function, unchanged.
    options=acquire(traces)
    return traces,policy,options,perf_counter_ns()-began


def calibrate(rows,policy):
    # Calibration is different from training and reserved evaluation.
    cal=[c for c in rows if c['split']=='development' and c['domain']=='maze' and c['level'] in (6,7)]
    results={}
    for alpha in (0.0,0.25,0.75,1.5,3.0):
        records=[]
        for c in cal:
            m=environment(c)
            records.append(explore(m,'learned',policy,alpha,budget=preregistered_budget(m.size)))
        # Pick maximum successful completions; tie-break on fewer traversals and
        # then smaller skill weight. This freezes control prior to held-out set.
        results[str(alpha)]=dict(successes=sum(r['solved'] for r in records),
                                 actions=sum(r['actions'] for r in records))
    alpha=min((float(k) for k in results),key=lambda x:(-results[str(x)]['successes'],
                                                          results[str(x)]['actions'],x))
    return alpha,results


def run(cases_path,out_path):
    cases=[json.loads(x) for x in Path(cases_path).read_text().splitlines() if x.strip()]
    assert len(cases)==210
    traces,model,options,train_ns=collect_training(cases)
    selected_alpha,calibration=calibrate(cases,model)
    reserved=[c for c in cases if c['split']=='reserved' and c['domain']=='maze']
    assert len(reserved)==20
    rows=[]
    for case in reserved:
        maze=environment(case)
        for world,instance in [('original',maze)]+([('shift',environment({**case,'input':case['rules']['shifted_input']}))]
                            if 'shifted_input' in case['rules'] else []):
            seed=int(hashlib.sha256((case['id']+world).encode()).hexdigest()[:8],16)
            for arm in ('heuristic','learned','shuffled','random'):
                r=explore(instance,arm,model,selected_alpha,seed,preregistered_budget(instance.size))
                # Internal traces are not included in public summary. There is
                # no gold-reference path at any point of policy execution.
                rows.append({**{k:case[k] for k in ('id','level','split','domain')},'world':world,
                             'arm':arm,'alpha':selected_alpha,'size':instance.size,
                             **{k:v for k,v in r.items() if k not in ('walk','observations','verified_path')}})
    summary={}
    for arm in ('heuristic','learned','shuffled','random'):
        rr=[x for x in rows if x['arm']==arm];ok=[x for x in rr if x['solved']]
        summary[arm]=dict(total=len(rr),solved=len(ok),total_actions=sum(x['actions'] for x in rr),
                          median_actions=statistics.median(x['actions'] for x in rr),
                          total_elapsed_ms=round(sum(x['elapsed_ns'] for x in rr)/1e6,3))
    levels={str(l):{arm:{'solved':sum(x['solved'] for x in rows if x['level']==l and x['arm']==arm),
                         'total':sum(1 for x in rows if x['level']==l and x['arm']==arm),
                         'actions':sum(x['actions'] for x in rows if x['level']==l and x['arm']==arm)}
                   for arm in summary}for l in range(1,11)}
    result=dict(study='COG-ATTN-091: self-explored local maze skill transfer',
                model='new transparent count-based local controller, plus unchanged COG-066 acquire diagnostic; not full COG-ATTN model',
                train_worlds=len(traces),training_source='local interaction with maze, verified successful episode; no reference paths',
                training_steps=sum(x['training_actions'] for x in traces),
                train_ms=round(train_ns/1e6,3),
                skill_features=len(model),skill_pairs=sum(map(len,model.values())),
                acquired_options=len(options),option_lengths=[len(x) for x in options],
                skill_source='research/cog-081-086/bridge/cog066_source.py SHA pinned in predecessor PR',
                selected_alpha=selected_alpha,calibration=calibration,
                budget='max(80,8*n+64) actual actions including backtracking, selected before reserve',
                summary=summary,by_level=levels,rows=rows,
                limitations=['only 20 held-out maze worlds and 6 shifted-world replans; exploratory descriptive results',
                             'reference BFS not invoked in test policy or training',
                             'local 3-field count policy is not an end-to-end neural COG-ATTN agent',
                             'COG-066 options are learned as a diagnostic, but are not executed by this local policy',
                             'timings include policy/observation but exclude environment generation and training',
                             'full-map goal coordinate supplied; per-step walls observed locally'])
    out=Path(out_path);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cases',default=str(EXTREME/'results'/'cases.jsonl'))
    p.add_argument('--out',default=str(HERE/'results'/'maze-091.json'))
    args=p.parse_args();r=run(args.cases,args.out)
    print(json.dumps({k:r[k] for k in ('train_worlds','training_steps','skill_features','skill_pairs','acquired_options','selected_alpha','summary')},indent=2))
