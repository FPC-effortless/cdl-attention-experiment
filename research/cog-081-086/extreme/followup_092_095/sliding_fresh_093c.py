"""COG-093c: disjoint and preregistered frozen-policy 8-puzzle replication.

Reference distance generation is quarantined here; the policy imported from
sliding_fast_093b gets only the complete grid, goal, legal transitions and the
frozen source-root-learned tie features. No reference path/label passed.
"""
from __future__ import annotations
import json,random,sys,statistics
from collections import Counter,defaultdict
from pathlib import Path
from time import perf_counter_ns
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parent))
from sliding_fast_093b import fast
from sliding_093 import search,learn
from difficulty import sliding_distance_buckets,make_slide_exact,digest,encode_puzzle
from score_ladder import environment
SEEDS=(11003,11027,11057,11071)
DEPTHS=(4,6,8,12,16,20,24,28)

def run():
    cases=[json.loads(x) for x in (HERE.parent/'results'/'cases.jsonl').read_text().splitlines()]
    dev=[c for c in cases if c['domain']=='sliding3' and c['split']=='development']
    training=[]
    for c in dev:
        r=search(environment(c),limit=250000)
        if r['solved']:training.append(dict(root=c['input_sha256'],solved=True,features=r['features']))
    model=learn(training)
    weights={(int(p[0]),int(p[1])*int(p[0])+int(p[2]),int(p[3]),p[4]):v
             for f,v in model.items() for p in [f.split('/')]}
    buckets=sliding_distance_buckets()
    existing={c['input_sha256'] for c in cases if c['domain']=='sliding3'}
    cohort=[]
    for seed in SEEDS:
        for i,depth in enumerate(DEPTHS):
            for retry in range(30):
                p=make_slide_exact(seed*100003+i*117+retry*10000019,depth,buckets)
                sha=digest(('sliding3',encode_puzzle(p)))
                if sha not in existing:break
            else:raise AssertionError('could not find disjoint exact-depth state')
            existing.add(sha)
            cohort.append(dict(seed=seed,depth=depth,instance=p,input_sha256=sha))
    rows=[];timings=[]
    for rep in range(3):
        for c in cohort:
            order=['primitive','learned','shuffled'];random.Random(rep*10**6+c['seed']*10+c['depth']).shuffle(order)
            for arm in order:
                r=fast(c['instance'],weights,arm,limit=40000)
                if rep==0:rows.append(dict(seed=c['seed'],depth=c['depth'],sha=c['input_sha256'],arm=arm,
                                           **{k:v for k,v in r.items() if k!='elapsed_ns'}))
                else:
                    prior=next(z for z in rows if z['sha']==c['input_sha256'] and z['arm']==arm)
                    assert all(r[k]==prior[k] for k in ('solved','expanded','attempts','actions'))
                timings.append(dict(rep=rep,seed=c['seed'],depth=c['depth'],arm=arm,elapsed_ns=r['elapsed_ns']))
    summary={arm:dict(n=sum(x['arm']==arm for x in rows),
                      solved=sum(x['solved'] for x in rows if x['arm']==arm),
                      expanded=sum(x['expanded'] for x in rows if x['arm']==arm),
                      attempts=sum(x['attempts'] for x in rows if x['arm']==arm),
                      total_ms_median=round(statistics.median(
                        sum(x['elapsed_ns'] for x in timings if x['rep']==rep and x['arm']==arm)/1e6 for rep in range(3)),3))
             for arm in ('primitive','learned','shuffled')}
    comparisons=[]
    for c in cohort:
        local={x['arm']:x for x in rows if x['sha']==c['input_sha256']}
        comparisons.append(dict(seed=c['seed'],depth=c['depth'],
            ratio=(local['learned']['expanded']/local['primitive']['expanded'] if local['primitive']['expanded'] else None),
            won=local['learned']['expanded']<local['primitive']['expanded'],
            lost=local['learned']['expanded']>local['primitive']['expanded']))
    result=dict(experiment='COG-093c disjoint exact-depth 8-puzzle replication',
                seeds=list(SEEDS),depths=list(DEPTHS),cohort='32 novel states distinct from original 210-case suite',
                model_frozen_from_093=True,model_support_features=len(weights),
                summary=summary,improved_states=sum(x['won'] for x in comparisons),
                worsened_states=sum(x['lost'] for x in comparisons),
                equal_states=sum(not x['won'] and not x['lost'] for x in comparisons),
                comparisons=comparisons,rows=rows,timings=timings,
                limitations=['known exact distances are task-generation metadata only, never used by A* control',
                   'the 32 tasks share the 8-puzzle transition rules and training domain',
                   'no prospective human trial, full neural controller, or general reasoning certification',
                   'CPU times include search but not training or puzzle generation'])
    (HERE/'results/sliding-093c.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:result[k] for k in ('summary','improved_states','worsened_states','equal_states')},indent=2))
    return result
if __name__=='__main__':run()
