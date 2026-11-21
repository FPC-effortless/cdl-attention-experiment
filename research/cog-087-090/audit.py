"""Independent archived-path and confidence-interval arithmetic."""
import itertools,json,random,statistics,zipfile
from hashlib import sha256
from pathlib import Path
from panel import ROOT,edges,inputs,distance
from tasks import BASE,tasks
from subgoal import cohort

def read(name):return json.loads((ROOT/'results'/name).read_bytes())
def interval(groups):
    rng=random.Random(900837);ratios=[]
    for _ in range(5000):
        draw=rng.choices(groups,k=len(groups))
        ratios.append(sum(x for x,y in draw)/sum(y for x,y in draw))
    ratios.sort()
    return [ratios[124],ratios[4874]]
def summarize():
    model=read('model.json');seen=set(model['dev_digests']+model['train_digests']);rows=[]
    goals={(t.kind,t.target) for k in ('train','dev') for t in tasks(k) if t.kind!='maze'}
    for part in ('held','rep'):
        ref=inputs(part,model);panel=read(part+'.json')
        assert panel['model_sha256']==sha256((ROOT/'results/model.json').read_bytes()).hexdigest()
        for t,row in zip(ref,panel['tasks'],strict=True):
            assert t.digest()==row['fixture'] and t.digest() not in seen
            assert t.kind=='maze' or (t.kind,t.target) not in goals
            seen.add(t.digest());best=distance(t)
            for r in row['arms'].values():
                assert r['steps']==best==r['replay_steps']==len(r['actions'])
                s=t.start
                for a in r['actions']:s=dict(edges(t,s))[tuple(a) if isinstance(a,list) else a]
                assert s==t.target
            rows.append(row)
    out={}
    for name,predicate in [('all',lambda x:True),('held',lambda x:x['id'].startswith('held/')),
                           ('rep',lambda x:x['id'].startswith('rep/')),('maze',lambda x:x['kind']=='maze'),
                           ('slide',lambda x:x['kind']=='slide'),('hanoi',lambda x:x['kind']=='hanoi'),
                           ('four_peg',lambda x:x['pegs']==4)]:
        rr=[r for r in rows if predicate(r)]
        by={}
        for arm in rr[0]['arms']:
            keys=('attempts','charged','popped','ns','generated','certified','activated','selector_calls','version_checks','cell_reads')
            by[arm]={k:sum(r['arms'][arm][k] for r in rr) for k in keys}
        groups=[]
        for seed in sorted({r['id'].split('/')[1] for r in rr}):
            subset=[r for r in rr if r['id'].split('/')[1]==seed]
            groups.append([sum(r['arms'][a]['ns'] for r in subset) for a in ('A','E')])
        out[name]=dict(n=len(rr),unique_goal_vectors=len({tuple(r['target']) for r in rr}),arms=by,time_ratio=by['A']['ns']/by['E']['ns'],ci=interval(groups))
    shifts=read('shifts.json')
    with zipfile.ZipFile(BASE/'artifacts/puzzle-081-086-full-evidence.zip') as z:
        old=json.loads(z.read('cog_attn_puzzle_081/results.json'))['shifted']
    assert {(r['seed'],r['trial'],r['old_length'],r['new_length']) for r in old}=={(r['seed'],r['trial'],r['old_steps'],r['new_steps']) for r in shifts}
    assert len(shifts)==36 and all(all(r[k] for k in ('retained','invalidated','old_rejected','goal_rejected','restored','hidden_deferred','offpath_rejected')) for r in shifts)
    out['shifts']={k:sum(r[k] for r in shifts) for k in ('retained','invalidated','old_rejected','offpath_rejected','hidden_deferred','restored')}
    sub=read('subgoal.json')
    for t,r in zip(cohort('eval'),sub,strict=True):
        assert t.digest()==r['digest'] and all(a['steps']==distance(t) for a in r['arms'].values())
    out['subgoal']={a:{k:sum(r['arms'][a][k] for r in sub) for k in ('steps','work','build','total','queries','ns')} for a in ('zero','cheap','always','gate')}
    return out
if __name__=='__main__':(ROOT/'results/summary.json').write_text(json.dumps(summarize(),indent=2)+'\n')
