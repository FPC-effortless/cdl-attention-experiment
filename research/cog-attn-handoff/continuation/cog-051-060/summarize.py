import json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
R=Path(__file__).resolve().parent/'results'
def get(p):return json.loads((R/p).read_text())
def med(x):return float(np.median(x))
def run():
    out={}
    candidates=[get(f'{s}/acquisition-candidate/result.json') for s in (5101,5107,5113)]
    trials=np.array([v for c in candidates for v in c['panel']['trials']])
    out['controlled']=dict(discovery=np.sum([np.sum(c['panel']['discovery'],axis=0) for c in candidates],axis=0).tolist(),
        scenes=24,language={k:{arm:sum(c['language']['accuracy_counts'][k][arm] for c in candidates)
            for arm in candidates[0]['language']['accuracy_counts'][k]} for k in candidates[0]['language']['accuracy_counts']},
        paths=len(trials),predicted_steps=trials[:,2].mean(),random_steps=trials[:,3].mean(),planned_correct=int(trials[:,5].sum()),random_correct=int(trials[:,4].sum()),
        trace_correct=int(trials[:,6].sum()),cache_hits=int(trials[:,7].sum()),speed=trials[:,8].sum()/trials[:,9].sum(),
        unchanged_bad=sum(c['panel']['stale_without_signal'] for c in candidates),replanned=sum(c['panel']['replanned_after_signal'] for c in candidates),
        retained=sum(sum(c['panel']['retention']) for c in candidates),overwrite=sum(sum(c['panel']['overwrite']) for c in candidates),
        session=[sum(t['exact'] for t in c['session']) for c in candidates],routes=[c['route_counts'] for c in candidates])
    out['novel_skills']={}
    supp=get('discovery-skills.json')
    rows=np.array([r for c in supp.values() for r in c['skills']['held']])
    out['novel_skills']=dict(queries=len(rows),macro_used=int(rows[:,4].sum()),correct=int(rows[:,3].sum()),
        primitive_steps=rows[:,0].mean(),macro_decisions=rows[:,1].mean(),physical_steps=rows[:,2].mean(),speed=rows[:,5].sum()/rows[:,6].sum())
    costs=get('online-cost.json')
    out['latency']={}
    for n in sorted({r['size'] for r in costs}):
        byarm={a:sorted([r for r in costs if r['size']==n and r['arm']==a],key=lambda r:r['seed']) for a in ('index','scoped_cached','scoped_sdpa')}
        out['latency'][n]={a:med([r['online_ns']/b['online_ns'] for r,b in zip(byarm[a],byarm['index'])]) for a in byarm}
        out['latency'][n]['weighted_cached']=med([r['score_ns']/b['score_ns'] for r,b in zip(byarm['scoped_cached'],byarm['index'])])
    stats=[get(f'{s}/issue-results.json') for s in (67003,67009,67021)]
    out['issue']={arm:{split:{'ap':med([r[arm][split]['ap'] for r in stats]),
        'tp_fp':[[r[arm][split]['tp'],r[arm][split]['fp']] for r in stats]} for split in ('test','external')} for arm in stats[0]}
    preds=get('67003/issue-predictions.json');out['intervals']={}
    for split in ('test','external'):
        heads={a:[r for r in preds if r[0]==a and r[1]==split] for a in stats[0]}
        ids=[(r[2],r[3]) for r in heads['strict_flat']]
        assert all([(r[2],r[3]) for r in v]==ids for v in heads.values())
        group,inv=np.unique([v[0] for v in ids],return_inverse=True)
        y=np.array([r[4] for r in heads['strict_flat']]);rng=np.random.default_rng(67003)
        diffs=[[],[]]
        for _ in range(1000):
            w=np.bincount(rng.integers(len(group),size=len(group)),minlength=len(group))[inv]
            if (w*y).sum()==0:continue
            ap={a:average_precision_score(y,[r[5] for r in v],sample_weight=w) for a,v in heads.items()}
            diffs[0].append(ap['separate_expert']-ap['pooled_modular']);diffs[1].append(ap['pooled_flat']-ap['strict_flat'])
        out['intervals'][split]=[np.quantile(v,[.025,.975]).tolist() for v in diffs]
    (R/'summary.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':run()
