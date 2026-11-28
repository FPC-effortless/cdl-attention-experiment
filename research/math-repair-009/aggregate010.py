"""All 36 exact terminal held receipts required; article-group bootstrap, conditional on seeds."""
import argparse, json, math, hashlib
from pathlib import Path
import numpy as np
ARMS=('gpt','HH','A0','U','UA','AU');SEEDS=(9001,9011,9029);INITS=('pretrained','fresh')
def main(a):
    root=Path(a.input);results={}
    for p in root.rglob('evaluation.json'):
        v=json.loads(p.read_text());k=(v['arm'],v['seed'],v['init']);assert k not in results;k0=v['model'];assert k0['step']==4096 and k0['qa_examples']==65536 and k0['replay_targets']==12582912;assert (k0['lock']['arm'],k0['lock']['seed'],k0['lock']['init'])==k;results[k]=v
    expected={(a,s,i) for a in ARMS for s in SEEDS for i in INITS};assert set(results)==expected,(len(results),expected-set(results))
    ref=next(iter(results.values()));ids=[r['id'] for r in ref['records']];titles=[r['title'] for r in ref['records']];assert len(set(ids))==len(ids);groups,group_idx=np.unique(titles,return_inverse=True);ng=len(groups);assert ng>=20
    assert len({v['model']['lock']['qa_manifest'] for v in results.values()})==1
    assert len({json.dumps(v['model']['lock']['source'],sort_keys=True) for v in results.values()})==1
    for k,v in results.items():
        assert [r['id'] for r in v['records']]==ids and [r['title'] for r in v['records']]==titles
        for side in ('correct_source','wrong_source'):
            for metric in ('em','f1'):assert math.isclose(np.mean([r[side][metric] for r in v['records']]),v['summary'][side][metric],abs_tol=1e-12)
    for s in SEEDS:
        assert len({(v['model']['lock']['order'],v['model']['lock']['replay_order'],v['model']['answer_targets']) for k,v in results.items() if k[1]==s})==1
    rng=np.random.Generator(np.random.PCG64(101001));counts=np.bincount(group_idx,minlength=ng);draw=rng.integers(ng,size=(10000,ng));den=counts[draw].sum(1)
    def ci(x):
        group_sums=np.bincount(group_idx,weights=x,minlength=ng);b=group_sums[draw].sum(1)/den;return [float(v) for v in np.quantile(b,[.025,.975])]
    comparisons=[]
    for arm in ARMS:
        row={'arm':arm,'seeds':[]};ds=[];source=[]
        for s in SEEDS:
            p,f=(results[(arm,s,i)] for i in INITS);x=np.array([r['correct_source']['em'] for r in p['records']]);y=np.array([r['correct_source']['em'] for r in f['records']]);w=np.array([r['wrong_source']['em'] for r in p['records']]);d=x-y;c=x-w;ds.append(d);source.append(c)
            row['seeds'].append(dict(seed=s,pretrained=p['summary'],fresh=f['summary'],pretrained_minus_fresh_em=float(d.mean()),conditional_article_bootstrap=ci(d),correct_minus_wrong_source_em=float(c.mean()),source_bootstrap=ci(c),continued_language_dev_bpb=p['model']['language_dev_bpb']))
        d=np.mean(ds,0);c=np.mean(source,0);row.update(mean_pretrained_em=float(np.mean([r['pretrained']['correct_source']['em'] for r in row['seeds']])),mean_fresh_em=float(np.mean([r['fresh']['correct_source']['em'] for r in row['seeds']])),mean_pretrained_minus_fresh_em=float(d.mean()),conditional_article_bootstrap=ci(d),mean_correct_minus_wrong_source_em=float(c.mean()),source_bootstrap=ci(c));comparisons.append(row)
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(dict(status='TERMINAL_QA_TRANSFER_ONLY_NOT_FULL_STATE_QUALIFICATION',fits=36,held_questions=len(ids),held_articles=ng,qa_presentations_per_fit=65536,replay_targets_per_fit=12582912,comparisons=comparisons,uncertainty='Article bootstrap conditional on three fitted seeds; not population uncertainty across training seeds'),indent=2,allow_nan=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--out',required=True);main(p.parse_args())
