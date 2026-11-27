"""Aggregate only complete, paired, source- and data-locked terminal validation runs."""
import argparse,json,math
from recover009 import source as expected_source
from pathlib import Path
import numpy as np
ARMS=('gpt','HH','A0','U','UA','AU')
SEEDS=(9001,9011,9029)
TARGETS=100663296
def main(root,out):
    paths=sorted(Path(root).rglob('validation.json'));assert len(paths)==18,len(paths)
    results={}
    for p in paths:
        r=json.loads(p.read_text());k=(r['arm'],r['seed']);assert k not in results
        c=r['checkpoint'];assert c['step']==8192 and c['targets']==TARGETS and c['lock']['arm']==k[0] and c['lock']['seed']==k[1];assert r['eval']['targets']>0
        assert abs(c['parameters']-1077504)/1077504<.012
        results[k]=r
    assert set(results)=={(a,s) for a in ARMS for s in SEEDS}
    assert len({r['data_manifest'] for r in results.values()})==1
    assert len({r['checkpoint']['lock']['data'] for r in results.values()})==1
    for (arm,seed),r in results.items():
        assert r['checkpoint']['lock']['source']==expected_source(arm),(arm,seed,'unreviewed source change')
    for s in SEEDS:assert len({results[a,s]['checkpoint']['lock']['data_order'] for a in ARMS})==1
    arrays={k:np.array(r['eval']['per_document'],np.float64) for k,r in results.items()}
    reference=arrays['gpt',9001]
    for k,x in arrays.items():
        assert x.shape==reference.shape and np.array_equal(x[:,2],reference[:,2]) and np.isfinite(x).all()
        assert int(x[:,2].sum())==results[k]['eval']['targets']
        assert abs(x[:,0].sum()/x[:,2].sum()/math.log(2)-results[k]['eval']['bpb'])<1e-7
    rng=np.random.default_rng(9013909);idx=rng.integers(0,len(reference),(10000,len(reference)))
    def paired(a,b,s):
        x=arrays[a,s];y=arrays[b,s];d=(x[:,0]-y[:,0]);n=x[:,2]
        boots=d[idx].sum(1)/n[idx].sum(1)/math.log(2)
        return {'delta_bpb':float(d.sum()/n.sum()/math.log(2)),'document_bootstrap_95pct':np.quantile(boots,[.025,.975]).tolist()}
    table=[]
    for a in ARMS:
        scores=[results[a,s]['eval']['bpb'] for s in SEEDS]
        table.append({'arm':a,'parameters':[results[a,s]['checkpoint']['parameters'] for s in SEEDS],'bpb_per_seed':scores,'mean_bpb':float(np.mean(scores)),'training_seconds':[results[a,s]['checkpoint']['training_seconds'] for s in SEEDS],'paired_vs_gpt':{str(s):paired(a,'gpt',s) for s in SEEDS},'paired_vs_A0':{str(s):paired(a,'A0',s) for s in SEEDS}})
    report={'status':'TERMINAL_LANGUAGE_COMPARISON_ONLY','fits':18,'predictions_per_fit':TARGETS,'total_predictions':18*TARGETS,'targets_per_validation':int(reference[:,2].sum()),'validation_manifest':next(iter(results.values()))['data_manifest'],'arms':table,'interpretation':'Paired document bootstrap within seed; not a confidence interval over training seeds or proof of architectural superiority. Downstream live-state, semantic transfer, open-world and 10M qualification are separate and not inferred from these scores.'}
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--out',required=True);a=p.parse_args();main(a.input,a.out)
