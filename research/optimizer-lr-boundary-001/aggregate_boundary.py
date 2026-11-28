"""Combine six historical paired controls with four new terminal fits; no new baseline fit credit."""
import argparse,hashlib,json,math,statistics
from pathlib import Path
RATES=(.0003,.0006,.0012,.002,.003);SEEDS=(9101,9109)
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(root,new):
    rows={}
    for p in root.rglob('result.json'):
        x=json.loads(p.read_text());c=x['config'];k=(float(c['rate']),c['seed'])
        assert k not in rows and c['recipe']=='adamw' and c['architecture']=='HH' and c['steps']==1024 and c['batch']==16 and c['context']==768
        assert x['terminal'] and x['reload_verified'] and x['targets']==12582912 and x['parameters']==1077504 and x['eval_targets']==5029184 and math.isfinite(x['bpb'])
        assert sha(p.parent/'checkpoint.pt')==x['checkpoint_sha256'] and sha(p.parent/'model.safetensors')==x['model_sha256']
        receipt=json.loads((p.parent/'receipt.json').read_text());assert receipt['terminal'] and receipt['step']==1024 and receipt['targets']==12582912 and receipt['checkpoint_sha256']==x['checkpoint_sha256'] and receipt['config']==c
        dev=json.loads((p.parent/'development.json').read_text());assert dev['bpb']==x['bpb'] and dev['targets']==x['eval_targets']
        assert c['data_manifest']=='2955e5b6e772cf4539615b0585c1b522f7cf168c3c8e47c95b29dc97dda40c35'
        if new:assert 'lr_boundary_001/run_boundary.py' in c['source']
        else:assert str(c['run_id'])=='46075420494' and 'lr_boundary_001/run_boundary.py' not in c['source']
        rows[k]=x
    assert set(rows)=={(r,s) for r in (RATES[3:] if new else RATES[:3]) for s in SEEDS},(new,sorted(rows))
    return rows
p=argparse.ArgumentParser();p.add_argument('--historical',type=Path,required=True);p.add_argument('--new',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
rows={**load(a.historical,False),**load(a.new,True)}
for seed in SEEDS:
    assert len({rows[r,seed]['config']['data_order'] for r in RATES})==1
base=[{k:v for k,v in x['config']['source'].items() if not k.startswith('lr_boundary_001/')} for x in rows.values()]
assert len({json.dumps(x,sort_keys=True) for x in base})==1
assert len({(x['config']['torch'],x['config']['numpy'],x['config']['threads']) for x in rows.values()})==1
summary=[]
for r in RATES:
    q=[rows[r,s] for s in SEEDS];b=[x['bpb'] for x in q];delta=[x['bpb']-rows[.0012,s]['bpb'] for x,s in zip(q,SEEDS)]
    summary.append(dict(rate=r,bpb=statistics.mean(b),seed_bpb=b,paired_delta_vs_0012=delta,clip_fraction=[x['clipped_steps']/1024 for x in q],training_seconds=[x['train_seconds'] for x in q],optimizer_seconds=[x['optimizer_seconds'] for x in q],source='historical' if r<=.0012 else 'new'))
best=min(summary,key=lambda x:x['bpb'])
result=dict(status='DEVELOPMENT_ONLY',historical_fits=6,new_terminal_fits=4,best_tested_rate=best['rate'],upper_edge=best['rate']==.003,all_seeds_upper_improve_vs_002=all(rows[.003,s]['bpb']<rows[.002,s]['bpb'] for s in SEEDS),rows=summary,warning='Two paired seeds, reused development data, heterogeneous runner timing. No optimizer or architecture promotion. Extend only if justified; full-budget fresh-seed and capability confirmation required.')
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n');print(a.out.read_text())
