"""Require all 18 exact terminal fits before ranking; reused dev only."""
import argparse,json,math,hashlib,statistics
from pathlib import Path
ARMS=('native_short','native_long','time_long','write_long','scalar_long','context_long');SEEDS=(9001,9011,9029)
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();rows={};sources=set()
for p in a.input.rglob('result.json'):
 r=json.loads(p.read_text());c=r['config'];k=(c['arm'],c['seed']);assert k not in rows and k[0] in ARMS and k[1] in SEEDS
 assert r['terminal'] and r['reload_verified'] and r['targets']==c['targets']==12582912 and c['steps']==1024 and r['parameters']==1077504
 assert sha(p.parent/'checkpoint.pt')==r['checkpoint_sha256'] and sha(p.parent/'model.safetensors')==r['model_sha256'] and sha(p.parent/'development.json')==r['development_sha256']
 dev=json.loads((p.parent/'development.json').read_text());assert all(r[f'bpb_{n}']==dev[f'bpb_{n}'] and math.isfinite(r[f'bpb_{n}']) for n in (64,768,2048));assert r['full_dev_bpb_768']==dev['full_dev_bpb_768'] and math.isfinite(r['full_dev_bpb_768'])
 assert len(dev['context_rows'])==828 and r['full_dev_targets']==5029184;rows[k]=r;sources.add(json.dumps(c['source'],sort_keys=True))
assert set(rows)=={(arm,s) for arm in ARMS for s in SEEDS} and len(sources)==1
for s in SEEDS:
 assert len({rows[a,s]['config']['order_sha256'] for a in ARMS})==1 and len({rows[a,s]['config']['parent_sha256'] for a in ARMS})==1
out={a:{m:{'mean':statistics.mean(rows[a,s][m] for s in SEEDS),'seedwise':[rows[a,s][m] for s in SEEDS]} for m in ('bpb_64','bpb_768','bpb_2048','full_dev_bpb_768','training_seconds','peak_rss_kib')} for a in ARMS}
contrasts={f'{a}_vs_{b}':{m:[rows[a,s][m]-rows[b,s][m] for s in SEEDS] for m in ('bpb_64','bpb_2048','full_dev_bpb_768')} for a,b in [('native_long','native_short'),('time_long','native_long'),('write_long','native_long'),('scalar_long','write_long'),('context_long','scalar_long'),('context_long','write_long')]}
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(dict(status='DEVELOPMENT_ONLY',fits=18,results=out,paired_differences=contrasts),indent=2));print(a.out.read_text())
