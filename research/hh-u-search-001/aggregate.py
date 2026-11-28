import argparse,hashlib,json,math,statistics
from pathlib import Path
ARMS=tuple(a+b for a in 'NSRB' for b in 'NSRB');SEEDS=(9301,9311,9329)
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();rows={}
 for f in a.input.rglob('result.json'):
  r=json.loads(f.read_text());k=(r['arm'],r['seed']);assert k not in rows and k[0] in ARMS and k[1] in SEEDS and r['terminal'] and r['reload_verified'] and r['step']==1024 and r['targets']==12582912 and r['eval_targets']==5029184 and math.isfinite(r['bpb']) and abs(r['parameters']-1077504)/1077504<.002
  assert sha(f.parent/'checkpoint.pt')==r['checkpoint_sha256'] and sha(f.parent/'model.safetensors')==r['model_sha256'];d=json.loads((f.parent/'development.json').read_text());assert d['bpb']==r['bpb'] and d['targets']==5029184;rows[k]=r
 assert set(rows)=={(a,s) for a in ARMS for s in SEEDS};assert len({json.dumps(r['lock']['source'],sort_keys=True) for r in rows.values()})==1 and len({r['lock']['data'] for r in rows.values()})==1
 for s in SEEDS:assert len({rows[a,s]['lock']['data_order'] for a in ARMS})==1
 table=[]
 for arm in ARMS:
  rr=[rows[arm,s] for s in SEEDS];ls=[r['bpb'] for r in rr];table.append(dict(arm=arm,mean_bpb=statistics.mean(ls),seed_bpb=ls,parameters=rr[0]['parameters'],delta_vs_HH=[rows[arm,s]['bpb']-rows['NN',s]['bpb'] for s in SEEDS],delta_vs_U=[rows[arm,s]['bpb']-rows['BB',s]['bpb'] for s in SEEDS],training_seconds=[r['training_seconds'] for r in rr]))
 # Factorial effect of each path across all other path-presence combinations.
 def bits(arm):return tuple(v for c in arm for v in (c in 'SB',c in 'RB'))
 effects={}
 for j,name in enumerate(('slow1','reflection1','slow2','reflection2')):
  on=[k for k in ARMS if bits(k)[j]];off=[k for k in ARMS if not bits(k)[j]];effects[name]=[statistics.mean(rows[k,s]['bpb'] for k in on)-statistics.mean(rows[k,s]['bpb'] for k in off) for s in SEEDS]
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(dict(status='DEVELOPMENT_ONLY',fits=48,results=sorted(table,key=lambda x:x['mean_bpb']),factorial_main_effects_by_seed=effects,warning='Shared recipe; reused dev; no promotion or equal FLOP claim.'),indent=2));print(a.out.read_text())
if __name__=='__main__':main()
