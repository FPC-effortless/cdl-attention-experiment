"""Read-only external evidence audit; no fitting, adaptation, or new forecast policies."""
import argparse,collections,hashlib,json,math,pathlib,statistics
import numpy as np
NAMES=('cog','protected-instant','dense-instant','protected-compact','static','gru','persistence','lag-ridge','kalman')
SEEDS=(171,311,591); REGIMES=('0.15-0','0.15-1','0.5-0','0.5-1'); H=(0,5,15)
FIELDS=('rmse_v','mae_v','coverage90','width_v')
def sha(b):return hashlib.sha256(b).hexdigest()
def filehash(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def avg(x):return statistics.mean(x)
def verify(root):
 rows=(root/'SHA256SUMS.txt').read_text().splitlines();seen=set()
 for row in rows:
  digest,name=row.split('  ',1);p=root/name
  assert p.resolve().is_relative_to(root.resolve()) and p.is_file() and name not in seen
  assert filehash(p)==digest,name;seen.add(name)
 assert {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name!='SHA256SUMS.txt'}=={str(pathlib.Path(n)) for n in seen}
 return len(rows)
def verify_arrays(z,records):
 keys={x['key']:x for x in records};assert set(keys)==set(z.files) and len(keys)==len(records)
 for k,r in keys.items():
  x=z[k];assert x.dtype==np.float32 and x.shape==(r['seconds'],2) and np.isfinite(x).all() and (x[:,0]>0).all()
  assert sha(x.astype('<f4').tobytes())==r['array_sha256'],k
 assert len({x['array_sha256'] for x in records})==len(records)
 return keys
def combine(rows):
 n=sum(x['n'] for x in rows);assert n>0
 return {'n':n,'rmse_v':math.sqrt(sum(x['sse'] for x in rows)/n),'mae_v':sum(x['sae'] for x in rows)/n,'coverage90':sum(x['covered'] for x in rows)/n,'width_v':sum(x['width'] for x in rows)/n}
def profile(x):
 return {'n':len(x),'mean':float(x.mean(dtype=np.float64)),'std':float(x.std(dtype=np.float64)),'min':float(x.min()),'p01':float(np.quantile(x,.01)),'median':float(np.median(x)),'p99':float(np.quantile(x,.99)),'max':float(x.max()),'negative_fraction':float((x<0).mean()),'zero_fraction':float((x==0).mean())}
def main():
 p=argparse.ArgumentParser()
 for n in ('archive','reconstructed','source','source-manifest','out'):p.add_argument('--'+n,required=True)
 a=p.parse_args();root=pathlib.Path(a.archive);fresh=pathlib.Path(a.reconstructed);out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
 files=verify(root);m=read(root/'cmu/manifest.json');m2=read(fresh/'manifest.json')
 assert m['source']['doi']=='10.1184/R1/12683453.v1' and m['source']['license']['name']=='CC BY 4.0'
 assert m['source']['downloads']==m2['source']['downloads'] and m['admitted']==m2['admitted'] and m['rejected']==m2['rejected']
 z=np.load(root/'cmu/battery-only.npz',allow_pickle=False);zz=np.load(fresh/'battery-only.npz',allow_pickle=False)
 records=verify_arrays(z,m['admitted']);verify_arrays(zz,m2['admitted'])
 for k in z.files:assert np.array_equal(z[k],zz[k]),k
 assert len(records)==203 and sum(x['seconds'] for x in records.values())==37507 and len(m['rejected'])==6
 raw=pathlib.Path(a.source_manifest).read_bytes();s=pathlib.Path(a.source);assert raw==s.with_suffix('.json').read_bytes()
 sm=json.loads(raw);sz=np.load(s,allow_pickle=False);source=verify_arrays(sz,sm['source']);assert len(source)==215
 donor=read(root/'donor-replay.json');assert donor['replayed_panels']==540 and donor['resume_exact']==45
 assert (root/'frozen-donor.sha256').read_text().split()[0]==filehash(root/'donor-replay.json')
 d={(r['fold'],r['model'],r['seed']):r for r in donor['rows']};assert len(d)==63
 target=np.concatenate([z[k] for k in sorted(z.files)])
 profiles={'external':{n:profile(target[:,i]) for i,n in enumerate(('voltage_v','current_a'))},'training_folds':{}}
 for g in 'YRG':
  tr=np.concatenate([sz[k] for k in sm['splits'][g]['train']])
  mu=tr.mean(0);sd=tr.std(0)
  for r in donor['rows']:
   if r['fold']==g:assert np.array_equal(mu,np.asarray(r['mean'],np.float32)) and np.array_equal(sd,np.asarray(r['std'],np.float32))
  profiles['training_folds'][g]={}
  for i,n in enumerate(('voltage_v','current_a')):
   v=target[:,i];x=tr[:,i];scale=(v-mu[i])/sd[i]
   profiles['training_folds'][g][n]={'train':profile(x),'external_mean_shift_in_train_sd':float((v.mean(dtype=np.float64)-mu[i])/sd[i]),'external_fraction_outside_train_minmax':float(((v<x.min())|(v>x.max())).mean()),'external_fraction_abs_z_gt3':float((np.abs(scale)>3).mean())}
 rows=read(root/'external-fits.json');original=read(root/'external-summary.json')
 expected={(g,n,s) for g in 'YRG' for n in NAMES for s in (SEEDS if n in NAMES[:6] else (None,))}
 assert len(rows)==len(expected)==63 and {(r['fold'],r['model'],r['seed']) for r in rows}==expected
 target_counts={};max_delta=0.;byday=collections.defaultdict(list)
 for k,r in records.items():byday[r['date']].append(k)
 days=sorted(byday);assert len(days)==18 and '' not in days
 all_cells={(reg,h) for reg in REGIMES for h in H};daily={n:[] for n in NAMES}
 for r in rows:
  ident=(r['fold'],r['model'],r['seed']);assert r['donor_q']==d[ident]['q']
  f={(x['flight'],x['regime']):x for x in r['flight_metrics']}
  assert len(f)==len(r['flight_metrics'])==len(records)*4 and set(f)=={(k,reg) for k in records for reg in REGIMES}
  for reg,h in all_cells:
   cell=[f[k,reg][str(h)] for k in sorted(records)]
   for k,x in zip(sorted(records),cell):
    key=(k,reg,h);n=x['n'];assert isinstance(n,int) and n>=0 and 0<=x['covered']<=n and x['sse']>=0 and x['sae']>=0 and x['width']>=0
    if key in target_counts:assert n==target_counts[key]
    else:target_counts[key]=n
   metrics=combine(cell);old=r['metrics'][f'{reg}/{h}'];assert old['n']==metrics['n']
   for field in FIELDS:
    delta=abs(old[field]-metrics[field]);assert math.isfinite(delta) and delta<2e-5,(ident,reg,h,field,delta);max_delta=max(max_delta,delta)
  v=np.array([[sum(f[k,'0.15-1']['15']['sse'] for k in byday[day]),sum(f[k,'0.15-1']['15']['n'] for k in byday[day])] for day in days],np.float64)
  daily[r['model']].append(v)
 assert len(target_counts)==2436
 summary={}
 for n in NAMES:
  rr=[x for x in rows if x['model']==n];summary[n]={}
  for reg,h in sorted(all_cells):
   key=f'{reg}/{h}';folds={g:{f:avg(x['metrics'][key][f] for x in rr if x['fold']==g) for f in FIELDS} for g in 'YRG'}
   val={f:avg(x[f] for x in folds.values()) for f in FIELDS}
   for f in FIELDS:assert abs(val[f]-original['summary'][n][key]['equal_fold'][f])<2e-9
   summary[n][key]={'folds':folds,'equal_fold':val}
 pairs=[('protected-compact',n) for n in NAMES if n!='protected-compact']+[('protected-instant','dense-instant'),('cog','persistence')]
 matrix={n:np.array(v) for n,v in daily.items()};rng=np.random.default_rng(21947)
 def score(n,idx):
  v=matrix[n][:,idx,:].sum(1);assert (v[:,1]>0).all()
  return float(np.sqrt(v[:,0]/v[:,1]).mean())
 samples={pair:[] for pair in pairs}
 for b in range(10000):
  idx=rng.integers(0,len(days),len(days));v={n:score(n,idx) for n in NAMES}
  for pair in pairs:samples[pair].append(v[pair[0]]-v[pair[1]])
 comparisons={}
 for pair,x in samples.items():
  point=summary[pair[0]]['0.15-1/15']['equal_fold']['rmse_v']-summary[pair[1]]['0.15-1/15']['equal_fold']['rmse_v']
  ci=np.quantile(x,[.025,.975]).tolist()
  if pair[0]=='protected-compact':
   prev=original['conditional_date_bootstrap'][pair[1]];assert abs(point-prev['compact_minus_baseline_v'])<2e-9 and np.max(np.abs(np.asarray(ci)-prev['conditional95_v']))<2e-9
  comparisons[' vs '.join(pair)]={'delta_v':point,'conditional95_v':ci,'scope':'post-result, fixed cohort/models, grouped by recording date'}
 winners={}
 for reg,h in sorted(all_cells):
  key=f'{reg}/{h}';winners[key]=min(NAMES,key=lambda n:summary[n][key]['equal_fold']['rmse_v'])
 result={'scope':'Read-only verification of archived external battery-monitoring results and descriptive donor/target distributions; no new fits, calibration, adaptation, or claim of physical causality.','archive_files_verified':files,'independently_reconstructed_external_arrays':len(records),'independently_reconstructed_source_arrays':len(source),'frozen_evaluations':len(rows),'matched_cells':len(target_counts),'max_flight_aggregation_discrepancy':max_delta,'source_replay_panels_recorded':540,'source_resume_checks_recorded':45,'profiles':profiles,'winners':winners,'conditional_comparisons':comparisons,'qualification':'HOLD'}
 write(out/'summary.json',result);write(out/'all-metrics.json',summary)
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
