"""Independent v0.3 replay and frozen-weight external battery-telemetry evaluation."""
import argparse,copy,hashlib,json,math,pathlib,random,statistics,sys
import numpy as np
import torch
NAMES=('cog','protected-instant','dense-instant','protected-compact','static','gru','persistence','lag-ridge','kalman')
SEEDS=(171,311,591)
H=(0,5,15)
FIELDS=('rmse_v','mae_v','coverage90','width_v')
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,obj):p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def avg(a):return statistics.mean(a)
def equal(a,b,tol=2e-5):
 assert set(a)==set(b)
 delta=0.
 for k in a:
  assert a[k]['n']==b[k]['n']
  for f in FIELDS:
   d=abs(a[k][f]-b[k][f]);assert math.isfinite(d) and d<tol,(k,f,d);delta=max(delta,d)
 return delta
def verify(path):
 lines=(path/'SHA256SUMS.txt').read_text().splitlines();names=set()
 for l in lines:
  digest,name=l.split('  ',1);p=path/name
  assert p.is_file() and p.resolve().is_relative_to(path.resolve()) and name not in names and sha(p.read_bytes())==digest,name;names.add(name)
 assert {str(p.relative_to(path)) for p in path.rglob('*') if p.is_file() and p.name!='SHA256SUMS.txt'}=={str(pathlib.Path(n)) for n in names}
 return len(names)
def load_model(edge,protected,name,path):
 m=edge.make('gru') if name=='gru' else protected.make(name)
 c=torch.load(path,weights_only=True);m.load_state_dict(c['model']);m.eval();return m,c
def resume(edge,m,c,tr):
 def update():
  m.load_state_dict(c['model']);m.train();o=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=.002);o.load_state_dict(copy.deepcopy(c['optimizer']))
  torch.set_rng_state(c['torch_rng']);random.setstate(c['python_rng']);rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(c['numpy_rng'])
  x,t,s=edge.batch(tr,rng);u,v=m(x);u=u[:,:t.shape[1]];v=v[:,:t.shape[1]]
  loss=(.5*((u-t)/v).square()+v.log())[s].mean();o.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);o.step()
  return float(loss.detach()),copy.deepcopy(m.state_dict()),copy.deepcopy(o.state_dict())
 a,x,o=update();b,y,p=update();assert math.isfinite(a) and a==b and all(torch.equal(v,y[k]) for k,v in x.items())
 for k,s in o['state'].items():
  assert all(torch.equal(v,p['state'][k][n]) if isinstance(v,torch.Tensor) else v==p['state'][k][n] for n,v in s.items())
def counts(rows):
 return {(r['flight'],r['regime'],h):r[str(h)]['n'] for r in rows for h in H}
def replay(args,edge,protected,kalman):
 root=pathlib.Path(args.archives);old=pathlib.Path(args.old);out=pathlib.Path(args.out);out.mkdir(parents=True,exist_ok=True)
 raw=(old/'source-manifest.json').read_bytes();src=pathlib.Path(args.source);assert src.with_suffix('.json').read_bytes()==raw
 meta=json.loads(raw);source={x['key']:x for x in meta['source']};z=np.load(src,allow_pickle=False)
 assert set(z.files)==set(source) and len(source)==215
 for k in z.files:assert len(z[k])==source[k]['seconds'] and sha(z[k].astype('<f4').tobytes())==source[k]['array_sha256']
 kal=read(old/'kalman-results.json')['folds'];rows=[];receipts=[]
 for fold in 'YRG':
  path=root/fold;files=verify(path);d=path/'fits';r=read(d/'results.json');prev=read(old/f'expanded-{fold}'/'results.json')
  assert r['complete'] and r['fold']==fold and r['source_manifest_sha256']==sha(raw) and r['updates']==600 and len(r['fits'])==15
  assert {(f['name'],f['seed']) for f in r['fits']}=={(n,s) for n in NAMES[:5] for s in SEEDS}
  split=r['splits'];assert split['train']==meta['splits'][fold]['train'] and split['test']==meta['splits'][fold]['test']
  assert set(split['selection'])|set(split['calibration'])==set(meta['splits'][fold]['dev'])
  packs=[{source[k]['pack'] for k in v} for v in split.values()];assert all(not packs[i]&packs[j] for i in range(4) for j in range(i))
  for s in SEEDS:assert len({f['training_stream_sha256'] for f in r['fits'] if f['seed']==s})==1
  tr_raw=np.concatenate([z[k].astype(np.float32) for k in split['train']]);mu=tr_raw.mean(0);sd=tr_raw.std(0)
  assert np.array_equal(mu,np.asarray(r['mean'],np.float32)) and np.array_equal(sd,np.asarray(r['std'],np.float32))
  tr=[(z[n]-mu)/sd for n in split['train']];names=split['calibration'];dev=edge.panels([(z[n]-mu)/sd for n in names],names,'cal-v2')
  n_expected=None
  for f in r['fits']:
   n=f['name'];seed=f['seed'];ck=d/f'{seed}-{n}';assert set(f['checkpoint_sha256'])=={'initial.pt','selected.pt','final.pt'}
   for name,digest in f['checkpoint_sha256'].items():assert sha((ck/name).read_bytes())==digest
   m,c=load_model(edge,protected,n,ck/'selected.pt');initial=torch.load(ck/'initial.pt',weights_only=True);final=torch.load(ck/'final.pt',weights_only=True)
   assert set(initial)==set(c['model']) and all(initial[k].shape==c['model'][k].shape for k in initial)
   assert c['update']==f['selected'] and final['update']==600 and c['name']==n and c['seed']==seed and c['mean']==r['mean'] and c['std']==r['std']
   resume(edge,m,final,tr);m.load_state_dict(c['model']);m.eval()
   pp=edge.predict(m,dev);metrics,fr=edge.metrics(pp,sd[0]);delta=equal(metrics,f['development_metrics'])
   assert counts(fr)==counts(f['flight_metrics'])
   if n_expected is None:n_expected=counts(fr)
   assert counts(fr)==n_expected
   rows.append({'fold':fold,'model':n,'seed':seed,'q':edge.calibrate(pp),'mean':r['mean'],'std':r['std'],'checkpoint_sha256':f['checkpoint_sha256']['selected.pt'],'source_replay_delta':delta,'resume_exact':True})
   print('replay',fold,n,seed,delta,flush=True)
  # Frozen historical controls were selected using the larger development pool. The
  # same donor calibration subset is not independent for those historical models.
  for f in prev['fits']:
   if f['name']!='gru':continue
   ck=old/f'expanded-{fold}'/f"{f['seed']}-gru"/'selected.pt';m,c=load_model(edge,protected,'gru',ck)
   assert c['mean']==r['mean'] and c['std']==r['std'] and c['seed']==f['seed']
   rows.append({'fold':fold,'model':'gru','seed':f['seed'],'q':edge.calibrate(edge.predict(m,dev)),'mean':r['mean'],'std':r['std'],'checkpoint_sha256':sha(ck.read_bytes()),'historical_selection_overlap':True})
  for n in ('persistence','lag-ridge','kalman'):
   if n=='kalman':
    k=kal[fold]['selected'];cfg=[np.asarray(k[x]) for x in ('A','b','Q')]+[k['R']];pp=kalman.predict(dev,*cfg)
   else:pp=edge.classic(dev,None if n=='persistence' else np.asarray(prev['baselines'][n]['coefficient']))
   rows.append({'fold':fold,'model':n,'seed':None,'q':edge.calibrate(pp),'mean':r['mean'],'std':r['std'],'historical_selection_overlap':True})
  receipts.append({'fold':fold,'manifest_files':files,'checkpoints':45,'flight_regime_horizon_counts':len(n_expected)})
 assert len(rows)==63 and sum('source_replay_delta' in r for r in rows)==45
 result={'scope':'Independent replay of 45 v0.3 selected models and exact final optimizer continuation. All donor calibration factors frozen before external labels; historical controls used a larger selection cohort. No new training.','receipts':receipts,'replayed_panels':45*12,'max_metric_delta':max(r['source_replay_delta'] for r in rows if 'source_replay_delta' in r),'resume_exact':sum(r.get('resume_exact',False) for r in rows),'rows':rows}
 write(out/'donor-replay.json',result)
 print('source replay passed',result['replayed_panels'],'panels, 45 continuations',flush=True)

def external(args,edge,protected,kalman):
 root=pathlib.Path(args.archives);old=pathlib.Path(args.old);out=pathlib.Path(args.out)
 donor=read(out/'donor-replay.json');d=pathlib.Path(args.external);meta=read(d/'manifest.json');z=np.load(d/'battery-only.npz',allow_pickle=False);source={r['key']:r for r in meta['admitted']};names=sorted(source)
 assert set(z.files)==set(names) and len(names)>=100
 for n in names:assert sha(z[n].astype('<f4').tobytes())==source[n]['array_sha256']
 kal=read(old/'kalman-results.json')['folds'];rows=[];expected=None
 for fold in 'YRG':
  prev=read(old/f'expanded-{fold}'/'results.json');r0=read(root/fold/'fits'/'results.json');mu=np.asarray(r0['mean'],np.float32);sd=np.asarray(r0['std'],np.float32)
  ps=edge.panels([(z[n]-mu)/sd for n in names],names,'external-cmu-v1')
  for r in [v for v in donor['rows'] if v['fold']==fold]:
   n=r['model'];seed=r['seed'];assert r['mean']==r0['mean'] and r['std']==r0['std']
   if n in NAMES[:6]:
    ck=(old/f'expanded-{fold}' if n=='gru' else root/fold/'fits')/f'{seed}-{n}'/'selected.pt';assert sha(ck.read_bytes())==r['checkpoint_sha256']
    m,c=load_model(edge,protected,n,ck);pp=edge.predict(m,ps)
   elif n=='kalman':
    k=kal[fold]['selected'];pp=kalman.predict(ps,*[np.asarray(k[x]) for x in ('A','b','Q')],k['R'])
   else:pp=edge.classic(ps,None if n=='persistence' else np.asarray(prev['baselines'][n]['coefficient']))
   metrics,fr=edge.metrics(pp,sd[0],r['q']);cnt=counts(fr)
   if expected is None:expected=cnt
   assert cnt==expected
   assert all(math.isfinite(v[f]) for v in metrics.values() for f in FIELDS)
   rows.append({'fold':fold,'model':n,'seed':seed,'metrics':metrics,'flight_metrics':fr,'donor_q':r['q']})
   print('external',fold,n,seed,metrics['0.15-1/15']['rmse_v'],flush=True)
   write(out/'external-fits.json',rows)
 assert len(rows)==63
 summary={}
 for n in NAMES:
  summary[n]={}
  rr=[r for r in rows if r['model']==n]
  for key in rr[0]['metrics']:
   folds={g:{f:avg([r['metrics'][key][f] for r in rr if r['fold']==g]) for f in FIELDS} for g in 'YRG'}
   summary[n][key]={'folds':folds,'equal_fold':{f:avg([v[f] for v in folds.values()]) for f in FIELDS}}
 # Conditional paired bootstrap by recorded collection date; not airframe-level inference.
 groups={}
 for n in names:groups.setdefault(source[n]['date'],[]).append(n)
 ci={}
 if len(groups)>=5 and '' not in groups:
  days=sorted(groups);rng=np.random.default_rng(21947)
  byname={}
  for n in NAMES:
   ff=[]
   for r in rows:
    if r['model']!=n:continue
    rr={x['flight']:x['15'] for x in r['flight_metrics'] if x['regime']=='0.15-1'}
    ff.append([[sum(rr[f]['sse'] for f in groups[day]),sum(rr[f]['n'] for f in groups[day])] for day in days])
   byname[n]=np.asarray(ff,dtype=np.float64)
  def compute(n,idx):
   v=byname[n][:,idx,:].sum(1)
   return float(np.sqrt(v[:,0]/v[:,1]).mean())
  deltas={n:[] for n in NAMES if n!='protected-compact'}
  for b in range(10000):
   idx=rng.integers(0,len(days),len(days));c=compute('protected-compact',idx)
   for n in deltas:deltas[n].append(c-compute(n,idx))
  ci={n:{'compact_minus_baseline_v':summary['protected-compact']['0.15-1/15']['equal_fold']['rmse_v']-summary[n]['0.15-1/15']['equal_fold']['rmse_v'],'conditional95_v':np.quantile(v,[.025,.975]).tolist()} for n,v in deltas.items()}
 result={'scope':'Zero-shot cross-dataset evaluation of frozen AMOVFLY models on CMU Matrice 100 battery-only telemetry. All target data excluded from model fitting, selection, normalization and interval calibration. One new platform, unknown physical pack identity. Simulated missingness, not actual faults. Date bootstrap conditional on dataset and fitted models, not independent aircraft.','flights':len(names),'seconds':sum(len(z[k]) for k in names),'date_groups':len(groups),'evaluations':len(rows),'matched_flight_regime_horizon_counts':len(expected),'summary':summary,'conditional_date_bootstrap':ci,'qualification':'HOLD: no fleet population or physical-device guarantee'}
 write(out/'external-summary.json',result);print(json.dumps({'flights':result['flights'],'seconds':result['seconds'],'primary':{n:summary[n]['0.15-1/15']['equal_fold'] for n in NAMES},'date_bootstrap':ci},indent=2))
def main():
 p=argparse.ArgumentParser()
 for n in ('restored','archives','old','out'):p.add_argument('--'+n,required=True)
 p.add_argument('--source');p.add_argument('--external');p.add_argument('--phase',choices=['replay','external'],required=True);a=p.parse_args()
 sys.path.insert(0,a.restored);sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'cog-edge-protected-v03'));import edge,protected,kalman
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 (replay if a.phase=='replay' else external)(a,edge,protected,kalman)
if __name__=='__main__':main()
