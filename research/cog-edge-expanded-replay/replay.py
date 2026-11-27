"""Independent replay and mechanism diagnostics for archived passive battery forecasts."""
import argparse, copy, hashlib, json, math, pathlib, random, sys
import numpy as np
import torch

def sha(b): return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(p.read_text())
def avg(a):return sum(a)/len(a)
NAMES=('cog','dense','gated','gru','static','persistence','lag-ridge','kalman')
FIELDS=('rmse_v','mae_v','coverage90','width_v')
REGIMES=('0.15-0','0.15-1','0.5-0','0.5-1')
H=(0,5,15)

def aggregate(rows,reg,h):
 r=[x[str(h)] for x in rows if x['regime']==reg]
 n=sum(x['n'] for x in r)
 assert n>0
 return {'n':n,'rmse_v':math.sqrt(sum(x['sse'] for x in r)/n),'mae_v':sum(x['sae'] for x in r)/n,'coverage90':sum(x['covered'] for x in r)/n,'width_v':sum(x['width'] for x in r)/n}

def compare_metrics(a,b,tol=2e-5):
 assert set(a)==set(b)
 delta=0.
 for k in a:
  assert a[k]['n']==b[k]['n'],k
  for f in FIELDS:
   d=abs(a[k][f]-b[k][f]);assert math.isfinite(d) and d<tol,(k,f,d)
   delta=max(delta,d)
 return delta

def audit_partition(meta,arrays,fold,r):
 s=meta['splits'][fold];assert r['splits']==s
 source={a['key']:a for a in meta['source']}
 groups=[set(s[k]) for k in ('train','dev','test')]
 packs=[{source[k]['pack'] for k in group} for group in groups]
 for i in range(3):
  assert len(groups[i])==len(s[('train','dev','test')[i]]) and all(x.isdigit() for x in packs[i])
  for j in range(i):assert not groups[i]&groups[j] and not packs[i]&packs[j]
 assert {source[k]['vehicle'] for k in groups[2]}=={fold}
 assert all(source[k]['vehicle']!=fold for k in groups[0]|groups[1])
 return {k:len(s[k]) for k in s}|{'test_pack_labels':len(packs[2])}

def find_fits(r,kalman,n):
 if n=='kalman':return [kalman]
 if n in r['baselines']:return [r['baselines'][n]]
 return sorted([f for f in r['fits'] if f['name']==n],key=lambda x:x['seed'])

def bootstrap(folds,source):
 seed=90421
 def rand():
  nonlocal seed
  seed=(1664525*seed+1013904223)&0xffffffff
  return seed/4294967296
 grouped=[]
 for r,kalman in folds:
  names=r['splits']['test'];groups={}
  for f in names:groups.setdefault(source[f]['pack'],[]).append(f)
  groups=[groups[k] for k in sorted(groups,key=int)]
  byname={}
  for n in NAMES:
   fits=find_fits(r,kalman,n)
   byname[n]=[]
   for fit in fits:
    rows={x['flight']:x['15'] for x in fit['flight_metrics'] if x['regime']=='0.15-1'}
    assert set(rows)==set(names)
    byname[n].append(np.array([[sum(rows[f]['n'] for f in g),sum(rows[f]['sse'] for f in g)] for g in groups],dtype=np.float64))
  grouped.append(byname)
 def compute(a,ids):
  return avg([math.sqrt(float(x[ids,1].sum()/x[ids,0].sum())) for x in a])
 deltas={n:[] for n in NAMES[1:]}
 for b in range(20000):
  byfold={n:[] for n in NAMES[1:]}
  for group in grouped:
   ng=len(group['cog'][0]);ids=[int(rand()*ng) for _ in range(ng)]
   cog=compute(group['cog'],ids)
   for n in NAMES[1:]:byfold[n].append(cog-compute(group[n],ids))
  for n in NAMES[1:]:deltas[n].append(avg(byfold[n]))
 result={}
 for n in NAMES[1:]:
  vals=sorted(deltas[n])
  result[n]={'delta_v':avg([avg([f['test']['0.15-1/15']['rmse_v'] for f in find_fits(r,k,'cog')])-avg([f['test']['0.15-1/15']['rmse_v'] for f in find_fits(r,k,n)]) for r,k in folds]),'conditional95_v':[vals[499],vals[19499]],'per_fold_seed_deltas':{r['fold']:[f['test']['0.15-1/15']['rmse_v']-z['test']['0.15-1/15']['rmse_v'] for f,z in zip(find_fits(r,k,'cog'),find_fits(r,k,n)* (3 if len(find_fits(r,k,n))==1 else 1))] for r,k in folds}}
 return result

@torch.no_grad()
def reset_predict(m,panels,chunk):
 out=[]
 for name,reg,x,t,s in panels:
  pairs=[m(x[:,i:i+chunk]) for i in range(0,x.shape[1],chunk)]
  u=torch.cat([a[0] for a in pairs],1)[:,:t.shape[1]]
  v=torch.cat([a[1] for a in pairs],1)[:,:t.shape[1]]
  out.append((name,reg,u.numpy(),v.numpy(),t.numpy(),s.numpy()))
 return out

def resume_one(edge,model,c,tr):
 def one():
  model.load_state_dict(c['model']);model.train()
  opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.002);opt.load_state_dict(copy.deepcopy(c['optimizer']))
  torch.set_rng_state(c['torch_rng']);random.setstate(c['python_rng'])
  rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(c['numpy_rng'])
  x,t,s=edge.batch(tr,rng);u,v=model(x);u=u[:,:t.shape[1]];v=v[:,:t.shape[1]]
  loss=(.5*((u-t)/v).square()+v.log())[s].mean()
  opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
  return float(loss.detach()),copy.deepcopy(model.state_dict()),copy.deepcopy(opt.state_dict())
 a,b,o=one();aa,bb,oo=one()
 assert math.isfinite(a) and a==aa and all(torch.equal(b[k],bb[k]) for k in b)
 for k in o['state']:
  for p,v in o['state'][k].items():
   z=oo['state'][k][p];assert torch.equal(v,z) if isinstance(v,torch.Tensor) else v==z

def main():
 p=argparse.ArgumentParser()
 for arg in ('restored','dataset','archived','out'):p.add_argument('--'+arg,required=True)
 a=p.parse_args();sys.path.insert(0,a.restored);import edge
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 root=pathlib.Path(a.archived);path=pathlib.Path(a.dataset)
 assert (root/'source-manifest.json').read_bytes()==path.with_suffix('.json').read_bytes()
 meta=read(root/'source-manifest.json');source={x['key']:x for x in meta['source']}
 data=np.load(path,allow_pickle=False)
 assert len(source)==215 and set(data.files)==set(source)
 assert len({x['array_sha256'] for x in source.values()})==215
 for k in data.files:
  x=data[k];assert len(x)==source[k]['seconds'] and sha(x.astype('<f4').tobytes())==source[k]['array_sha256']
 kalman=read(root/'kalman-results.json')['folds'];folds=[];rows=[];splits={};match=0;recorded_delta=0.
 for g in 'YRG':
  d=root/f'expanded-{g}';r=read(d/'results.json');k=kalman[g];folds.append((r,k));splits[g]=audit_partition(meta,data,g,r)
  assert r['manifest_sha256']==sha((root/'source-manifest.json').read_bytes())
  assert r['updates']==600 and r['window']==64 and r['batch']==16 and len(r['fits'])==15
  tr=np.concatenate([data[n].astype(np.float32) for n in r['splits']['train']]);mean=tr.mean(0);std=tr.std(0)
  assert np.array_equal(mean,np.asarray(r['mean'],np.float32)) and np.array_equal(std,np.asarray(r['std'],np.float32))
  train=[(data[n]-mean)/std for n in r['splits']['train']]
  dev=edge.panels([(data[n]-mean)/std for n in r['splits']['dev']],r['splits']['dev'],'dev-v1')
  test=edge.panels([(data[n]-mean)/std for n in r['splits']['test']],r['splits']['test'],'test-v1')
  expected={(name,reg):{str(h):int(s[...,j].sum()) for j,h in enumerate(H)} for name,reg,x,t,s in test}
  for n in NAMES:
   for f in find_fits(r,k,n):
    raw=f['flight_metrics'];assert len(raw)==len(expected)
    assert len({(x['flight'],x['regime']) for x in raw})==len(expected)
    for x in raw:
     for h in H:assert x[str(h)]['n']==expected[x['flight'],x['regime']][str(h)]
    metrics={f'{reg}/{h}':aggregate(raw,reg,h) for reg in REGIMES for h in H}
    recorded_delta=max(recorded_delta,compare_metrics(metrics,f['test']))
  match+=12
  for f in r['fits']:
   n=f['name'];ck=d/f"{f['seed']}-{n}";m=edge.make(n)
   init=torch.load(ck/'initial.pt',weights_only=True)
   assert set(init)==set(m.state_dict()) and all(init[p].shape==v.shape for p,v in m.state_dict().items())
   c=torch.load(ck/'selected.pt',weights_only=True);final=torch.load(ck/'final.pt',weights_only=True)
   for snap,u in ((c,f['selected']),(final,600)):
    assert snap['name']==n and snap['seed']==f['seed'] and snap['update']==u
    assert set(('optimizer','torch_rng','numpy_rng','python_rng','model'))<=set(snap)
   resume_one(edge,m,final,train);m.load_state_dict(c['model']);m.eval()
   q=edge.calibrate(edge.predict(m,dev));qd=max(abs(x-y) for x,y in zip(q,f['q']));assert qd<2e-5
   full=edge.predict(m,test);replayed,flight=edge.metrics(full,std[0],f['q'])
   md=compare_metrics(replayed,f['test'])
   prefix=0.
   with torch.no_grad():
    for _,_,x,*_ in test:
     cut=x.shape[1]//2;expected_prefix=torch.cat(m(x),-1)[:,:cut]
     perturbed=x.clone();perturbed[:,cut:]=0
     prefix=max(prefix,float((torch.cat(m(perturbed),-1)[:,:cut]-expected_prefix).abs().max()),float((torch.cat(m(x[:,:cut]),-1)-expected_prefix).abs().max()))
   assert prefix<2e-5
   reset={}
   for chunk in (16,64):
    summary,_=edge.metrics(reset_predict(m,test,chunk),std[0],f['q'])
    if n=='static':assert compare_metrics(summary,replayed)<2e-5
    reset[str(chunk)]=summary
   rows.append({'fold':g,'seed':f['seed'],'model':n,'metric_delta':md,'calibration_delta':qd,'prefix_delta':prefix,'resume_exact':True,'checkpoint_sha256':{p.name:sha(p.read_bytes()) for p in ck.glob('*.pt')},'full':replayed,'reset':reset})
   print(g,n,f['seed'],md,flush=True)
  print('fold',g,'passed',flush=True)
 bymodel={}
 for n in ('cog','dense','gated','gru','static'):
  rr=[r for r in rows if r['model']==n];bymodel[n]={}
  for key in [f'{reg}/{h}' for reg in REGIMES for h in H]:
   bymodel[n][key]={c:avg([avg([r['full'][key]['rmse_v'] if c=='full' else r['reset'][c][key]['rmse_v'] for r in rr if r['fold']==g]) for g in 'YRG']) for c in ('full','16','64')}
 result={'scope':'Expanded archive independent checkpoint replay; no new fit or fresh held-out cohort. Reset diagnostic is post-result and preserves preprocessor history. Conditional bootstrap does not generalize to unobserved vehicles.','admitted_logs':len(source),'seconds':sum(x['seconds'] for x in source.values()),'splits':splits,'matches':match,'replay_panels':len(rows)*12,'checkpoints':len(rows)*3,'resume_exact':sum(x['resume_exact'] for x in rows),'max_recorded_aggregation_delta':recorded_delta,'max_metric_delta':max(x['metric_delta'] for x in rows),'max_calibration_delta':max(x['calibration_delta'] for x in rows),'max_prefix_delta':max(x['prefix_delta'] for x in rows),'conditional_bootstrap':bootstrap(folds,source),'history_reset_summary':bymodel,'fit_replay':rows,'qualification':'HOLD; observed cohorts and conditional analysis only; no physical embedded or fault-label evidence.'}
 pathlib.Path(a.out).write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k not in ('fit_replay','history_reset_summary')},indent=2))
if __name__=='__main__':main()
