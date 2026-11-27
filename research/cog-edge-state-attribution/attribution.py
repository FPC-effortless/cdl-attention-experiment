"""Development-only attribution of protected and predictive recurrent state."""
import argparse, hashlib, json, pathlib, math, sys
import numpy as np
import torch

H=(0,5,15)
REGIMES=('0.15-0','0.15-1','0.5-0','0.5-1')
ARMS=('cog','dense','gated')
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def mean(x):return sum(x)/len(x)

@torch.no_grad()
def intervene(edge,m,x,period=0,phase=0,which='full'):
 a,b,c,d=m.coeff(x)
 if period:
  t=torch.arange(x.shape[1]); reset=(t>=phase)&((t-phase)%period==0)
  keep=(~reset).to(x.dtype).view(1,-1,1)
  if which in ('protected','both'):a=a*keep.unsqueeze(-1)
  if which in ('predictive','both'):c=c*keep
 h=edge.scan(a,b);p=edge.scan(c,d)
 return m.emit(h,p,x),(h,p)

@torch.no_grad()
def verify(edge,m,x,period,phase,which):
 out,states=intervene(edge,m,x,period,phase,which)
 full,reference=intervene(edge,m,x)
 if which=='protected':assert torch.equal(states[1],reference[1])
 if which=='predictive':assert torch.equal(states[0],reference[0])
 s=m.init(len(x));pred=[];states_step=[[],[]]
 for t in range(x.shape[1]):
  if t>=phase and (t-phase)%period==0:
   s=tuple(torch.zeros_like(v) if which=='both' or which==('protected','predictive')[i] else v for i,v in enumerate(s))
  o,s=m.step(x[:,t],s)
  pred.append(torch.cat(o,-1))
  for i in range(2):states_step[i].append(s[i])
 delta=max(float((torch.stack(pred,1)-torch.cat(out,-1)).abs().max()),*[float((torch.stack(states_step[i],1)-states[i]).abs().max()) for i in range(2)])
 assert delta<2e-5
 if which=='both' and phase==0:
  pieces=[torch.cat(m(x[:,i:i+period]),-1) for i in range(0,x.shape[1],period)]
  assert float((torch.cat(pieces,1)-torch.cat(out,-1)).abs().max())<2e-5
 return delta

@torch.no_grad()
def evaluate(edge,m,panels,std,q,period=0,phase=0,which='full'):
 predictions=[]
 for name,reg,x,t,s in panels:
  (u,v),_=intervene(edge,m,x,period,phase,which)
  predictions.append((name,reg,u[:,:t.shape[1]].numpy(),v[:,:t.shape[1]].numpy(),t.numpy(),s.numpy()))
 return edge.metrics(predictions,std,q)

def main():
 ap=argparse.ArgumentParser()
 for n in ('restored','dataset','archived','out'):ap.add_argument('--'+n,required=True)
 a=ap.parse_args();sys.path.insert(0,a.restored);import edge
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 root=pathlib.Path(a.archived);path=pathlib.Path(a.dataset);out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
 raw=(root/'source-manifest.json').read_bytes();assert raw==path.with_suffix('.json').read_bytes()
 meta=json.loads(raw);source={x['key']:x for x in meta['source']};data=np.load(path,allow_pickle=False)
 assert len(source)==215 and set(data.files)==set(source)
 for k in data.files:assert len(data[k])==source[k]['seconds'] and sha(data[k].astype('<f4').tobytes())==source[k]['array_sha256']
 protocol=[(p,offset,mode) for p in (16,64) for offset in (0,p//2) for mode in ('protected','predictive','both')]
 rows=[];checks=[];partition={}
 for g in 'YRG':
  d=root/f'expanded-{g}';r=read(d/'results.json');split=meta['splits'][g];assert split==r['splits']
  train=np.concatenate([data[k].astype(np.float32) for k in split['train']]);mu=train.mean(0);std=train.std(0)
  assert np.array_equal(mu,np.asarray(r['mean'],np.float32)) and np.array_equal(std,np.asarray(r['std'],np.float32))
  assert not set(split['dev'])&set(split['test'])
  packs=[{source[k]['pack'] for k in split[part]} for part in ('train','dev','test')]
  assert all(not packs[i]&packs[j] for i in range(3) for j in range(i))
  dev=edge.panels([(data[n]-mu)/std for n in split['dev']],split['dev'],'dev-v1')
  partition[g]={'flights':len(split['dev']),'pack_labels':len(packs[1]),'panels':len(dev)}
  for f in r['fits']:
   if f['name'] not in ARMS:continue
   name=f['name'];seed=f['seed'];ck=d/f'{seed}-{name}'/'selected.pt';snap=torch.load(ck,weights_only=True)
   assert snap['name']==name and snap['seed']==seed and snap['update']==f['selected']
   m=edge.make(name);m.load_state_dict(snap['model']);m.eval()
   original=edge.predict(m,dev);q=edge.calibrate(original);assert max(abs(x-y) for x,y in zip(q,f['q']))<2e-5
   assert abs(edge.val(original,std[0])-f['val'])<2e-5
   full,flight=evaluate(edge,m,dev,std[0],q)
   expected,expected_flight=edge.metrics(original,std[0],q)
   for key in full:
    assert full[key]['n']==expected[key]['n']
    assert max(abs(full[key][k]-expected[key][k]) for k in ('rmse_v','mae_v','coverage90','width_v'))<2e-5
   baseline_counts={(x['flight'],x['regime'],h):x[str(h)]['n'] for x in flight for h in H}
   receipt={'fold':g,'seed':seed,'model':name,'selected_sha256':sha(ck.read_bytes()),'full':full,'flight_full':flight,'interventions':{}}
   for period,phase,mode in protocol:
    key=f'{period}/{phase}/{mode}'
    # Independent scalar recurrence check on one deterministic dev-panel prefix.
    x=dev[0][2][:,:160]
    delta=verify(edge,m,x,period,phase,mode);checks.append(delta)
    metrics,rr=evaluate(edge,m,dev,std[0],q,period,phase,mode)
    assert {(z['flight'],z['regime'],h):z[str(h)]['n'] for z in rr for h in H}==baseline_counts
    receipt['interventions'][key]={'metrics':metrics,'flight_metrics':rr,'step_delta':delta}
   write(out/f'{g}-{seed}-{name}.json',receipt)
   rows.append({k:v for k,v in receipt.items() if k not in ('flight_full','interventions')}|{'interventions':{k:v['metrics'] for k,v in receipt['interventions'].items()}})
   print(g,seed,name,'passed',flush=True)
 assert len(rows)==27 and len(checks)==324
 summary={}
 for m in ARMS:
  summary[m]={}
  for key in [f'{reg}/{h}' for reg in REGIMES for h in H]:
   rr=[r for r in rows if r['model']==m]
   full=mean([mean([r['full'][key]['rmse_v'] for r in rr if r['fold']==g]) for g in 'YRG'])
   impacts={}
   for p in (16,64):
    for mode in ('protected','predictive','both'):
     val=mean([mean([r['interventions'][f'{p}/{phase}/{mode}'][key]['rmse_v'] for r in rr if r['fold']==g for phase in (0,p//2)]) for g in 'YRG'])
     impacts[f'{p}/{mode}']={'rmse_v':val,'delta_v':val-full,'relative_change':val/full-1}
   summary[m][key]={'full_rmse_v':full,'phase_averaged':impacts}
 result={'scope':'Historical development data only; frozen-weight post-result intervention; no new training or test-set use. Reset dependence is not architecture superiority.','source_manifest_sha256':sha(raw),'partition':partition,'models':len(rows),'intervention_configurations':len(checks),'max_step_delta':max(checks),'summary':summary,'fit_results':rows,'qualification':'HOLD'}
 write(out/'summary.json',result)
 print(json.dumps({k:summary[k]['0.15-1/15'] for k in ARMS},indent=2))
if __name__=='__main__':main()
