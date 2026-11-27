"""Offline battery-telemetry forecasting: separate model selection and calibration."""
import argparse, copy, hashlib, json, math, pathlib, random, sys
import numpy as np
import torch
from torch import nn

H=(0,5,15); REGIMES=('0.15-0','0.15-1','0.5-0','0.5-1')
METHODS=('pooled','age-stratified','conservative-age','pack-max')
AGES=('0-3','4-15','16+')
def sha(b):return hashlib.sha256(b).hexdigest()
def write(path,obj):pathlib.Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def q90(x):
 assert len(x) and np.isfinite(x).all()
 return float(np.quantile(x,.9,method='higher'))
def group_age(x):
 age=np.rint(np.expm1(x[...,4].numpy()*4))
 return np.where(age<=3,0,np.where(age<=15,1,2))
def keypack(source,n):return source[n]['pack']

def fit(edge,name,seed,tr,selection,mean,std,updates,path):
 torch.manual_seed(seed);random.seed(seed);rng=np.random.default_rng(seed)
 m=edge.make(name);opt=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=.002);path.mkdir(parents=True,exist_ok=False)
 torch.save(m.state_dict(),path/'initial.pt')
 def snap(i):return {'model':copy.deepcopy(m.state_dict()),'optimizer':copy.deepcopy(opt.state_dict()),'torch_rng':torch.get_rng_state().clone(),'numpy_rng':copy.deepcopy(rng.bit_generator.state),'python_rng':random.getstate(),'update':i,'mean':mean.tolist(),'std':std.tolist(),'name':name,'seed':seed}
 def update():
  m.train();x,t,s=edge.batch(tr,rng);u,v=m(x);u=u[:,:t.shape[1]];v=v[:,:t.shape[1]]
  loss=(.5*((u-t)/v).square()+v.log())[s].mean();opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
  value=float(loss.detach());assert math.isfinite(value);return value
 best=None;trace=[]
 for i in range(updates):
  loss=update()
  if (i+1)%max(1,updates//5)==0 or i+1==updates:
   val=edge.val(edge.predict(m,selection),std[0]);trace.append([i+1,loss,val]);assert math.isfinite(val)
   if best is None or val<best[0]:best=(val,snap(i+1))
 torch.save(snap(updates),path/'final.pt');torch.save(best[1],path/'selected.pt')
 # Exact one-update replay from final snapshot, not additional admitted fit.
 l=update();state=copy.deepcopy(m.state_dict())
 c=torch.load(path/'final.pt',weights_only=True);m.load_state_dict(c['model']);opt.load_state_dict(c['optimizer']);torch.set_rng_state(c['torch_rng']);rng.bit_generator.state=c['numpy_rng'];random.setstate(c['python_rng'])
 assert l==update() and all(torch.equal(state[k],v) for k,v in m.state_dict().items())
 return {'name':name,'seed':seed,'params':sum(x.numel() for x in m.parameters()),'selected':best[1]['update'],'trace':trace,'val':best[0],'resume_exact':True,'checkpoint_sha256':{p.name:sha(p.read_bytes()) for p in path.glob('*.pt')}}

def calibrate(edge,m,panels,source):
 # Calibration only: normalized absolute forecast residual divided by the model scale.
 pred=edge.predict(m,panels);out={}
 for j,h in enumerate(H):
  pooled=[];groups=[[] for _ in AGES];packs={};agepacks=[set() for _ in AGES]
  for (name,reg,x,t,s),(n,r,u,v,y,z) in zip(panels,pred):
   assert (name,reg)==(n,r);age=group_age(x[:,:t.shape[1]]);mask=z[...,j]
   score=np.abs(u[...,j]-y[...,j])/v[...,j]
   pooled.append(score[mask]);pack=keypack(source,name);packs.setdefault(pack,[]).append(score[mask])
   for a in range(3):
    which=mask&(age==a)
    if which.any():groups[a].append(score[which]);agepacks[a].add(pack)
  p=q90(np.concatenate(pooled));by_age=[];group_info=[]
  for a in range(3):
   count=sum(len(x) for x in groups[a]);enough=count>=100 and len(agepacks[a])>=3
   by_age.append(q90(np.concatenate(groups[a])) if enough else p)
   group_info.append({'age':AGES[a],'n':count,'pack_labels':len(agepacks[a]),'fallback':not enough})
  pq={k:q90(np.concatenate(v)) for k,v in packs.items() if sum(map(len,v))}
  assert len(pq)>=3
  out[str(h)]={'pooled':p,'age-stratified':by_age,'conservative-age':[max(p,x) for x in by_age],'pack-max':max([p,*pq.values()]),'groups':group_info,'pack_q90':pq}
 return out

def calc_row(e,width,mask):
 a=e[mask];w=width[mask]
 return {'n':int(len(a)),'sse':float(np.square(a,dtype=np.float64).sum()),'sae':float(np.abs(a).sum(dtype=np.float64)),'covered':int((np.abs(a)<=w).sum()),'width':float((2*w).sum(dtype=np.float64)),'signed_error':float(a.sum(dtype=np.float64))}

def summarize(rows):
 n=sum(r['n'] for r in rows)
 if not n:return {'n':0}
 return {'n':n,'rmse_v':math.sqrt(sum(r['sse'] for r in rows)/n),'mae_v':sum(r['sae'] for r in rows)/n,'coverage90':sum(r['covered'] for r in rows)/n,'width_v':sum(r['width'] for r in rows)/n,'bias_v':sum(r['signed_error'] for r in rows)/n}

def evaluate(edge,m,panels,mean,std,source,cut,cal,methods):
 preds=edge.predict(m,panels);rows=[];seg=[]
 for (name,reg,x,t,s),(n,r,u,v,y,z) in zip(panels,preds):
  assert (name,reg)==(n,r)
  xx=x[:,:t.shape[1]];age=group_age(xx)
  last=xx[...,6:8].numpy()*std+mean
  groups={'age':(age,AGES),'last-observed-voltage':(np.digitize(last[...,0],cut[0]),('low','mid','high')),'last-observed-current':(np.digitize(last[...,1],cut[1]),('low','mid','high'))}
  for j,h in enumerate(H):
   mask=z[...,j];e=(u[...,j]-y[...,j])*std[0]
   for method in methods:
    q=cal[str(h)][method];q=np.asarray(q,dtype=np.float32)[age] if isinstance(q,list) else float(q)
    w=v[...,j]*std[0]*q
    assert np.isfinite(w).all() and (w>=0).all()
    tags={'flight':name,'pack_label':keypack(source,name),'regime':reg,'horizon':h,'method':method}
    rows.append(tags|calc_row(e,w,mask))
    for group,(idx,labels) in groups.items():
     for i,label in enumerate(labels):seg.append(tags|{'group':group,'bin':label}|calc_row(e,w,mask&(idx==i)))
 summary={f'{m}/{r}/{h}':summarize([x for x in rows if x['method']==m and x['regime']==r and x['horizon']==h]) for m in methods for r in REGIMES for h in H}
 stratified={f'{m}/{r}/{h}/{g}/{b}':summarize([x for x in seg if x['method']==m and x['regime']==r and x['horizon']==h and x['group']==g and x['bin']==b]) for m in methods for r in REGIMES for h in H for g,labels in [('age',AGES),('last-observed-voltage',('low','mid','high')),('last-observed-current',('low','mid','high'))] for b in labels}
 # Every stratum partition must account for exactly the original targets.
 for m in methods:
  for r in REGIMES:
   for h in H:
    n=summary[f'{m}/{r}/{h}']['n']
    for g,labels in [('age',AGES),('last-observed-voltage',('low','mid','high')),('last-observed-current',('low','mid','high'))]:assert sum(stratified[f'{m}/{r}/{h}/{g}/{b}']['n'] for b in labels)==n
 for r in REGIMES:
  for h in H:
   first=summary[f'{methods[0]}/{r}/{h}']
   for m in methods[1:]:
    z=summary[f'{m}/{r}/{h}'];assert z['n']==first['n'] and z['rmse_v']==first['rmse_v'] and z['mae_v']==first['mae_v']
 return {'test':summary,'strata':stratified,'flight_metrics':rows,'flight_strata':seg}

def main():
 p=argparse.ArgumentParser()
 for n in ('restored','dataset','archived','fold','out'):p.add_argument('--'+n,required=True)
 a=p.parse_args();sys.path.insert(0,a.restored);import edge
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 assert a.fold in 'YRG' and len(a.fold)==1
 root=pathlib.Path(a.archived);path=pathlib.Path(a.dataset);out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=False)
 raw=(root/'source-manifest.json').read_bytes();assert raw==path.with_suffix('.json').read_bytes()
 meta=json.loads(raw);source={x['key']:x for x in meta['source']};old=meta['splits'][a.fold]
 data=np.load(path,allow_pickle=False);assert len(source)==215 and set(data.files)==set(source)
 for k in data.files:assert len(data[k])==source[k]['seconds'] and sha(data[k].astype('<f4').tobytes())==source[k]['array_sha256']
 split=json.loads(pathlib.Path(__file__).with_name('splits.json').read_text())[a.fold]
 assert split['train']==old['train'] and split['test']==old['test']
 assert set(split['selection'])|set(split['calibration'])==set(old['dev'])
 groups=[set(keypack(source,k) for k in ids) for ids in split.values()]
 for i in range(len(groups)):
  for j in range(i):assert not groups[i]&groups[j]
 for part,ids in split.items():
  assert len(ids)==len(set(ids))
  assert {source[k]['vehicle'] for k in ids}==({a.fold} if part=='test' else set('YRG')-{a.fold})
 tr_raw=np.concatenate([data[n].astype(np.float32) for n in split['train']]);mean=tr_raw.mean(0);std=tr_raw.std(0);assert (std>0).all()
 cut=[np.quantile(tr_raw[:,j],[1/3,2/3]).tolist() for j in range(2)]
 norm=lambda names:[(data[n].astype(np.float32)-mean)/std for n in names]
 tr=norm(split['train']);selection=edge.panels(norm(split['selection']),split['selection'],'dev-v1')
 # Do not construct calibration or test panels until all 15 fits have been selected and saved.
 result={'scope':'offline developmental calibration study; previously scored test cohort; no population coverage guarantee','fold':a.fold,'source_manifest_sha256':sha(raw),'splits':split,'mean':mean.tolist(),'std':std.tolist(),'strata_cutpoints':cut,'updates':600,'fits':[],'historical_cog':[],'env':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__}}
 for seed in edge.SEEDS:
  for name in edge.ARMS:
   f=fit(edge,name,seed,tr,selection,mean,std,600,out/f'{seed}-{name}');result['fits'].append(f)
   write(out/'results.json',result);print('fit',a.fold,name,seed,f['selected'],flush=True)
 calibration=edge.panels(norm(split['calibration']),split['calibration'],'cal-v2');test=edge.panels(norm(split['test']),split['test'],'test-v1')
 for f in result['fits']:
  m=edge.make(f['name']);m.load_state_dict(torch.load(out/f"{f['seed']}-{f['name']}"/'selected.pt',weights_only=True)['model']);m.eval()
  c=calibrate(edge,m,calibration,source);f['calibration']=c
  f['evaluation']=evaluate(edge,m,test,mean,std,source,cut,c,METHODS)
  write(out/'results.json',result)
  print('calibrated',a.fold,f['name'],f['seed'],{k:round(f['evaluation']['test'][f'{k}/0.15-1/15']['coverage90'],4) for k in METHODS},flush=True)
 # Historical residual diagnostic uses original weights and original interval factors only.
 original=read_original(root,a.fold)
 for f in original['fits']:
  if f['name']!='cog':continue
  m=edge.make('cog');m.load_state_dict(torch.load(root/f'expanded-{a.fold}'/f"{f['seed']}-cog"/'selected.pt',weights_only=True)['model']);m.eval()
  factors={str(h):{'original':f['q'][j]} for j,h in enumerate(H)}
  e=evaluate(edge,m,test,mean,std,source,cut,factors,('original',))
  for r in REGIMES:
   for h in H:
    old=f['test'][f'{r}/{h}'];new=e['test'][f'original/{r}/{h}']
    assert old['n']==new['n'] and all(abs(old[k]-new[k])<2e-5 for k in ('rmse_v','coverage90','width_v'))
  result['historical_cog'].append({'seed':f['seed'],'q':f['q'],'evaluation':e})
  write(out/'results.json',result)
 result['complete']=True;write(out/'results.json',result)
def read_original(root,g):return json.loads((root/f'expanded-{g}'/'results.json').read_text())
if __name__=='__main__':main()
