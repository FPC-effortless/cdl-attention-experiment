"""COG-EDGE v0.3: matched passive battery-telemetry development experiment."""
import argparse,copy,hashlib,json,math,pathlib,random,statistics,sys
import numpy as np
import torch
from torch import nn

H=(0,5,15)
REGIMES=('0.15-0','0.15-1','0.5-0','0.5-1')
SEEDS=(171,311,591)
ARMS=tuple((bank,history,loss) for bank in ('selective','dense') for history in ('recurrent','local') for loss in ('standard','weighted'))
def sha(b):return hashlib.sha256(b).hexdigest()
def write(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def avg(x):return statistics.mean(x)
def model_hash(m):
 h=hashlib.sha256()
 for k,v in sorted(m.state_dict().items()):
  h.update(k.encode()+b'\0');h.update(str(v.dtype).encode()+str(tuple(v.shape)).encode());h.update(v.detach().contiguous().cpu().numpy().tobytes())
 return h.hexdigest()
def equal(a,b):
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b

def make(edge,bank,history):
 mode='cog' if bank=='selective' else 'dense'
 if history=='recurrent':return edge.Edge(mode)
 class LocalEdge(edge.Edge):
  def forward(self,x):
   a,b,c,d=self.coeff(x)
   return self.emit(edge.scan(a,b),d,x)
  def init(self,b):return (torch.zeros(b,2,8),)
  def step(self,x,s):
   a,b,c,d=self.coeff(x);h=a*s[0]+b
   return self.emit(h,d,x),(h,)
 return LocalEdge(mode)

@torch.no_grad()
def parity(m,x,bank):
 m.eval();s=m.init(len(x));ref=torch.cat(m(x),-1);outputs=[];hold=0;mid=x.shape[1]//2
 for j in range(x.shape[1]):
  prev=s;o,s=m.step(x[:,j],s);outputs.append(torch.cat(o,-1))
  if bank=='selective':
   quiet=~x[:,j,2:4].bool()
   assert torch.equal(prev[0][quiet],s[0][quiet]);hold+=int(quiet.sum())*8
  if j==mid-1:stored=tuple(v.clone() for v in s)
 streamed=torch.stack(outputs,1);delta=float((ref-streamed).abs().max());assert delta<2e-5
 s=stored;tail=[]
 for j in range(mid,x.shape[1]):
  o,s=m.step(x[:,j],s);tail.append(torch.cat(o,-1))
 assert torch.equal(torch.stack(tail,1),streamed[:,mid:])
 prefix=float((torch.cat(m(x[:,:mid]),-1)-ref[:,:mid]).abs().max());assert prefix<2e-5
 changed=x.clone();changed[:,mid:]=0
 assert float((torch.cat(m(changed),-1)[:,:mid]-ref[:,:mid]).abs().max())<2e-5
 return {'hold_coordinates':hold,'batch_stream_delta':delta,'prefix_delta':prefix,'restart_exact':True,'state_bytes_fp32':sum(v.numel()*v.element_size() for v in stored)//len(x)}

def fit(edge,arm,seed,tr,selection,mean,std,high,updates,out):
 bank,history,policy=arm;name='-'.join(arm)
 torch.manual_seed(seed);random.seed(seed);rng=np.random.default_rng(seed)
 m=make(edge,bank,history);opt=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=.002);out.mkdir(parents=True,exist_ok=False)
 initial_hash=model_hash(m);batch_digest=hashlib.sha256()
 def snap(i):
  return {'model':copy.deepcopy(m.state_dict()),'optimizer':copy.deepcopy(opt.state_dict()),'torch_rng':torch.get_rng_state().clone(),'numpy_rng':copy.deepcopy(rng.bit_generator.state),'python_rng':random.getstate(),'update':i,'mean':mean.tolist(),'std':std.tolist(),'arm':arm,'seed':seed,'high_current_z':high}
 def update(record):
  m.train();x,t,s=edge.batch(tr,rng)
  if record:
   for z in (x,t,s):batch_digest.update(z.contiguous().numpy().tobytes())
  u,v=m(x);u=u[:,:t.shape[1]];v=v[:,:t.shape[1]]
  nll=.5*((u-t)/v).square()+v.log()
  if policy=='weighted':
   age=torch.expm1(x[:,:t.shape[1],4]*4)
   w=1+.5*(age>=15.5).float()+.5*(x[:,:t.shape[1],7]>=high).float()
   w=w.unsqueeze(-1).expand_as(nll)[s]
   loss=(nll[s]*w).sum()/w.sum()
  else:loss=nll[s].mean()
  opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
  value=float(loss.detach());assert math.isfinite(value)
  return value
 torch.save(snap(0),out/'initial.pt')
 best=None;trace=[]
 for i in range(updates):
  l=update(True)
  if (i+1)%120==0 or i+1==updates:
   val=edge.val(edge.predict(m,selection),std[0]);assert math.isfinite(val);trace.append([i+1,l,val])
   if best is None or val<best[0]:best=(val,snap(i+1))
 torch.save(snap(updates),out/'final.pt');torch.save(best[1],out/'selected.pt')
 l=update(False);post=copy.deepcopy(m.state_dict());post_opt=copy.deepcopy(opt.state_dict())
 c=torch.load(out/'final.pt',weights_only=True);m.load_state_dict(c['model']);opt.load_state_dict(c['optimizer']);torch.set_rng_state(c['torch_rng']);rng.bit_generator.state=c['numpy_rng'];random.setstate(c['python_rng'])
 assert l==update(False) and equal(post,m.state_dict()) and equal(post_opt,opt.state_dict())
 m.load_state_dict(best[1]['model']);m.eval()
 # Contract check is on a fixed training input; not a performance evaluation.
 x,*_=edge.batch(tr,np.random.default_rng(7701))
 check=parity(m,x[:1],bank)
 return {'arm':name,'seed':seed,'params':sum(p.numel() for p in m.parameters()),'initial_model_sha256':initial_hash,'training_batches_sha256':batch_digest.hexdigest(),'selected_update':best[1]['update'],'selection_score':best[0],'trace':trace,'resume_exact':True,'contract':check,'checkpoint_sha256':{p.name:sha(p.read_bytes()) for p in out.glob('*.pt')}}

def row(err,sig,nll,z):
 e=err[z];w=sig[z]
 return {'n':int(len(e)),'sse':float(np.square(e,dtype=np.float64).sum()),'sae':float(np.abs(e).sum(dtype=np.float64)),'bias':float(e.sum(dtype=np.float64)),'covered':int((abs(e)<=1.6448536269514722*w).sum()),'width':float((2*1.6448536269514722*w).sum(dtype=np.float64)),'nll':float(nll[z].sum(dtype=np.float64))}
def aggregate(rows):
 n=sum(r['n'] for r in rows)
 if n==0:return {'n':0}
 return {'n':n,'rmse_v':math.sqrt(sum(r['sse'] for r in rows)/n),'mae_v':sum(r['sae'] for r in rows)/n,'bias_v':sum(r['bias'] for r in rows)/n,'nominal_gaussian_coverage90':sum(r['covered'] for r in rows)/n,'width_v':sum(r['width'] for r in rows)/n,'normalized_nll':sum(r['nll'] for r in rows)/n}

def evaluate(edge,m,panels,std,high):
 preds=edge.predict(m,panels);raw=[];strata=[]
 for (name,reg,x,t,s),(n,r,u,v,y,z) in zip(panels,preds):
  assert (n,r)==(name,reg);xx=x[:,:t.shape[1]].numpy()
  age=np.expm1(xx[...,4]*4);old=age>=15.5;high_current=xx[...,7]>=high
  masks={'stale':old,'fresh':~old,'high-current':high_current,'other-current':~high_current,'stale-high-current':old&high_current,'other-joint':~(old&high_current)}
  for j,h in enumerate(H):
   err=(u[...,j]-y[...,j])*std;sig=v[...,j]*std
   nll=.5*((u[...,j]-y[...,j])/v[...,j])**2+np.log(v[...,j]);target=z[...,j]
   assert np.isfinite(err).all() and np.isfinite(sig).all() and (sig>0).all()
   tag={'flight':name,'regime':reg,'horizon':h}
   raw.append(tag|row(err,sig,nll,target))
   for group,mask in masks.items():strata.append(tag|{'group':group}|row(err,sig,nll,target&mask))
 results={};sresult={}
 for reg in REGIMES:
  for h in H:
   k=f'{reg}/{h}'
   results[k]=aggregate([r for r in raw if r['regime']==reg and r['horizon']==h])
   for g in masks:sresult[f'{k}/{g}']=aggregate([r for r in strata if r['regime']==reg and r['horizon']==h and r['group']==g])
   for a,b in [('stale','fresh'),('high-current','other-current'),('stale-high-current','other-joint')]:
    assert sresult[f'{k}/{a}']['n']+sresult[f'{k}/{b}']['n']==results[k]['n']
 return {'metrics':results,'strata':sresult,'flight_metrics':raw,'flight_strata':strata}

def main():
 p=argparse.ArgumentParser()
 for n in ('restored','dataset','source','splits','fold','out'):p.add_argument('--'+n,required=True)
 a=p.parse_args();sys.path.insert(0,a.restored);import edge
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 assert a.fold in 'YRG' and len(a.fold)==1
 path=pathlib.Path(a.dataset);raw=pathlib.Path(a.source).read_bytes();assert raw==path.with_suffix('.json').read_bytes()
 meta=json.loads(raw);source={r['key']:r for r in meta['source']};data=np.load(path,allow_pickle=False)
 assert len(source)==215 and set(source)==set(data.files)
 for k in data.files:assert len(data[k])==source[k]['seconds'] and sha(data[k].astype('<f4').tobytes())==source[k]['array_sha256']
 split=json.loads(pathlib.Path(a.splits).read_text())[a.fold];old=meta['splits'][a.fold]
 assert split['train']==old['train'] and split['test']==old['test'] and set(split['selection'])|set(split['calibration'])==set(old['dev'])
 packs=[{source[k]['pack'] for k in ids} for ids in split.values()]
 assert all(not packs[i]&packs[j] for i in range(4) for j in range(i))
 for name in ('train','selection','calibration'):
  assert all(source[k]['vehicle']!=a.fold for k in split[name])
 assert all(source[k]['vehicle']==a.fold for k in split['test'])
 train=np.concatenate([data[k].astype(np.float32) for k in split['train']]);mean=train.mean(0);std=train.std(0);assert (std>0).all()
 high=float(np.quantile((train[:,1]-mean[1])/std[1],2/3))
 def norm(part):
  assert part!='test'
  return [(data[k].astype(np.float32)-mean)/std for k in split[part]]
 tr=norm('train');selection=edge.panels(norm('selection'),split['selection'],'dev-v1')
 out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=False)
 result={'scope':'development only; calibration subset reused for diagnostics, not independent qualification; test arrays hash-verified but not evaluated or used for training/selection','fold':a.fold,'source_manifest_sha256':sha(raw),'splits':split,'mean':mean.tolist(),'std':std.tolist(),'high_current_z':high,'updates':600,'fits':[],'env':{'python':sys.version,'numpy':np.__version__,'torch':torch.__version__}}
 for seed in SEEDS:
  for arm in ARMS:
   f=fit(edge,arm,seed,tr,selection,mean,std,high,600,out/f"{seed}-{'-'.join(arm)}");result['fits'].append(f)
   write(out/'results.json',result);print('fit',a.fold,seed,f['arm'],f['selected_update'],flush=True)
  same=[f for f in result['fits'] if f['seed']==seed]
  assert len(same)==8 and len({f['initial_model_sha256'] for f in same})==1 and len({f['training_batches_sha256'] for f in same})==1 and {f['params'] for f in same}=={1914}
 # Construct developmental diagnostic panels after all selected weights are saved.
 diag=edge.panels(norm('calibration'),split['calibration'],'cal-v2')
 counts=None
 for f in result['fits']:
  bank,history,policy=f['arm'].split('-');m=make(edge,bank,history)
  c=torch.load(out/f"{f['seed']}-{f['arm']}"/'selected.pt',weights_only=True);m.load_state_dict(c['model']);m.eval()
  e=evaluate(edge,m,diag,std[0],high);f['evaluation']=e
  current={(r['flight'],r['regime'],r['horizon']):r['n'] for r in e['flight_metrics']}
  if counts is None:counts=current
  assert counts==current
  print('diagnostic',a.fold,f['seed'],f['arm'],e['metrics']['0.15-1/15']['rmse_v'],flush=True)
  write(out/'results.json',result)
 result['complete']=True;write(out/'results.json',result)
 keys=[f'{r}/{h}' for r in REGIMES for h in H]
 summary={'fold':a.fold,'fits':len(result['fits']),'source_manifest_sha256':sha(raw),'matched_target_cells':len(counts),'total_hold_coordinates':sum(f['contract']['hold_coordinates'] for f in result['fits']),'arms':{}}
 for arm in ARMS:
  name='-'.join(arm);rr=[f for f in result['fits'] if f['arm']==name]
  metrics={k:{field:avg([f['evaluation']['metrics'][k][field] for f in rr]) for field in ('rmse_v','mae_v','bias_v','nominal_gaussian_coverage90','width_v','normalized_nll')} for k in keys}
  strata={}
  for k in rr[0]['evaluation']['strata']:
   vals=[f['evaluation']['strata'][k] for f in rr];assert len({v['n'] for v in vals})==1
   strata[k]={'n':vals[0]['n']}|({field:avg([v[field] for v in vals]) for field in ('rmse_v','mae_v','bias_v','nominal_gaussian_coverage90','width_v','normalized_nll')} if vals[0]['n'] else {})
  summary['arms'][name]={'metrics':metrics,'strata':strata,'state_bytes_fp32':rr[0]['contract']['state_bytes_fp32'],'params':rr[0]['params']}
 write(out/'summary.json',summary)
if __name__=='__main__':main()
