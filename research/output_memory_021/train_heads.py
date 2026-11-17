"""Matched 021 frozen-condition head training. No automatic launch/admission."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,json,time
from pathlib import Path
import torch
from torch.nn import functional as F
from heads import OutputHead,ar_inputs,MASK,PAD

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'))
def configure():
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.set_float32_matmul_precision('highest')
def runtime():
 return {'torch':str(torch.__version__),'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'deterministic':torch.are_deterministic_algorithms_enabled(),'threads':torch.get_num_threads(),'cublas':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32,'cudnn_deterministic':torch.backends.cudnn.deterministic,'matmul_precision':torch.get_float32_matmul_precision()}
def save_atomic(p,obj):
 p=Path(p);tmp=p.with_suffix('.tmp');torch.save(obj,tmp);os.replace(tmp,p)
def train(cache_path,out,mode,condition,seed,updates,batch=128,width=64,layers=2,lr=3e-4,device='cpu',resume=None,stop=None,save_every=256):
 if mode not in ('ar','diffusion') or condition not in ('learned','exact'):raise ValueError('invalid arm')
 if updates<1 or batch<1:raise ValueError('positive budget required')
 configure();torch.manual_seed(seed)
 data=torch.load(cache_path,map_location='cpu',weights_only=True)
 features=data['features' if condition=='learned' else 'exact_features'];targets=data['targets'];ids=data['ids']
 if features.ndim!=2 or targets.ndim!=2 or len(ids)!=len(features) or len(set(ids))!=len(ids):raise ValueError('invalid cache roster')
 if targets.shape!=(len(features),64) or features.shape[1]!=392 or not torch.isfinite(features).all():raise ValueError('invalid cache shape/values')
 if targets.dtype!=torch.long or targets.min()<0 or targets.max()>PAD:raise ValueError('invalid target support')
 cfg={'mode':mode,'condition':condition,'seed':seed,'updates':updates,'batch':batch,'width':width,'layers':layers,'lr':lr,'length':64,'condition_dim':392,'time_min':1e-4,'padding_loss':'all64 positions, including EOS/PAD','weight_decay':0.01,'betas':[0.9,0.999],'eps':1e-8,'gradient_clip_norm':1.0}
 bindings={'config':cfg,'cache_sha256':sha(cache_path),'cache_bindings':data['bindings'],'source':{n:sha(Path(__file__).parent/n) for n in ('heads.py','train_heads.py')},'runtime':{**runtime(),'requested_device':device}}
 head=OutputHead(392,width,layers,max_length=64).to(device);opt=torch.optim.AdamW(head.parameters(),lr=lr,weight_decay=0.01,betas=(0.9,0.999),eps=1e-8)
 sampler=torch.Generator().manual_seed(seed+10000);time_rng=torch.Generator().manual_seed(seed+20000);noise=torch.Generator(device=device).manual_seed(seed+30000)
 start=0
 if resume:
  old=torch.load(resume,map_location='cpu',weights_only=True)
  if old['bindings']!=bindings:raise ValueError('resume binding mismatch')
  start=old['updates'];head.load_state_dict(old['model'],strict=True);opt.load_state_dict(old['optimizer']);sampler.set_state(old['sampler']);time_rng.set_state(old['time_rng']);noise.set_state(old['noise_rng'])
  torch.set_rng_state(old['torch_rng'])
  if device.startswith('cuda'):torch.cuda.set_rng_state_all(old['cuda_rng'])
 end=updates if stop is None else min(stop,updates)
 if not start<end:raise ValueError('empty/reversed segment')
 out=Path(out);out.mkdir(parents=True,exist_ok=False);features=features.to(device);targets=targets.to(device);head.train();beg=time.perf_counter()
 def checkpoint(u):
  return {'schema':'output021.head.v1','model':head.state_dict(),'optimizer':opt.state_dict(),'updates':u,'bindings':bindings,'sampler':sampler.get_state(),'time_rng':time_rng.get_state(),'noise_rng':noise.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.startswith('cuda') else [],'parameters':sum(p.numel() for p in head.parameters())}
 with open(out/'updates.jsonl','w') as trace:
  for u in range(start,end):
   update_start=time.perf_counter()
   idx=torch.randint(len(ids),(batch,),generator=sampler);ix=idx.to(device);c=features[ix];clean=targets[ix];opt.zero_grad(set_to_none=True)
   if mode=='ar':
    loss=F.cross_entropy(head(ar_inputs(clean),c,causal=True).transpose(1,2),clean);t=0.;masked=0
   else:
    t=1e-4+(1-1e-4)*torch.rand((),generator=time_rng).item();keep=torch.rand(clean.shape,device=device,generator=noise)<t;noisy=torch.where(keep,MASK,clean)
    ce=F.cross_entropy(head(noisy,c,t).transpose(1,2),clean,reduction='none');loss=(ce*keep).mean()/t;masked=int(keep.sum())
   if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
   loss.backward();norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True);opt.step()
   trace.write(canonical({'update':u+1,'loss':float(loss.detach()),'gradient_norm':float(norm),'time':t,'masked_targets':masked,'answer_exposures':batch,'target_positions':batch*64,'nonpad_targets':int((clean!=PAD).sum()),'step_elapsed_before_trace_write_seconds':time.perf_counter()-update_start,'sampled_ids_sha256':hashlib.sha256(canonical([ids[i] for i in idx.tolist()]).encode()).hexdigest()})+'\n');trace.flush()
   if (u+1)%save_every==0 or u+1==end:save_atomic(out/'checkpoint.pt',checkpoint(u+1))
 if device.startswith('cuda'):torch.cuda.synchronize()
 result={'start_update':start,'end_update':end,'training_updates':end-start,'answer_exposures':(end-start)*batch,'target_positions':(end-start)*batch*64,'elapsed_seconds':time.perf_counter()-beg,'parameters':sum(p.numel() for p in head.parameters()),'bindings':bindings,'scope':'frozen-condition output heads, not conditioner training or qualification'}
 (out/'result.json').write_text(json.dumps(result,indent=2));return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--out',required=True);p.add_argument('--mode',choices=['ar','diffusion'],required=True);p.add_argument('--condition',choices=['learned','exact'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--updates',type=int,required=True);p.add_argument('--batch',type=int,default=128);p.add_argument('--width',type=int,default=64);p.add_argument('--layers',type=int,default=2);p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--device',default='cpu');p.add_argument('--resume');p.add_argument('--stop',type=int);a=p.parse_args();print(json.dumps(train(a.cache,a.out,a.mode,a.condition,a.seed,a.updates,a.batch,a.width,a.layers,a.lr,a.device,a.resume,a.stop)))
