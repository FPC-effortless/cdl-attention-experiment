"""Matched supervised operation/argument parser training. No automatic admission."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,json,time,dataclasses
from pathlib import Path
import torch
from models import Config,SentenceParser,tensorize_texts,supervised_loss

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'))
def configure():
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.set_float32_matmul_precision('highest')
def runtime(device):return {'torch':str(torch.__version__),'cuda':torch.version.cuda,'device':device,'gpu':torch.cuda.get_device_name(0) if device.startswith('cuda') else None,'threads':torch.get_num_threads(),'deterministic':torch.are_deterministic_algorithms_enabled(),'cublas':os.environ.get('CUBLAS_WORKSPACE_CONFIG')}
def save(p,x):
 p=Path(p);tmp=p.with_suffix('.tmp');torch.save(x,tmp);os.replace(tmp,p)
def train(examples_path,out,arm,seed,updates,batch=64,lr=3e-4,device='cpu',resume=None,stop=None):
 configure();torch.manual_seed(seed)
 examples=[json.loads(s) for s in Path(examples_path).read_text().splitlines() if s.strip()]
 ids=[x['id'] for x in examples]
 if not ids or len(set(ids))!=len(ids):raise ValueError('nonunique/empty training roster')
 config=Config(arm=arm);cfg={'model':dataclasses.asdict(config),'seed':seed,'updates':updates,'batch':batch,'lr':lr,'betas':[.9,.999],'eps':1e-8,'weight_decay':.01,'clip':1.}
 if updates<1 or batch<1:raise ValueError('positive budget required')
 bindings={'config':cfg,'data_sha256':sha(examples_path),'sources':{n:sha(Path(__file__).parent/n) for n in ('train.py','models.py')},'runtime':runtime(device)}
 model=SentenceParser(config).to(device);opt=torch.optim.AdamW(model.parameters(),lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=.01)
 sampler=torch.Generator().manual_seed(seed+10000);start=0
 if resume:
  old=torch.load(resume,map_location='cpu',weights_only=True)
  if old['bindings']!=bindings:raise ValueError('resume bindings differ')
  model.load_state_dict(old['model'],strict=True);opt.load_state_dict(old['optimizer']);sampler.set_state(old['sampler']);torch.set_rng_state(old['torch_rng']);start=old['updates']
  if device.startswith('cuda'):torch.cuda.set_rng_state_all(old['cuda_rng'])
 end=updates if stop is None else min(stop,updates)
 if not start<end:raise ValueError('empty/reversed segment')
 out=Path(out);out.mkdir(parents=True,exist_ok=False);model.train();begin=time.perf_counter()
 def checkpoint(u):return {'schema':'parallel022.parser.v1','model':model.state_dict(),'optimizer':opt.state_dict(),'updates':u,'bindings':bindings,'sampler':sampler.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.startswith('cuda') else [],'parameters':sum(p.numel() for p in model.parameters())}
 with open(out/'updates.jsonl','w') as f:
  for u in range(start,end):
   t=time.perf_counter();idx=torch.randint(len(examples),(batch,),generator=sampler).tolist();rows=[examples[i] for i in idx]
   x,l=tensorize_texts([q['text'] for q in rows],device,config.max_length);opt.zero_grad(set_to_none=True)
   z=model(x,l);loss=supervised_loss(z,[q['labels'] for q in rows])
   if not torch.isfinite(loss):raise ValueError('nonfinite loss')
   loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
   if device.startswith('cuda'):torch.cuda.synchronize()
   f.write(canonical({'update':u+1,'loss':float(loss.detach()),'grad_norm':float(norm),'instructions':batch,'actual_input_bytes':int(l.sum()),'sampled_ids_sha256':hashlib.sha256(canonical([ids[i] for i in idx]).encode()).hexdigest(),'seconds':time.perf_counter()-t})+'\n');f.flush()
   if (u+1)%256==0 or u+1==end:save(out/'checkpoint.pt',checkpoint(u+1))
 result={'start_update':start,'end_update':end,'optimizer_updates':end-start,'instruction_exposures':batch*(end-start),'parameters':sum(p.numel() for p in model.parameters()),'training_seconds':time.perf_counter()-begin,'bindings':bindings}
 (out/'result.json').write_text(json.dumps(result,indent=2));return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--examples',required=True);p.add_argument('--out',required=True);p.add_argument('--arm',choices=['parallel','gru'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--updates',type=int,required=True);p.add_argument('--batch',type=int,default=64);p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--device',default='cpu');p.add_argument('--resume');p.add_argument('--stop',type=int);a=p.parse_args();print(json.dumps(train(a.examples,a.out,a.arm,a.seed,a.updates,a.batch,a.lr,a.device,a.resume,a.stop)))
