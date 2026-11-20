"""Draft full-sequence mixed-language/task trainer. Not admitted for research execution.
No token-boundary state reset inside an example; every task history has full BPTT.
Resume includes optimizer, RNG and exact update count. Native TrainLab wrapper pending.
"""
import argparse,hashlib,json,random,time,os
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch.nn import functional as F
from models import make

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(c):
 torch.set_num_threads(1)
 out=Path(c.output);out.mkdir(parents=True,exist_ok=True);torch.manual_seed(c.seed);random.seed(c.seed)
 torch.use_deterministic_algorithms(True)
 device='cuda' if torch.cuda.is_available() else 'cpu';m=make(c.arm,width=c.width,slots=c.slots).to(device);opt=torch.optim.AdamW(m.parameters(),lr=c.lr)
 docs=[json.loads(x)['text'].encode('utf-8') for x in Path(c.prose).read_text().splitlines()];docs=[x for x in docs if len(x)>c.prose_context+1];
 if not docs:raise ValueError('no adequate length prose documents')
 tasks=[json.loads(x) for x in Path(c.tasks).read_text().splitlines()];rng=random.Random(c.seed);start=0;targets=0
 runtime={'torch':torch.__version__,'cuda':torch.version.cuda,'device':device,'gpu':torch.cuda.get_device_name(0) if device=='cuda' else None,'deterministic':torch.are_deterministic_algorithms_enabled()}
 if c.runtime_manifest:
  expected=json.loads(Path(c.runtime_manifest).read_text())
  if expected['runtime']!=runtime:raise ValueError('runtime mismatch')
 bindings={'runtime':runtime,'runtime_manifest_sha':sha(c.runtime_manifest) if c.runtime_manifest else None,'slots':c.slots,'vocab':256,'transformer_layers':2,'max_context':16384,'prose':sha(c.prose),'tasks':sha(c.tasks),'arm':c.arm,'width':c.width,'seed':c.seed,'lr':c.lr,'prose_context':c.prose_context,'task_batch':c.task_batch,'prose_batch':c.prose_batch,'answer_weight':c.answer_weight,'source_sha':{p:sha(Path(__file__).parent/p) for p in ['train.py','models.py','native_csm.py']}}
 if c.resume:
  q=torch.load(c.resume,map_location='cpu',weights_only=False)
  if q['bindings']!=bindings:raise ValueError('resume binding mismatch')
  m.load_state_dict(q['model']);opt.load_state_dict(q['optimizer']);rng.setstate(q['sampler']);torch.set_rng_state(q['cpu_rng'].cpu());random.setstate(q['python_rng'])
  if device=='cuda':torch.cuda.set_rng_state_all([s.cpu() for s in q['cuda_rng']])
  start=q['updates'];targets=q['targets']
 initial_targets=targets;sequence_count=0;schedule=[];run_started=time.perf_counter()
 if start>=c.updates:raise ValueError('endpoint already complete')
 for step in range(start,c.updates):
  m.train();t=time.perf_counter()
  if step%2==0:
   bs=c.prose_batch;seq=[];selected=[]
   for _ in range(bs):
    di=rng.randrange(len(docs));text=docs[di];p=rng.randrange(len(text)-c.prose_context-1);seq.append(torch.tensor(list(text[p:p+c.prose_context+1]),dtype=torch.long));selected.append({'article_index':di,'offset':p})
   schedule.append({'update':step+1,'kind':'prose','samples':selected})
   b=torch.stack(seq).to(device);x,y=b[:,:-1],b[:,1:];weights=torch.ones_like(y,dtype=torch.float)
  else:
   chosen=[tasks[rng.randrange(len(tasks))] for _ in range(c.task_batch)];seq=[list(a['text'].encode('ascii')) for a in chosen];n=max(map(len,seq));b=torch.zeros(len(seq),n,dtype=torch.long,device=device);weights=torch.zeros(len(seq),n-1,device=device)
   for j,(row,a) in enumerate(zip(seq,chosen)):
    b[j,:len(row)]=torch.tensor(row,device=device);weights[j,:len(row)-1]=1.
    for ans in a['answers']:weights[j,ans['start']-1:ans['end']-1]=c.answer_weight
   x,y=b[:,:-1],b[:,1:]
   schedule.append({'update':step+1,'kind':'tasks','samples':[a['id'] for a in chosen]})
  sequence_count+=len(b)
  opt.zero_grad(set_to_none=True);z=m(x);losses=F.cross_entropy(z.flatten(0,1),y.flatten(),reduction='none').reshape_as(weights);loss=(losses*weights).sum()/weights.sum()
  if not torch.isfinite(loss):raise ValueError('nonfinite loss')
  loss.backward();gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step();targets+=int((weights>0).sum())
  if device=='cuda':torch.cuda.synchronize()
  rec={'sequences':len(b),'actual_targets':int((weights>0).sum()),'padded_token_positions':int(x.numel()),'samples':schedule[-1],'update':step+1,'targets':targets,'loss':float(loss.detach()),'grad_norm':float(gn),'seconds':time.perf_counter()-t}
  with (out/'updates.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n');f.flush()
  if (step+1)%c.segment==0 or step+1==c.updates:
   ck={'model':m.state_dict(),'optimizer':opt.state_dict(),'sampler':rng.getstate(),'cpu_rng':torch.get_rng_state(),'python_rng':random.getstate(),'cuda_rng':torch.cuda.get_rng_state_all() if device=='cuda' else [],'updates':step+1,'targets':targets,'bindings':bindings,'torch':torch.__version__}
   tmp=out/'checkpoint.tmp';torch.save(ck,tmp);os.replace(tmp,out/'checkpoint.pt');print(json.dumps(rec),flush=True)
   # Execution is externally segmented so durable backup is required before resume.
   result={'arm':c.arm,'seed':c.seed,'parameters':sum(p.numel() for p in m.parameters()),'started_update':start,'finished_update':step+1,'new_updates':step+1-start,'cumulative_targets':targets,'segment_targets':targets-initial_targets,'sequences':sequence_count,'schedule_sha256':hashlib.sha256(json.dumps(schedule,sort_keys=True).encode()).hexdigest(),'segment_seconds':time.perf_counter()-run_started,'device':device,'torch':torch.__version__,'bindings':bindings}
   (out/'schedule.json').write_text(json.dumps(schedule));(out/'result.json').write_text(json.dumps(result,indent=2))
   break
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--output',required=True);p.add_argument('--prose',required=True);p.add_argument('--tasks',required=True);p.add_argument('--resume');p.add_argument('--slots',type=int,default=16);p.add_argument('--runtime-manifest');p.add_argument('--seed',type=int,default=101);p.add_argument('--width',type=int,default=128);p.add_argument('--updates',type=int,default=1024);p.add_argument('--segment',type=int,default=256);p.add_argument('--prose-context',type=int,default=512);p.add_argument('--prose-batch',type=int,default=32);p.add_argument('--task-batch',type=int,default=32);p.add_argument('--answer-weight',type=float,default=4.);p.add_argument('--lr',type=float,default=.001)
 run(p.parse_args())
