"""Registered matched CSM training/evaluation; no protected-test scoring."""
import argparse, hashlib, json, math, platform, resource, sys, time, zipfile
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'csm_parity_002'))
from regulated import Regulated, State, base, ROOT
SHA='86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed'
ARMS=('instant','single','multi','multi_tb64')

def corpus():
 with zipfile.ZipFile(ROOT/'artifacts/csm-original-scope-reconstructed-20261109/CSM-GRAPH-001-verified-project.zip') as z:
  raw=z.read('CSM-GRAPH-001/csm-foundation/experiments/csm_foundation_001/data/input.txt')
 assert hashlib.sha256(raw).hexdigest()==SHA
 return torch.tensor(list(raw[:1003854])),torch.tensor(list(raw[1003854:1059624]))

def panels(dev):
 starts=torch.linspace(0,len(dev)-2049,16).long()
 assert all(int(starts[i+1]-starts[i])>=2049 for i in range(15))
 return torch.stack([dev[s:s+2049] for s in starts]),starts

def model(arm):return Regulated(base.Config(vocab_size=256),'multi' if arm=='multi_tb64' else arm).float()
def detach(s):
 c=s.core
 return State(base.RecordState(c.values.detach(),c.links.detach(),c.cursor.detach(),c.history.detach(),c.position),s.regulation.detach())
def state_dict(s):
 c=s.core
 return dict(values=c.values,links=c.links,cursor=c.cursor,history=c.history,position=c.position,regulation=s.regulation)

def evaluate(m,data,length=2048,intervention='none',details=True):
 m.eval();s=m.init_state(len(data),'cpu');losses=[];correct=[];conf=[];dest=torch.zeros(8);ops=torch.zeros(4)
 admit=links=gain=0.;rr=torch.zeros(3)
 with torch.no_grad():
  for t in range(length):
   if intervention=='trace_reset':s.regulation.zero_()
   if intervention=='trace_swap' and t==256:s.regulation=s.regulation.roll(1,0)
   if intervention=='record_reset' and t==256:s.core.values=torch.zeros_like(s.core.values)
   y,s,tr=m.step(data[:,t],s)
   losses.append(F.cross_entropy(y,data[:,t+1],reduction='none'))
   c,p=y.softmax(-1).max(-1);correct.append(p==data[:,t+1]);conf.append(c)
   if details:
    dest+=torch.bincount(tr['destination'],minlength=8);ops+=torch.bincount(tr['operator'],minlength=4)
    admit+=tr['edits_value'].sum().item();links+=tr['edits_link'].sum().item();gain+=tr['gate'].sum().item()
    rr+=s.regulation.square().reshape(len(data),3,4).mean((0,2))
 loss=torch.stack(losses,1).double();acc=torch.stack(correct,1).double();confidence=torch.stack(conf,1).double()
 nll=loss.mean().item();ece=0
 for i in range(10):
  mask=(confidence>=i/10)&(confidence<(i+1)/10 if i<9 else confidence<=1)
  if mask.any():ece+=mask.double().mean().item()*abs(acc[mask].mean().item()-confidence[mask].mean().item())
 out=dict(nll=nll,bpc=nll/math.log(2),perplexity=math.exp(nll),accuracy=acc.mean().item(),ece=ece,targets=loss.numel(),per_excerpt_nll=loss.mean(1).tolist(),post256_nll=loss[:,256:].mean().item() if length>256 else None)
 out['age_nll']={f'{a}:{min(b,length)}':loss[:,a:min(b,length)].mean().item() for a,b in ((0,32),(32,256),(256,1024),(1024,2048)) if a<length}
 if details:out['diagnostics']=dict(destination_counts=dest.tolist(),operator_counts=ops.tolist(),write_admission_rate=admit/loss.numel(),link_edit_rate=links/loss.numel(),mean_effective_gain=gain/loss.numel(),regulator_rms=(rr/length).sqrt().tolist())
 m.train();return out

def baselines(train,starts,dev):
 uni=np.zeros(256);bi=np.zeros((256,256))
 for row in starts:
  for s in row:
   stream=train[int(s):int(s)+2049].numpy();np.add.at(uni,stream[1:],1);np.add.at(bi,(stream[:-1],stream[1:]),1)
 up=(uni+.1)/(uni.sum()+25.6);bp=(bi+.1)/(bi.sum(1,keepdims=True)+25.6)
 x=dev[:,:-1].numpy();y=dev[:,1:].numpy()
 return dict(unigram_nll=float(-np.log(up[y]).mean()),bigram_nll=float(-np.log(bp[x,y]).mean()),fitted_targets=int(uni.sum()),scope='same sampled targets; not equal parameter or compute budgets')

def train(seed,arms=ARMS,updates=256,smoke=False):
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 train_data,dev_data=corpus();dev,devstarts=panels(dev_data)
 g=torch.Generator().manual_seed(seed+3000);starts=torch.randint(0,len(train_data)-2049,(32,4),generator=g)
 out=HERE/('smoke' if smoke else f'seed-{seed}');out.mkdir(exist_ok=True)
 (out/'baselines.json').write_text(json.dumps(baselines(train_data,starts,dev),indent=2))
 for arm in arms:
  if (out/f'{arm}-result.json').exists() and not smoke:
   print(json.dumps(dict(seed=seed,arm=arm,status='already completed')),flush=True);continue
  torch.manual_seed(seed);m=model(arm);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01)
  initial=hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in m.parameters())).hexdigest()
  curves=[dict(update=0,dev=evaluate(m,dev,512,details=False))];train_seconds=0.;losses=[]
  state=None;resume_at=0;prior_telemetry_complete=True
  checkpoint=out/f'{arm}-latest.pt'
  if checkpoint.exists() and not smoke:
   ck=torch.load(checkpoint,weights_only=True)
   assert ck['seed']==seed and ck['arm']==arm and torch.equal(ck['starts'],starts)
   resume_at=ck['update'];assert resume_at%8==0
   m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);torch.set_rng_state(ck['rng'])
   curves=json.loads((out/f'{arm}-curve.json').read_text())
   prior_telemetry_complete='train_seconds' in ck
   train_seconds=ck.get('train_seconds',0.);losses=ck.get('losses',[None]*resume_at)
   print(json.dumps(dict(seed=seed,arm=arm,resume_at=resume_at,prior_telemetry_complete=prior_telemetry_complete)),flush=True)
  for update in range(resume_at,updates):
   if update%8==0:
    episode=update//8;data=torch.stack([train_data[int(s):int(s)+2049] for s in starts[episode]])
    state=m.init_state(4,'cpu')
   tick=time.perf_counter();opt.zero_grad();start=(update%8)*256;window=64 if arm=='multi_tb64' else 256;total=0.
   for a in range(start,start+256,window):
    ys=[]
    for t in range(a,a+window):
     y,state,_=m.step(data[:,t],state);ys.append(y)
    loss=F.cross_entropy(torch.stack(ys,1).flatten(0,1),data[:,a+1:a+window+1].flatten())*(window/256)
    assert torch.isfinite(loss);loss.backward();total+=loss.item();state=detach(state)
   grad=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1.));assert math.isfinite(grad)
   gn={k:v.grad.reshape(3,4,-1).norm(dim=(1,2)).tolist() if k.startswith('driver') else v.grad.reshape(32,3,4).norm(dim=(0,2)).tolist() for k,v in m.named_parameters() if k.startswith(('driver.','injection.'))}
   opt.step();train_seconds+=time.perf_counter()-tick;losses.append(total)
   if update+1 in (32,64,128,256) or update+1==updates:
    row=dict(update=update+1,train_nll=total,preclip_gradient_norm=grad,trace_group_gradient_norms=gn,dev=evaluate(m,dev,512,details=False));curves.append(row)
    print(json.dumps(dict(seed=seed,arm=arm,update=update+1,train_nll=total,dev_nll=row['dev']['nll'])),flush=True)
    torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state(),state=state_dict(state),starts=starts,dev_starts=devstarts,seed=seed,arm=arm,update=update+1,train_seconds=train_seconds,losses=losses),out/f'{arm}-latest.pt')
    (out/f'{arm}-curve.json').write_text(json.dumps(curves,indent=2))
  length=64 if smoke else 2048
  evals={k:evaluate(m,dev,length,k) for k in ('none','trace_reset','trace_swap','record_reset')}
  first=torch.stack([train_data[int(s):int(s)+2049] for s in starts[0]])
  result=dict(seed=seed,arm=arm,parameters=sum(p.numel() for p in m.parameters()),initial_parameters_sha256=initial,training_updates=updates,training_targets=updates*1024,resume_at=resume_at,prior_telemetry_complete=prior_telemetry_complete,train_seconds=train_seconds if prior_telemetry_complete else None,measured_segment_seconds=train_seconds,training_tokens_per_second=updates*1024/train_seconds if prior_telemetry_complete else None,train_nll_curve=losses,train_first_episode=evaluate(m,first,length,details=False),evaluation=evals,process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,torch=torch.__version__,python=platform.python_version(),checkpoint_sha256=hashlib.sha256((out/f'{arm}-latest.pt').read_bytes()).hexdigest(),schedule_sha256=hashlib.sha256(starts.numpy().tobytes()).hexdigest(),corpus_sha256=SHA,protocol_sha256=hashlib.sha256((HERE/'PROTOCOL.md').read_bytes()).hexdigest())
  (out/f'{arm}-result.json').write_text(json.dumps(result,indent=2))
  print(json.dumps(dict(seed=seed,arm=arm,final_dev_nll=evals['none']['nll'])),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
 train(args.seed,('multi',) if args.smoke else ARMS,2 if args.smoke else 256,args.smoke)
