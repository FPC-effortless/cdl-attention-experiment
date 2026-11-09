import hashlib,json,time,zipfile
from pathlib import Path
import torch
from regulated import Regulated,base,State,ROOT
OUT=Path(__file__).resolve().parent
archive=ROOT/'artifacts/csm-original-scope-reconstructed-20261109/CSM-GRAPH-001-verified-project.zip'
raw=zipfile.ZipFile(archive).read('CSM-GRAPH-001/csm-foundation/experiments/csm_foundation_001/data/input.txt')
assert hashlib.sha256(raw).hexdigest()=='86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed'
train=torch.tensor(list(raw[:1003854]));dev=torch.tensor(list(raw[1003854:1003854+1025])).unsqueeze(0)
torch.set_num_threads(1)
def detach(s):
 c=s.core
 return State(base.RecordState(c.values.detach(),c.links.detach(),c.cursor.detach(),c.history.detach(),c.position),s.regulation.detach())
def score(m,reset=False):
 with torch.no_grad():
  s=m.init_state(1,'cpu');loss=0
  for t in range(1024):
   if reset:s.regulation=torch.zeros_like(s.regulation)
   y,s,_=m.step(dev[:,t],s);loss+=torch.nn.functional.cross_entropy(y,dev[:,t+1]).item()
  return loss/1024
rows=[]
for seed in (11,23,37):
 g=torch.Generator().manual_seed(seed+1000);starts=torch.randint(0,len(train)-2049,(4,),generator=g)
 data=torch.stack([train[s:s+2049] for s in starts])
 for variant in ('instant','single','multi'):
  torch.manual_seed(seed);m=Regulated(base.Config(vocab_size=256),variant).float()
  opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01)
  state=m.init_state(4,'cpu');start=time.perf_counter()
  for update in range(32):
   opt.zero_grad();ys=[]
   for t in range(update*64,(update+1)*64):
    y,state,_=m.step(data[:,t],state);ys.append(y)
   loss=torch.nn.functional.cross_entropy(torch.stack(ys,1).flatten(0,1),data[:,update*64+1:(update+1)*64+1].flatten())
   loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();state=detach(state)
  seconds=time.perf_counter()-start;nll=score(m);reset_nll=score(m,True)
  path=OUT/f'checkpoint-{seed}-{variant}.pt'
  torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),starts=starts,seed=seed,variant=variant,updates=32,state=state),path)
  reloaded=Regulated(base.Config(vocab_size=256),variant).float();reloaded.load_state_dict(torch.load(path,weights_only=False)['model'])
  replay=score(reloaded);assert replay==nll
  row=dict(seed=seed,variant=variant,parameters=sum(p.numel() for p in m.parameters()),dev_nll=nll,trace_reset_nll=reset_nll,replay_nll=replay,train_seconds=seconds,checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
  rows.append(row);print(json.dumps(row),flush=True)
  (OUT/'pilot_results.json').write_text(json.dumps(dict(rows=rows,scope='Exploratory exposed-dev pilot; 32 updates, truncated gradients at 64; no test access or superiority claim'),indent=2)+'\n')
