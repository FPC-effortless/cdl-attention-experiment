import argparse,hashlib,json,math,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
from variants import make,ARMS
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'csm_longbench_003'))
from experiment import corpus

def sequence(m,x):
 s=m.init_state(len(x),'cpu');ys=[]
 for t in range(x.shape[1]-1):
  y,s,_=m.step(x[:,t],s);ys.append(y)
 return torch.stack(ys,1)

def score(m,x):
 with torch.no_grad():
  y=sequence(m,x);loss=F.cross_entropy(y.flatten(0,1),x[:,1:].flatten());pred=y.argmax(-1)
 return dict(nll=loss.item(),accuracy=(pred==x[:,1:]).float().mean().item(),space_share=(pred==32).float().mean().item(),distinct_top1=int(pred.unique().numel()),targets=pred.numel())

def audit(m,x):
 handles=[];features={}
 names={'frontend':m.core.frontend,'controller':m.core.controller,'proposal_hidden':m.core.transform[0],'proposal':m.core.transform[2]}
 def hook(name):
  def collect(mod,args,out):features.setdefault(name,[]).append(out.detach())
  return collect
 for name,layer in names.items():handles.append(layer.register_forward_hook(hook(name)))
 handles.append(m.core.output.register_forward_pre_hook(lambda mod,args:features.setdefault('readout',[]).append(args[0].detach()) and None))
 m.zero_grad();y=sequence(m,x);loss=F.cross_entropy(y.flatten(0,1),x[:,1:].flatten());loss.backward()
 for h in handles:h.remove()
 stats={}
 for k,v in features.items():
  a=torch.stack(v,1);stats[k]=dict(rms=a.square().mean().sqrt().item(),mean_coordinate_std=a.std(1,unbiased=False).mean().item(),abs_above_3=(a.abs()>3).float().mean().item(),abs_above_095=(a.abs()>.95).float().mean().item())
 gradients={}
 for k,p in m.named_parameters():
  gradients[k]=dict(norm=None if p.grad is None else p.grad.norm().item(),zero_fraction=None if p.grad is None else (p.grad==0).float().mean().item())
 m.zero_grad();return dict(activations=stats,gradients=gradients)

def main(seed,smoke=False):
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);tr,dev=corpus();x=tr[:129].unsqueeze(0);adj=tr[129:258].unsqueeze(0);dv=dev[:1025].unsqueeze(0)
 out=HERE/('smoke' if smoke else f'seed-{seed}');out.mkdir(exist_ok=True)
 for arm in (('hard',) if smoke else ARMS):
  if (out/f'{arm}.json').exists() and not smoke:continue
  torch.manual_seed(seed);m=make(arm);initial=hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in m.parameters())).hexdigest()
  opt=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=.01);curves=[dict(update=0,fit=score(m,x))];steps=2 if smoke else 256;elapsed=0
  start=0;ckpath=out/f'{arm}.pt'
  if ckpath.exists() and not smoke:
   ck=torch.load(ckpath,weights_only=True);assert ck['seed']==seed and ck['arm']==arm
   m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);torch.set_rng_state(ck['rng']);start=ck['update'];curves=ck['curves'];elapsed=ck['elapsed']
  for u in range(start,steps):
   tick=time.perf_counter();opt.zero_grad();y=sequence(m,x);loss=F.cross_entropy(y.flatten(0,1),x[:,1:].flatten());assert torch.isfinite(loss);loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1));opt.step();elapsed+=time.perf_counter()-tick
   if u+1 in (32,64,128,256) or u+1==steps:
    row=dict(update=u+1,fit=score(m,x),preclip_gradient_norm=gn);curves.append(row);print(json.dumps(dict(seed=seed,arm=arm,**row)),flush=True)
    torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state(),seed=seed,arm=arm,update=u+1,curves=curves,elapsed=elapsed),out/f'{arm}.pt')
  fit=score(m,x);r=dict(seed=seed,arm=arm,parameters=sum(p.numel() for p in m.parameters()),initial_sha256=initial,fit=fit,fit_criterion_passed=fit['nll']<.7 and fit['accuracy']>=.8,adjacent_train=score(m,adj),exposed_dev=score(m,dv),curves=curves,audit=audit(m,x),training_seconds=elapsed,training_targets=steps*128,checkpoint_sha256=hashlib.sha256((out/f'{arm}.pt').read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256((HERE/'PROTOCOL.md').read_bytes()).hexdigest(),training_excerpt_sha256=hashlib.sha256(bytes(tr[:129].tolist())).hexdigest())
  (out/f'{arm}.json').write_text(json.dumps(r,indent=2)+'\n')
  print(json.dumps(dict(seed=seed,arm=arm,fit=fit,dev=r['exposed_dev'])),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args();main(a.seed,a.smoke)
