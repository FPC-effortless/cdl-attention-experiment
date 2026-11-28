import torch,json,math
from pathlib import Path
from retention import retained_delta,reference,learnable_log_retention
from model import delta
torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
rows=[]
for seed in (9001,9011,9029):
 for dt in (torch.float32,torch.float64):
  torch.manual_seed(seed);k=torch.nn.functional.normalize(torch.randn(3,79,24,dtype=dt),dim=-1);q=torch.nn.functional.normalize(torch.randn_like(k),dim=-1);v=torch.randn(3,79,8,dtype=dt);g=torch.rand(3,79,dtype=dt);g[:,::7]=0;raw=torch.randn_like(g);s=torch.randn(3,24,8,dtype=dt)
  logf=learnable_log_retention(g,raw);a,b=retained_delta(k,v,q,g,logf,s);c,d=reference(k,v,q,g,logf,s);tol=1e-4 if dt==torch.float32 else 1e-10
  err=max(float((a-c).abs().max()),float((b-d).abs().max()));assert err<tol,err
  aa,bb=retained_delta(k,v,q,g,torch.zeros_like(g),s);cc,dd=delta(k,v,q,g,s);ne=max(float((aa-cc).abs().max()),float((bb-dd).abs().max()));assert ne<tol,ne
  y1,st=retained_delta(k[:,:23],v[:,:23],q[:,:23],g[:,:23],logf[:,:23],s);y2,st2=retained_delta(k[:,23:],v[:,23:],q[:,23:],g[:,23:],logf[:,23:],st);ce=max(float((torch.cat([y1,y2],1)-a).abs().max()),float((st2-b).abs().max()));assert ce<tol,ce
  no=torch.zeros_like(g);_,hold=retained_delta(k,v,q,no,learnable_log_retention(no,raw),s);assert torch.equal(hold,s)
  def grads(fun):
   kk,vv,qq,gg,rr,ss=[x.detach().clone().requires_grad_() for x in (k,v,q,g,raw,s)];yy,zz=fun(kk,vv,qq,gg,learnable_log_retention(gg,rr),ss);return torch.autograd.grad(yy.square().mean()+zz.square().mean(),(kk,vv,qq,gg,rr,ss))
  ga,gb=grads(retained_delta),grads(reference);ge=max(float((x-y).abs().max()) for x,y in zip(ga,gb));assert ge<tol,ge;assert float(ga[4].abs().sum())>0
  rows.append(dict(seed=seed,dtype=str(dt),sequential_max_error=err,parent_no_decay_max_error=ne,chunk_max_error=ce,gradient_max_error=ge,exact_no_write_hold=True))
Path(__file__).with_name('kernel_verification.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
