import torch,json,pathlib
from scan import sequential
from hierarchical import hierarchical
from hold_repair import exact_hold_scan
P=pathlib.Path(__file__).parent
class ExactForward(torch.autograd.Function):
 @staticmethod
 def forward(ctx,exact,surrogate):return exact
 @staticmethod
 def backward(ctx,grad):return None,grad

def st_scan(z,a,s):
 y,_=hierarchical(z,a,s);exact,_=exact_hold_scan(z,a,s);out=ExactForward.apply(exact,y);return out,out[:,-1]
FUN={'serial':sequential,'hier':hierarchical,'gather':exact_hold_scan,'st':st_scan}

def grad(fn,z,g,s,mask,reset,weights):
 z,g,s=[q.clone().requires_grad_() for q in (z,g,s)];soft=g.sigmoid();chosen=torch.where(mask,torch.ones_like(soft),torch.where(reset,torch.zeros_like(soft),soft.detach()));a=chosen+(soft-soft.detach());a.retain_grad();y,_=fn(z,a,s);loss=(y*weights).sum()/weights.numel();zz,gg,ss,aa=torch.autograd.grad(loss,(z,g,s,a));prev=torch.cat((s[:,None],y[:,:-1]),1);r=dict(hold_n=int(mask.sum()),hold_exact=int((y[mask].view(torch.int32)==prev[mask].view(torch.int32)).sum()),reset_n=int(reset.sum()),reset_equal=int((y[reset]==z[reset]).sum()),hold_proposal_nonzero=int((zz[mask]!=0).sum()),hold_gate_logit_nonzero=int((gg[mask]!=0).sum()),finite=bool(y.isfinite().all() and zz.isfinite().all() and gg.isfinite().all()),hold_logit_grad_norm=float(gg[mask].norm()));return r,(y.detach(),zz,gg,ss,aa)
if __name__=='__main__':
 torch.set_num_threads(1);records=[]
 for d in (32,192,320):
  for L in (32,128):
   for seed in (173,271,907):
    g=torch.Generator().manual_seed(1313000+d+L+seed);z=torch.randn(2,L,d,generator=g);a=torch.randn(2,L,d,generator=g);s=torch.randn(2,d,generator=g);u=torch.rand(2,L,d,generator=g);hold=u<.25;reset=u>.85;w=torch.randn(z.shape,generator=g)
    for objective in ('all','hold_only'):
     wt=w if objective=='all' else w*hold;rs={};ref=None
     for name,fn in FUN.items():
      r,vals=grad(fn,z,a,s,hold,reset,wt)
      if ref is None:ref=vals
      r['max_differences']={k:float((q-v).abs().max()) for k,q,v in zip(('output','proposal_grad','logit_grad','initial_grad','gate_grad'),vals,ref)};rs[name]=r
     records.append(dict(d=d,L=L,seed=seed,objective=objective,methods=rs))
 # Explicit consecutive holds starting from -0, +0, nonzero: bitwise exact-value requirement.
 z=torch.tensor([[[4.,-5.,8.],[3.,7.,-8.],[-0.,0.,2.],[1.,2.,3.]]]);a=torch.tensor([[[1.,1.,1.],[1.,1.,1.],[0.,0.,0.],[1.,1.,1.]]]);s=torch.tensor([[-0.,0.,-1.]])
 signed={}
 for name,fn in FUN.items():
  y,_=fn(z,a,s);prev=torch.cat((s[:,None],y[:,:-1]),1);mask=a==1;reset=a==0;signed[name]=dict(hold_exact=int((y[mask].view(torch.int32)==prev[mask].view(torch.int32)).sum()),hold_n=int(mask.sum()),reset_bits_exact=int((y[reset].view(torch.int32)==z[reset].view(torch.int32)).sum()),reset_n=int(reset.sum()))
 (P/'hold_grad.json').write_text(json.dumps({'panels':records,'signed_zero_stress':signed},indent=2));print('panels',len(records));print('signed',signed)
 for o in ('all','hold_only'):
  r=[x['methods'] for x in records if x['objective']==o]
  for name in FUN:
   print(o,name,{k:sum(x[name][k] for x in r) for k in ('hold_n','hold_exact','hold_proposal_nonzero','hold_gate_logit_nonzero')},'max grad errors', {k:max(x[name]['max_differences'][k] for x in r) for k in ('proposal_grad','logit_grad','initial_grad','gate_grad')})
