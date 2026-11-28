import torch,json,pathlib
from hold_grad import ExactForward,grad
from hierarchical import hierarchical
P=pathlib.Path(__file__).parent

def exact_hold_reset(z,a,s):
 y,_=hierarchical(z,a,s);correct=torch.where(a==0,z,y);idx=torch.arange(1,z.shape[1]+1,device=z.device).reshape(1,-1,*([1]*(z.ndim-2)));idx=torch.where(a!=1,idx,0).cummax(1).values;exact=torch.gather(torch.cat([s[:,None],correct],1),1,idx.expand_as(y));out=ExactForward.apply(exact,y);return out,out[:,-1]
if __name__=='__main__':
 from scan import sequential
 torch.set_num_threads(1);records=[]
 for d in (32,192,320):
  for L in (32,128):
   for seed in (173,271,907):
    g=torch.Generator().manual_seed(1313000+d+L+seed);z=torch.randn(2,L,d,generator=g);a=torch.randn(2,L,d,generator=g);s=torch.randn(2,d,generator=g);u=torch.rand(2,L,d,generator=g);hold=u<.25;reset=u>.85;w=torch.randn(z.shape,generator=g)
    for objective in ('all','hold_only'):
     wt=w if objective=='all' else w*hold;r,vals=grad(exact_hold_reset,z,a,s,hold,reset,wt);_,ref=grad(sequential,z,a,s,hold,reset,wt);r.update(d=d,L=L,seed=seed,objective=objective,reset_bits_exact=int((vals[0][reset].view(torch.int32)==z[reset].view(torch.int32)).sum()),max_differences={k:float((q-v).abs().max()) for k,q,v in zip(('output','proposal_grad','logit_grad','initial_grad','gate_grad'),vals,ref)});records.append(r)
 z=torch.tensor([[[4.,-5.,8.],[3.,7.,-8.],[-0.,0.,2.],[1.,2.,3.]]]);a=torch.tensor([[[1.,1.,1.],[1.,1.,1.],[0.,0.,0.],[1.,1.,1.]]]);s=torch.tensor([[-0.,0.,-1.]]);y,_=exact_hold_reset(z,a,s);prev=torch.cat((s[:,None],y[:,:-1]),1);mask=a==1;reset=a==0;signed=dict(hold_exact=int((y[mask].view(torch.int32)==prev[mask].view(torch.int32)).sum()),hold_n=int(mask.sum()),reset_bits_exact=int((y[reset].view(torch.int32)==z[reset].view(torch.int32)).sum()),reset_n=int(reset.sum()));(P/'hold_reset.json').write_text(json.dumps(dict(panels=records,signed_zero_stress=signed),indent=2));print(signed)
 for objective in ('all','hold_only'):
  p=[x for x in records if x['objective']==objective];print(objective,{k:sum(x[k] for x in p) for k in ('hold_n','hold_exact','reset_n','reset_bits_exact','hold_proposal_nonzero','hold_gate_logit_nonzero')},'max gate grad err',max(x['max_differences']['gate_grad'] for x in p))
