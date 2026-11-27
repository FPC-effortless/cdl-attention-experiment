import torch
from torch import nn
import torch.nn.functional as F
import edge

class NoPredictiveHistory(edge.Edge):
    """Identical weights/parameter shapes to parent; predictive feature is instantaneous."""
    def __init__(self,mode='cog'):
        super().__init__(mode)
    def init(self,b):return torch.zeros(b,2,8)
    def step(self,x,h):
        a,b,_,d=self.coeff(x); h=a*h+b
        return self.emit(h,d,x),h
    def forward(self,x):
        a,b,_,d=self.coeff(x)
        return self.emit(edge.scan(a,b),d,x)

class Compact(nn.Module):
    """Protected bank plus current causal features, no predictive feature network."""
    def __init__(self):
        super().__init__();self.register_buffer('ids',torch.eye(2));self.enc=nn.Sequential(nn.Linear(10,20),nn.SiLU(),nn.Linear(20,16));self.head=nn.Sequential(nn.Linear(24,24),nn.SiLU(),nn.Linear(24,6))
    def coeff(self,x):
        z=torch.cat((x.unsqueeze(-2).expand(*x.shape[:-1],2,8),self.ids.expand(*x.shape[:-1],2,2)),-1)
        g,u=self.enc(z).chunk(2,-1);g=g.sigmoid()*x[...,2:4,None]
        return 1-g,g*u.tanh()
    def emit(self,h,x):
        q=self.head(torch.cat((h.flatten(-2),x),-1));return x[...,6:7]+q[...,:3],F.softplus(q[...,3:])+.015
    def init(self,b):return torch.zeros(b,2,8)
    def step(self,x,h):
        a,b=self.coeff(x);h=a*h+b;return self.emit(h,x),h
    def forward(self,x):
        a,b=self.coeff(x);return self.emit(edge.scan(a,b),x)

ARMS=('cog','protected-instant','dense-instant','protected-compact','static')
parent_make=edge.make
def make(name):
    if name in ('cog','static'):return parent_make(name)
    if name=='protected-instant':return NoPredictiveHistory('cog')
    if name=='dense-instant':return NoPredictiveHistory('dense')
    if name=='protected-compact':return Compact()
    raise ValueError(name)

def checks(m,x,name):
    with torch.no_grad():
        m.eval();batch=torch.cat(m(x),-1);mid=x.shape[1]//2
        perturbed=x.clone();perturbed[:,mid:]=0
        prefix=float((torch.cat(m(perturbed),-1)[:,:mid]-batch[:,:mid]).abs().max());assert prefix<2e-5
        if name=='static':return dict(protected_checks=0,step_delta=None,prefix_delta=prefix,restart_exact=None)
        s=m.init(len(x));out=[];checks=0
        for j in range(x.shape[1]):
            old=s[0] if isinstance(s,tuple) else s
            o,s=m.step(x[:,j],s);new=s[0] if isinstance(s,tuple) else s
            if name!='dense-instant':
                quiet=~x[:,j,2:4].bool();assert torch.equal(old[quiet],new[quiet]);checks+=int(quiet.sum())*8
            out.append(torch.cat(o,-1))
            if j==mid-1:saved=tuple(z.clone() for z in s) if isinstance(s,tuple) else s.clone()
        seq=torch.stack(out,1);delta=float((seq-batch).abs().max());assert delta<2e-5
        s=saved;tail=[]
        for j in range(mid,x.shape[1]):
            o,s=m.step(x[:,j],s);tail.append(torch.cat(o,-1))
        assert torch.equal(torch.stack(tail,1),seq[:,mid:])
        return dict(protected_checks=checks,step_delta=delta,prefix_delta=prefix,restart_exact=True)

if __name__=='__main__':
    import numpy as np
    torch.set_num_threads(1)
    x,t,s=edge.inputs(np.random.default_rng(1).normal(size=(2,96,2)).astype('float32'),np.random.default_rng(2),.15,True)
    torch.manual_seed(171);orig=make('cog')
    for n in ARMS:
        torch.manual_seed(171);m=make(n)
        if n in ('cog','protected-instant','dense-instant'):assert all(torch.equal(v,m.state_dict()[k]) for k,v in orig.state_dict().items())
        y,z=m(x);loss=(.5*((y[:,:t.shape[1]]-t)/z[:,:t.shape[1]]).square()+z[:,:t.shape[1]].log())[s].mean();loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        print(n,sum(p.numel() for p in m.parameters()),checks(m,x,n))
