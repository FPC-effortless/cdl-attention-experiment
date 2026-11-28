"""Stage A DeltaSeq and AltLayer (pinned source: competence.py, 7ccd1fc0...)."""
import torch
from torch import nn
from torch.nn import functional as F
from model import D,count
class DeltaSeq(nn.Module):
    """Same reference recurrence as DISCOVERY-010; not optimized Gated DeltaNet."""
    def __init__(self):
        super().__init__();self.qkv=nn.Linear(D,2*64+D);self.gates=nn.Linear(D,8);self.out=nn.Linear(D,D)
    def forward(self,x,state=None):
        b,t,d=x.shape
        q,k,v=self.qkv(x).split((64,64,128),-1)
        q=F.normalize(q.reshape(b,t,4,16).transpose(1,2),dim=-1)
        k=F.normalize(k.reshape(b,t,4,16).transpose(1,2),dim=-1)
        v=v.reshape(b,t,4,32).transpose(1,2)
        a,g=self.gates(x).transpose(1,2).chunk(2,1);a=.95+.05*a.sigmoid();g=g.sigmoid()
        s=x.new_zeros(b,4,16,32) if state is None else state;out=[]
        for Q,K,V,A,G in zip(q.split(16,2),k.split(16,2),v.split(16,2),a.split(16,2),g.split(16,2)):
            p=A.cumprod(-1);decay=p[..., :,None]/p[...,None,:];H=(K@K.transpose(-1,-2))*decay
            L=(G[...,None]*H).tril(-1);rhs=G[...,None]*(V-p[...,None]*(K@s))
            u=torch.linalg.solve_triangular(L,rhs,upper=False,unitriangular=True)
            z=p[...,None]*(Q@s)+((Q@K.transpose(-1,-2))*decay).tril()@u
            s=p[...,-1,None,None]*s+K.transpose(-1,-2)@(u*(p[...,-1,None]/p)[...,None]);out.append(z)
        return self.out(torch.cat(out,2).transpose(1,2).reshape(b,t,D)),s
class AltLayer(nn.Module):
    def __init__(self,old,seq):
        super().__init__();self.norm=old.norm;self.norm2=old.norm2;self.seq=seq
        diff=count(seq)-count(old.qkv)-count(old.out)
        width=old.ff[0].out_features-round(diff/(2*D+1))
        assert width>0
        self.ff=nn.Sequential(nn.Linear(D,width),nn.GELU(),nn.Linear(width,D))
        with torch.no_grad():
            self.ff[0].weight.copy_(old.ff[0].weight[:width]);self.ff[0].bias.copy_(old.ff[0].bias[:width]);self.ff[2].weight.copy_(old.ff[2].weight[:,:width]);self.ff[2].bias.copy_(old.ff[2].bias)
    def forward(self,h,state=None,side=None):
        z,s=self.seq(self.norm(h),state)
        if side is not None:z=z+side
        h=h+z;return h+self.ff(self.norm2(h)),s
