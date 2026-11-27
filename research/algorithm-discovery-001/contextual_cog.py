"""COG-ALG-DISCOVERY-001: parameter-neutral second-layer routing candidates.

Exact parent: FPC-effortless/cdl-attention-experiment, commit
99a97c1b2ff9c669f4e45356b02083d3bc29a621, model.py blob
31c3555521469425d884c700aed01b48c4fab50e.
No oracle labels, extra history, external retrieval, or attention in HH candidates.
"""
import torch
from torch import nn
from torch.nn import functional as F
from model import NativeLM, CogLayer, count, hard, scan, delta

MODES = ('local_joint', 'context_role', 'context_address', 'context_joint')

class ContextCogLayer(CogLayer):
    """Reuse the first 8 address and first 4 readaddr rows, unused in HH.

    Context comes from a preceding causal COG layer, not the future or full-history
    attention. A local-only control uses the same rows and local input instead.
    All state transitions and payloads are inherited from the parent mathematics.
    """
    def forward(self, h, ax, payload_map, state=None, side=None, *, local_h=None, mode='context_joint', routing=False):
        if mode not in MODES:
            raise ValueError(mode)
        typ,key,val=ax
        u=self.norm(h)
        ap=self.address(u); qp=self.readaddr(u)
        if mode=='local_joint':
            lu=self.norm(local_h)
            ka=F.linear(lu,self.address.weight[:8],self.address.bias[:8])
            kr=F.linear(lu,self.readaddr.weight[:4],self.readaddr.bias[:4])
        else:
            ka=ap[...,:8]; kr=qp[...,:4]
        if mode in ('local_joint','context_role','context_joint'):typ=typ+kr
        if mode in ('local_joint','context_address','context_joint'):key=key+ka
        role=typ.softmax(-1);addr=key.softmax(-1);a=hard(addr);rp=hard(role);w=rp[...,1:2]+rp[...,3:4];rd=rp[...,2:3];gate=w*a
        z,g=self.inp(u).chunk(2,-1);g=.5+.49*g.sigmoid();g=g.unflatten(-1,(128//2,2)).mean(-1,keepdim=True).expand(*g.shape[:-1],128//2,2).flatten(-2);r,rs=scan(z.tanh(),g,None if state is None else state[0]);s=h.new_zeros(h.size(0),32,8) if state is None else state[1];slot,bg=s[:,:8],s[:,8:];payload=payload_map(val.softmax(-1)).tanh();v=payload*rp[...,1:2];zz=v.unsqueeze(-2).expand(-1,-1,8,-1);aa=(1-gate).unsqueeze(-1).expand_as(zz)
        if not self.training:
            b,t=h.shape[:2];idx=((torch.arange(t,device=h.device)[None,:,None]+1)*gate.long()).cummax(1).values;src=torch.cat([slot[:,None],zz],1);seq=src.gather(1,idx[...,None].expand(-1,-1,-1,8));ns=seq[:,-1].clone()
        else:
            seq,ns=scan(zz.flatten(-2),aa.flatten(-2),slot.flatten(-2));seq=seq.unflatten(-1,(8,8));ns=ns.unflatten(-1,(8,8))
        msg=(a.unsqueeze(-1)*seq).sum(-2);k=F.normalize(ap[...,8:],dim=-1);q=F.normalize(qp[...,8:],dim=-1);vbg=self.payload(u).tanh();gb=self.write(u).squeeze(-1).sigmoid().square()*rp[...,0];mb,nb=delta(k,vbg,q,gb,bg);msg=msg*(rd+w)+mb*rp[...,0:1];out=r+self.combine(u+r).sigmoid()*self.direct(msg)
        if side is not None:out=out+side
        h=h+self.out(out);h=h+self.ff(self.norm2(h));s=(rs,torch.cat((ns,nb),1))
        return (h,s,(typ,key)) if routing else (h,s)

class ContextCOG(NativeLM):
    """Native HH parent with a single experimental change in COG layer 2."""
    def __init__(self,mode='context_joint',ff=None):
        if mode not in MODES:raise ValueError(mode)
        super().__init__('HH',ff=ff)
        self.mode=mode
        original=self.core.blocks[1]
        replacement=ContextCogLayer(self.ff[1])
        replacement.load_state_dict(original.state_dict())
        self.core.blocks[1]=replacement
        # These rows had no effect in the parent. Zeroing them gives the parent
        # function at initialization, before a new routing rule is learned.
        self.zero_routing()
    @torch.no_grad()
    def zero_routing(self):
        layer=self.core.blocks[1]
        layer.address.weight[:8].zero_(); layer.address.bias[:8].zero_()
        layer.readaddr.weight[:4].zero_(); layer.readaddr.bias[:4].zero_()
    def forward(self,x,state=None,carry=False,features=False,*,routing=False):
        h,last,ax=self.local(x,self.core.emb,None if state is None else state[1]); local_h=h
        h,s0=self.core.blocks[0](h,ax,self.payload_map[0],None if state is None else state[0][0]);
        out=self.core.blocks[1](h,ax,self.payload_map[1],None if state is None else state[0][1],local_h=local_h,mode=self.mode,routing=routing)
        h,s1=out[:2]; h=self.core.norm(h);z=F.linear(h,self.core.emb.weight);s=([s0,s1],last)
        if routing:return z,s,out[2]
        if features:return z,s,h
        return (z,s) if carry else z

def make(arm,seed):
    torch.manual_seed(seed)
    if arm in ('HH','A0'):return NativeLM(arm)
    return ContextCOG(arm)
