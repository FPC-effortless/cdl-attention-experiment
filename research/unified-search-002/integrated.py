"""COG whole-model search. Shared byte/symbol embedding, shared layers/decoder.
Exact native/temporal source, presence-aware protected update port, roster-free
factorized address adaptation. No separate parser, oracle, or task-selected core.
"""
import copy,math,sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from model import NativeLM,CogLayer,D,hard,scan,delta,count
from variants import fresh,tensor_bytes
sys.path.insert(0,str(Path(__file__).parent/'original014'))
from hold_reset import exact_hold_reset
VOCAB=288
# 3 backbone patterns x 2 bank semantics x 2 address algorithms, + reference/control recipes.
ARMS={f'{p}_{b}_{a}':(p,b,a,'random_full') for p in ('NN','SS','BB') for b in ('native','presence') for a in ('native','factor')}
ARMS.update({'GQA':('A0','native','native','random_full'),
 'SS_native_native_mixed':('SS','native','native','random_mixed'),
 'SS_presence_factor_mixed':('SS','presence','factor','random_mixed'),
 'SS_native_native_fixed':('SS','native','native','fixed_full'),
 'SS_presence_factor_fixed':('SS','presence','factor','fixed_full')})
class IntegratedLayer(CogLayer):
    def configure(self,presence,factor,seed):
        self.presence=presence;self.factor=factor
        if factor:
            g=torch.Generator().manual_seed(seed)
            self.slot_keys=nn.Parameter(torch.randn(8,2,D,generator=g)/math.sqrt(D))
            self.lex_role=nn.Linear(D,1)
            nn.init.zeros_(self.lex_role.weight);nn.init.zeros_(self.lex_role.bias)
            self.factor_gain=nn.Parameter(torch.zeros(()))
    def factor_scores(self,emb_history):
        # UNIFIED-014's per-component max then sum over identity components,
        # adapted to learned slot prototypes, not supplied entity names.
        z=F.normalize(emb_history,dim=-1);keys=F.normalize(self.slot_keys,dim=-1)
        sim=15*torch.einsum('btd,knd->btkn',z,keys)
        sim=sim+self.lex_role(z)[...,None]
        t=sim.size(1)-4
        return torch.stack([sim[:,i:i+t] for i in range(5)],-1).amax(-1).sum(-1)
    def forward(self,h,ax,payload_map,state=None,side=None,emb_history=None):
        typ,key,val=ax
        if self.factor:key=key+self.factor_gain*self.factor_scores(emb_history)
        role=typ.softmax(-1);addr=key.softmax(-1);a=hard(addr);rp=hard(role);w=rp[...,1:2]+rp[...,3:4];rd=rp[...,2:3];gate=w*a
        u=self.norm(h);z,g=self.inp(u).chunk(2,-1);g=.5+.49*g.sigmoid();g=g.unflatten(-1,(D//2,2)).mean(-1,keepdim=True).expand(*g.shape[:-1],D//2,2).flatten(-2);r,rs=scan(z.tanh(),g,None if state is None else state[0]);s=h.new_zeros(h.size(0),32,8) if state is None else state[1];slot,bg=s[:,:8],s[:,8:];payload=payload_map(val.softmax(-1)).tanh();v=payload*rp[...,1:2];zz=v.unsqueeze(-2).expand(-1,-1,8,-1);aa=(1-gate).unsqueeze(-1).expand_as(zz)
        if self.presence:
            ps=h.new_zeros(h.size(0),8,1) if state is None else state[2]
            # Explicit presence independent of value; retain exact UNIFIED-014
            # gather-forward / hierarchical surrogate-gradient transition.
            old=torch.cat((slot,ps),-1);proposal=torch.cat((zz,rp[...,1:2,None].expand(-1,-1,8,1)),-1)
            allseq,ns9=exact_hold_reset(proposal,(1-gate).unsqueeze(-1),old)
            seq=allseq[...,:8]*allseq[...,8:];ns=ns9[...,:8].contiguous();npres=ns9[...,8:].contiguous()
        elif not self.training:
            b,t=h.shape[:2];idx=((torch.arange(t,device=h.device)[None,:,None]+1)*gate.long()).cummax(1).values;src=torch.cat([slot[:,None],zz],1);seq=src.gather(1,idx[...,None].expand(-1,-1,-1,8));ns=seq[:,-1].clone()
        else:
            seq,ns=scan(zz.flatten(-2),aa.flatten(-2),slot.flatten(-2));seq=seq.unflatten(-1,(8,8));ns=ns.unflatten(-1,(8,8))
        msg=(a.unsqueeze(-1)*seq).sum(-2);k=F.normalize(self.address(u)[...,8:],dim=-1);q=F.normalize(self.readaddr(u)[...,8:],dim=-1);vbg=self.payload(u).tanh();gb=self.write(u).squeeze(-1).sigmoid().square()*rp[...,0];mb,nb=delta(k,vbg,q,gb,bg);msg=msg*(rd+w)+mb*rp[...,0:1];out=r+self.combine(u+r).sigmoid()*self.direct(msg)
        if side is not None:out=out+side
        h=h+self.out(out);h=h+self.ff(self.norm2(h));st=(rs,torch.cat((ns,nb),1))
        if self.presence:st=(*st,npres)
        return h,st
class WholeCOG(nn.Module):
    def __init__(self,arm,seed):
        super().__init__();self.arm=arm;self.pattern,self.bank,self.addressing,self.regime=ARMS[arm];self.seed=seed
        torch.manual_seed(seed)
        self.parent=NativeLM('A0') if self.pattern=='A0' else fresh(self.pattern,seed)
        m=self.parent
        # Match total parameters after adding 32 shared symbol embeddings and optional prototypes.
        trim=8+(9 if self.addressing=='factor' else 0)
        widths=[]
        for i,l in enumerate(m.core.blocks):
            old=l.ff;w=old[0].out_features-trim;widths.append(w)
            new=nn.Sequential(nn.Linear(D,w),nn.GELU(),nn.Linear(w,D))
            with torch.no_grad():
                new[0].weight.copy_(old[0].weight[:w]);new[0].bias.copy_(old[0].bias[:w]);new[2].weight.copy_(old[2].weight[:,:w]);new[2].bias.copy_(old[2].bias)
            l.ff=new
            if self.pattern!='A0' and (self.bank=='presence' or self.addressing=='factor'):
                l.__class__=IntegratedLayer;l.configure(self.bank=='presence',self.addressing=='factor',seed+731+i)
        m.ff=tuple(widths)
        old=m.core.emb;g=torch.Generator().manual_seed(seed+59901);e=nn.Embedding(VOCAB,D)
        with torch.no_grad():e.weight[:256].copy_(old.weight);e.weight[256:].copy_(torch.randn(VOCAB-256,D,generator=g)*.02)
        m.core.emb=e
    def forward(self,x,state=None,carry=False,features=False):
        m=self.parent;prior=None if state is None else state[1]
        h,last,ax=m.local(x,m.core.emb,prior);ss=[];sd=[]
        eh=None
        if self.addressing=='factor':
            prev=torch.zeros(len(x),4,dtype=x.dtype,device=x.device) if prior is None else prior
            eh=m.core.emb(torch.cat((prev,x),1))
        for i,l in enumerate(m.core.blocks):
            st=None if state is None else state[0][i];extra=None;stside=None
            if m.with_side and m.sides[i] is not None:extra,stside=m.sides[i](h,None if state is None else state[2][i])
            if m.kinds[i]=='A':h,s=l(h,st,extra)
            elif isinstance(l,IntegratedLayer):h,s=l(h,ax,m.payload_map[i],st,extra,eh)
            else:h,s=l(h,ax,m.payload_map[i],st,extra)
            ss.append(s);sd.append(stside)
        h=m.core.norm(h);z=F.linear(h,m.core.emb.weight);s=(ss,last,sd) if m.with_side else (ss,last)
        if features:return z,s,h
        return (z,s) if carry else z
