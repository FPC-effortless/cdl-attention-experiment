"""REPAIR-009: exact HH/A0 core port; additive temporal candidate and layer-selective hybrids."""
import math, torch
from torch import nn
from torch.nn import functional as F
D=128

def count(m): return sum(p.numel() for p in m.parameters())
def hard(p):
    q=F.one_hot(p.argmax(-1),p.size(-1)).to(p.dtype)
    return q+(p-p.detach())

def scan(z,a,s=None):
    if s is None:s=torch.zeros_like(z[:,0])
    out=[]
    for zc,ac in zip(z.split(64,1),a.split(64,1)):
        A=ac; C=(1-ac)*zc; j=1
        while j<zc.shape[1]:
            A,C=torch.cat((A[:,:j],A[:,j:]*A[:,:-j]),1),torch.cat((C[:,:j],C[:,j:]+A[:,j:]*C[:,:-j]),1);j*=2
        y=A*s[:,None]+C;s=y[:,-1];out.append(y)
    return torch.cat(out,1),s

def delta(k,v,q,g,s,chunk=32):
    out=[]
    for kk,vv,qq,gg in zip(k.split(chunk,1),v.split(chunk,1),q.split(chunk,1),g.split(chunk,1)):
        gg=gg[...,None];system=(gg*(kk@kk.transpose(-1,-2))).tril(-1);rhs=gg*(vv-kk@s)
        u=torch.linalg.solve_triangular(system,rhs,upper=False,unitriangular=True)
        out.append(qq@s+(qq@kk.transpose(-1,-2)).tril()@u);s=s+kk.transpose(-1,-2)@u
    return torch.cat(out,1),s

class Local(nn.Module):
    def __init__(self):
        super().__init__();self.trunk=nn.Sequential(nn.Linear(5*D,96),nn.GELU(),nn.LayerNorm(96));self.key=nn.Linear(96,32);self.val=nn.Linear(96,32);self.typ=nn.Linear(96,4);self.keyhead=nn.Linear(32,8);self.valhead=nn.Linear(32,8);self.out=nn.Linear(64,D,bias=False);nn.init.normal_(self.out.weight,std=.01)
    def forward(self,x,emb,last=None):
        last=torch.zeros(len(x),4,dtype=x.dtype,device=x.device) if last is None else last
        ids=torch.cat((last,x),1);a=emb(ids);h=self.trunk(torch.cat([a[:,:-4],a[:,1:-3],a[:,2:-2],a[:,3:-1],a[:,4:]],-1));k=self.key(h);v=self.val(h);ty=self.typ(h);p=ty.softmax(-1);z=self.out(torch.cat([k*p[...,1:].sum(-1,keepdim=True),v*p[...,1:2]],-1))
        return a[:,4:]+z,ids[:,-4:].clone(),(ty,self.keyhead(k),self.valhead(v))

class CogLayer(nn.Module):
    def __init__(self,ff):
        super().__init__();self.norm=nn.LayerNorm(D);self.inp=nn.Linear(D,2*D);self.address=nn.Linear(D,32);self.payload=nn.Linear(D,8);self.write=nn.Linear(D,1);self.readaddr=nn.Linear(D,32);self.direct=nn.Linear(8,D);self.combine=nn.Linear(D,D);self.out=nn.Linear(D,D);self.norm2=nn.LayerNorm(D);self.ff=nn.Sequential(nn.Linear(D,ff),nn.GELU(),nn.Linear(ff,D))
    def forward(self,h,ax,payload_map,state=None,side=None):
        typ,key,val=ax;role=typ.softmax(-1);addr=key.softmax(-1);a=hard(addr);rp=hard(role);w=rp[...,1:2]+rp[...,3:4];rd=rp[...,2:3];gate=w*a
        u=self.norm(h);z,g=self.inp(u).chunk(2,-1);g=.5+.49*g.sigmoid();g=g.unflatten(-1,(D//2,2)).mean(-1,keepdim=True).expand(*g.shape[:-1],D//2,2).flatten(-2);r,rs=scan(z.tanh(),g,None if state is None else state[0]);s=h.new_zeros(h.size(0),32,8) if state is None else state[1];slot,bg=s[:,:8],s[:,8:];payload=payload_map(val.softmax(-1)).tanh();v=payload*rp[...,1:2];zz=v.unsqueeze(-2).expand(-1,-1,8,-1);aa=(1-gate).unsqueeze(-1).expand_as(zz)
        if not self.training:
            b,t=h.shape[:2];idx=((torch.arange(t,device=h.device)[None,:,None]+1)*gate.long()).cummax(1).values;src=torch.cat([slot[:,None],zz],1);seq=src.gather(1,idx[...,None].expand(-1,-1,-1,8));ns=seq[:,-1].clone()
        else:
            seq,ns=scan(zz.flatten(-2),aa.flatten(-2),slot.flatten(-2));seq=seq.unflatten(-1,(8,8));ns=ns.unflatten(-1,(8,8))
        msg=(a.unsqueeze(-1)*seq).sum(-2);k=F.normalize(self.address(u)[...,8:],dim=-1);q=F.normalize(self.readaddr(u)[...,8:],dim=-1);vbg=self.payload(u).tanh();gb=self.write(u).squeeze(-1).sigmoid().square()*rp[...,0];mb,nb=delta(k,vbg,q,gb,bg);msg=msg*(rd+w)+mb*rp[...,0:1];out=r+self.combine(u+r).sigmoid()*self.direct(msg)
        if side is not None:out=out+side
        h=h+self.out(out);h=h+self.ff(self.norm2(h));return h,(rs,torch.cat((ns,nb),1))

class GQA(nn.Module):
    def __init__(self,ff):
        super().__init__();self.norm=nn.LayerNorm(D);self.qkv=nn.Linear(D,2*D);self.out=nn.Linear(D,D);self.norm2=nn.LayerNorm(D);self.ff=nn.Sequential(nn.Linear(D,ff),nn.GELU(),nn.Linear(ff,D));self.register_buffer('freq',torch.exp(-math.log(10000)*torch.arange(D//8)/(D//8)),persistent=False)
    def forward(self,h,state=None,side=None):
        b,t,d=h.shape;q,k,v=self.qkv(self.norm(h)).split([D,D//2,D//2],-1);q=q.reshape(b,t,4,D//4).transpose(1,2);k=k.reshape(b,t,2,D//4).transpose(1,2);v=v.reshape(b,t,2,D//4).transpose(1,2);p=0 if state is None else state[0].shape[2];th=torch.arange(p,p+t,device=h.device)[:,None]*self.freq;co,si=th.cos(),th.sin()
        def rot(w):
            a,b=w.unflatten(-1,(-1,2)).unbind(-1);return torch.stack((a*co-b*si,a*si+b*co),-1).flatten(-2)
        q,k=rot(q),rot(k)
        if state is not None:k=torch.cat([state[0],k],2);v=torch.cat([state[1],v],2)
        mask=None if p==0 or t==1 else torch.arange(p+t,device=h.device)[None,:]<=torch.arange(p,p+t,device=h.device)[:,None]
        z=F.scaled_dot_product_attention(q,k.repeat_interleave(2,1),v.repeat_interleave(2,1),attn_mask=mask,is_causal=p==0 and t>1).transpose(1,2).reshape(b,t,d)
        if side is not None:z=z+side
        h=h+self.out(z);return h+self.ff(self.norm2(h)),(k,v)

class Side(nn.Module):
    """New experimental ungated-by-oracle slow and group-consistent reflection channels."""
    def __init__(self):
        super().__init__();self.norm=nn.LayerNorm(D);self.inp=nn.Linear(D,16);self.action=nn.Linear(D,3);self.axes=nn.Parameter(torch.randn(2,8));self.initial=nn.Parameter(torch.randn(8)*.02);self.proj=nn.Linear(16,D,bias=False);self.register_buffer('retention',torch.logit(torch.tensor([.75,.75,.97,.97,.997,.997,.9999,.9999])));nn.init.zeros_(self.proj.weight)
    def forward(self,h,state=None):
        u=self.norm(h);z,a=self.inp(u).chunk(2,-1);slow,ss=scan(z.tanh(),(a+self.retention).sigmoid(),None if state is None else state[0]);k=F.normalize(self.axes[0],dim=-1);w=F.normalize(self.axes[1]-(self.axes[1]@k)*k,dim=-1);v=.5*k+math.sqrt(.75)*w;axes=torch.stack([k,v]);p=hard(self.action(u).softmax(-1));eye=torch.eye(8,device=h.device,dtype=h.dtype);Hs=eye[None]-2*axes[:,:,None]*axes[:,None,:];A=torch.einsum('bti,ijk->btjk',p,torch.cat([eye[None],Hs],0));s=self.initial[None].expand(len(h),-1) if state is None else state[1];out=[]
        for block in A.split(64,1):
            j=1
            while j<block.size(1):block=torch.cat([block[:,:j],block[:,j:]@block[:,:-j]],1);j*=2
            y=(block@s[:,None,:,None]).squeeze(-1);s=y[:,-1];out.append(y)
        r=torch.cat(out,1);return self.proj(torch.cat([slow,r],-1)),(ss,s)

class Core(nn.Module):
    def __init__(self,kinds,ff):
        super().__init__();self.emb=nn.Embedding(256,D);self.norm=nn.LayerNorm(D);self.blocks=nn.ModuleList([CogLayer(f) if k=='H' else GQA(f) for k,f in zip(kinds,ff)])
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m,nn.Linear):m.weight.normal_(0,1/math.sqrt(m.in_features));m.bias.zero_() if m.bias is not None else None
                elif isinstance(m,nn.Embedding):m.weight.normal_(0,.02)

class NativeLM(nn.Module):
    def __init__(self,kind,ff=None):
        super().__init__();self.kind=kind;self.kinds={'HH':'HH','A0':'AA','U':'HH','UA':'HA','AU':'AH'}[kind];self.with_side=kind in ('U','UA','AU')
        self.ff=ff if ff is not None else {'HH':(1581,1581),'A0':(1686,1686),'U':(1560,1560),'UA':(1613,1613),'AU':(1613,1613)}[kind];self.core=Core(self.kinds,self.ff);self.local=Local()
        if 'H' in self.kinds:
            self.payload_map=nn.ModuleList([nn.Linear(8,8) if k=='H' else nn.Identity() for k in self.kinds]);self.payload_probe=nn.ModuleList([nn.Linear(8,8) if k=='H' else nn.Identity() for k in self.kinds])
            for m in list(self.payload_map)+list(self.payload_probe):
                if isinstance(m,nn.Linear):nn.init.eye_(m.weight);nn.init.zeros_(m.bias)
        if self.with_side:self.sides=nn.ModuleList([Side() for _ in range(2)])
    def forward(self,x,state=None,carry=False,features=False):
        h,last,ax=self.local(x,self.core.emb,None if state is None else state[1]);ss=[];sd=[]
        for i,l in enumerate(self.core.blocks):
            st=None if state is None else state[0][i];extra=None
            if self.with_side:extra,st_side=self.sides[i](h,None if state is None else state[2][i]);sd.append(st_side)
            if self.kinds[i]=='H':h,s=l(h,ax,self.payload_map[i],st,extra)
            else:h,s=l(h,st,extra)
            ss.append(s)
        h=self.core.norm(h);z=F.linear(h,self.core.emb.weight);s=(ss,last,sd) if self.with_side else (ss,last)
        if features:return z,s,h
        return (z,s) if carry else z
