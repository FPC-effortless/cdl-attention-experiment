"""COG-LM-001. All indexed successors are observed before the prediction they affect."""
from collections import OrderedDict
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from vendor.mamba2 import Mamba2, Mamba2Config, InferenceCache


@dataclass(frozen=True)
class Config:
    arm: str
    scale: str = 'compact'
    vocab: int = 2048
    slots: int = 8
    bank_dim: int = 32
    period: int = 16
    capacity: int = 256

    @property
    def width(self):
        return {'compact':128,'scale':640}[self.scale]

    @property
    def depth(self):
        return {'compact':3,'scale':6}[self.scale]


class Successors:
    """Bounded last observed successor, keyed by two consecutive token IDs."""
    def __init__(self, capacity):
        self.table = OrderedDict()
        self.tail = ()
        self.capacity = capacity

    def ingest(self, token):
        if len(self.tail)==2:
            self.table[self.tail]=token
            self.table.move_to_end(self.tail)
            if len(self.table)>self.capacity:
                self.table.popitem(last=False)
        self.tail=(*self.tail[-1:],token)
        return self.table.get(self.tail,-1)

    def apply(self,tokens):
        return [self.ingest(t) for t in tokens]


def successor_ids(x,capacity,states=None):
    states=states if states is not None else [Successors(capacity) for _ in range(len(x))]
    return torch.tensor([s.apply(row) for s,row in zip(states,x.tolist())],device=x.device)


def mix(logp,ids,gate):
    valid=ids>=0
    base=logp+torch.where(valid,F.logsigmoid(-gate),0.).unsqueeze(-1)
    ix=ids.clamp_min(0).unsqueeze(-1)
    extra=torch.where(valid,F.logsigmoid(gate),-torch.inf).unsqueeze(-1)
    return base.scatter(-1,ix,torch.logaddexp(base.gather(-1,ix),extra))


class Diagonal(nn.Module):
    """Learned affine state transition with an exact block scan."""
    def __init__(self,w):
        super().__init__()
        self.inputs=nn.Linear(w,2*w)
        self.out=nn.Linear(w,w,bias=False)

    def forward(self,x,state=None):
        f,v=self.inputs(x).chunk(2,-1)
        f=.15+.84*torch.sigmoid(f+1.)
        v=(1-f)*v.tanh()
        state=x.new_zeros(len(x),x.size(-1)) if state is None else state
        pieces=[]
        for a,b in zip(f.split(16,1),v.split(16,1)):
            prod=a.cumprod(1)
            hidden=prod*(state[:,None]+(b/prod).cumsum(1))
            pieces.append(hidden)
            state=hidden[:,-1]
        return self.out(torch.cat(pieces,1)),state


def rotate(x,offset):
    size=x.size(-1)
    if size%2:
        raise ValueError('RoPE requires an even head dimension')
    pos=torch.arange(offset,offset+x.size(2),device=x.device)
    scale=10000.**(-torch.arange(0,size,2,device=x.device)/size)
    angle=pos[:,None]*scale[None,:]
    a,b=x[...,::2],x[...,1::2]
    return torch.stack((a*angle.cos()-b*angle.sin(),b*angle.cos()+a*angle.sin()),-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self,w):
        super().__init__()
        self.qkv=nn.Linear(w,3*w,bias=False)
        self.out=nn.Linear(w,w,bias=False)

    def forward(self,x,state=None):
        b,t,w=x.shape
        q,k,v=[a.reshape(b,t,w//32,32).transpose(1,2) for a in self.qkv(x).chunk(3,-1)]
        offset=0 if state is None else state[0].size(2)
        q,k=rotate(q,offset),rotate(k,offset)
        if state is not None:
            if t!=1:
                raise ValueError('cached attention expects one new token')
            k=torch.cat([state[0],k],2)
            v=torch.cat([state[1],v],2)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=state is None)
        return self.out(y.transpose(1,2).reshape(b,t,w)),(k,v)


class Block(nn.Module):
    def __init__(self,c):
        super().__init__()
        w=c.width
        self.norm=nn.LayerNorm(w)
        if c.arm=='mamba2':
            self.mcfg=Mamba2Config(w,n_layer=c.depth,d_state=8,expand=4,headdim=32,chunk_size=16)
            self.mixer=Mamba2(self.mcfg)
            # The upstream inference reference leaves these empty for loading pretrained weights.
            with torch.no_grad():
                h=self.mcfg.nheads
                self.mixer.A_log.copy_(torch.empty(h).uniform_(1.,16.).log())
                dt=torch.empty(h).uniform_(math.log(.001),math.log(.1)).exp()
                self.mixer.dt_bias.copy_(dt+torch.log(-torch.expm1(-dt)))
                self.mixer.D.fill_(1.)
        else:
            self.mixer=Attention(w) if c.arm=='transformer' else Diagonal(w)
            wide=4*w if c.arm=='transformer' else 9*w//2
            self.ff=nn.Sequential(nn.LayerNorm(w),nn.Linear(w,wide),nn.GELU(),nn.Linear(wide,w))

    def forward(self,x,state=None):
        y,state=self.mixer(self.norm(x),state)
        x=x+y
        if hasattr(self,'ff'):
            x=x+self.ff(x)
        return x,state


class SlotBank(nn.Module):
    """Read previous blocks. Convex row and link rewrites commit every period inputs."""
    def __init__(self,c):
        super().__init__()
        self.c=c
        self.route=nn.Linear(c.width,c.slots)
        self.write=nn.Linear(c.width,2*c.slots+c.bank_dim)
        self.readout=nn.Linear(2*c.bank_dim,c.width,bias=False)

    def initial(self,x):
        c=self.c
        return (x.new_zeros(len(x),c.slots,c.bank_dim),x.new_zeros(len(x),c.slots,c.slots),
                x.new_zeros(len(x),c.width),0)

    def forward(self,h,state=None):
        rows,links,total,used=self.initial(h) if state is None else state
        output=[]; start=0
        while start<h.size(1):
            block=h[:,start:start+self.c.period-used]
            route=self.route(block).softmax(-1)
            first=route@rows
            related=(route@links)@rows
            output.append(self.readout(torch.cat([first,related],-1)))
            total=total+block.sum(1)
            used+=block.size(1)
            start+=block.size(1)
            if used==self.c.period:
                write,link,value=self.write(total/used).split([self.c.slots,self.c.slots,self.c.bank_dim],-1)
                rate=write.sigmoid().unsqueeze(-1)
                rows=rows*(1-rate)+rate*value.tanh()[:,None]
                links=links*(1-rate)+rate*link.softmax(-1)[:,None]
                total=torch.zeros_like(total)
                used=0
        return torch.cat(output,1),(rows,links,total,used)


class LM(nn.Module):
    def __init__(self,c):
        super().__init__()
        if c.arm not in ('transformer','mamba2','local','cog','slots','episode'):
            raise ValueError(c.arm)
        self.c=c
        self.emb=nn.Embedding(c.vocab,c.width)
        self.blocks=nn.ModuleList(Block(c) for _ in range(c.depth))
        self.norm=nn.LayerNorm(c.width)
        nn.init.normal_(self.emb.weight,std=.02)
        if c.arm in ('cog','slots','episode'):
            self.bank=SlotBank(c)
            self.gate=nn.Linear(c.width,1)
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias,-1.5)

    def forward(self,x,state=None,intervention='full'):
        # Full prefill or training may take an arbitrary input length. The upstream
        # Mamba SSD path is padded and its tail cache corrected via genuine step().
        if self.c.arm=='mamba2' and state is None and x.size(1)%16:
            cut=x.size(1)//16*16
            outs=[]
            if cut:
                p,state=self.forward(x[:,:cut])
                outs.append(p)
            else:
                state=([InferenceCache.alloc(len(x),b.mcfg,device=x.device) for b in self.blocks],None,None)
            for t in range(cut,x.size(1)):
                p,state=self.forward(x[:,t:t+1],state)
                outs.append(p)
            return torch.cat(outs,1),state
        h=self.emb(x)
        old,bank,episodes=state if state is not None else ([None]*len(self.blocks),None,None)
        new=[]
        for block,prev in zip(self.blocks,old):
            h,s=block(h,prev)
            new.append(s)
        h=self.norm(h)
        if self.c.arm in ('cog','slots') and intervention not in ('no_slots','no_memory'):
            r,bank=self.bank(h,bank)
            h=h+r
        logp=F.log_softmax(F.linear(h,self.emb.weight),-1)
        if self.c.arm in ('cog','episode') and intervention not in ('no_episode','no_memory'):
            if episodes is None:
                episodes=[Successors(self.c.capacity) for _ in range(len(x))]
            keys=successor_ids(x,self.c.capacity,episodes)
            logp=mix(logp,keys,self.gate(h).squeeze(-1))
        return logp,(new,bank,episodes)

    @torch.no_grad()
    def generate(self,tokens,count,seed=31,temperature=.8):
        if tokens.size(0)!=1 or tokens.size(1)<1 or count<0 or temperature<=0:
            raise ValueError('expected one nonempty prompt and positive temperature')
        p,state=self(tokens)
        gen=torch.Generator(device=tokens.device).manual_seed(seed)
        result=[]
        for _ in range(count):
            nxt=torch.multinomial((p[:,-1]/temperature).softmax(-1),1,generator=gen)
            result.append(nxt.item())
            if nxt.item()==0:
                break
            p,state=self(nxt,state)
        return result


def restore(path):
    s=torch.load(path,map_location='cpu',weights_only=True)
    model=LM(Config(**s['config']))
    model.load_state_dict(s['state'])
    return model.eval()
