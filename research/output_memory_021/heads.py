"""021 output heads: untrained, read-only conditioning, matched decoder family."""
import hashlib
import math
import torch
from torch import nn
from torch.nn import functional as F

BYTE=256
EOS,PAD,MASK=256,257,258


def _time(t):
    if not math.isfinite(float(t)) or not 0 <= t <= 1:
        raise ValueError('finite time in [0,1] required')


def streams(ids, seed, device):
    if len(set(ids)) != len(ids): raise ValueError('unique example IDs required')
    return [torch.Generator(device=device).manual_seed(int.from_bytes(hashlib.sha256(f'{seed}:{i}'.encode()).digest()[:8],'big')%(2**63-1)) for i in ids]


class Block(nn.Module):
    def __init__(self,width,heads):
        super().__init__();self.h=heads
        self.norm1=nn.LayerNorm(width);self.qkv=nn.Linear(width,3*width);self.proj=nn.Linear(width,width)
        self.norm2=nn.LayerNorm(width);self.ff=nn.Sequential(nn.Linear(width,width*4),nn.GELU(),nn.Linear(width*4,width))
    def forward(self,x,causal=False,cache=None,incremental=False):
        b,l,d=x.shape
        q,k,v=self.qkv(self.norm1(x)).chunk(3,-1)
        q,k,v=[z.reshape(b,l,self.h,d//self.h).transpose(1,2) for z in (q,k,v)]
        if incremental and cache is not None:
            k=torch.cat((cache[0],k),2);v=torch.cat((cache[1],v),2)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=causal and not incremental)
        x=x+self.proj(y.transpose(1,2).reshape(b,l,d));x=x+self.ff(self.norm2(x))
        return x,(k,v) if incremental else None


class OutputHead(nn.Module):
    """Use same config/initial weights for paired fits; AR time=0, diffusion time>0."""
    def __init__(self,condition_dim,width=64,layers=2,heads=4,max_length=64):
        super().__init__()
        if width%heads:raise ValueError('width divisible by heads')
        self.max_length=max_length;self.condition_dim=condition_dim
        self.embed=nn.Embedding(259,width);self.pos=nn.Embedding(max_length,width)
        self.condition=nn.Linear(condition_dim,width);self.time=nn.Linear(1,width)
        self.blocks=nn.ModuleList([Block(width,heads) for _ in range(layers)])
        self.norm=nn.LayerNorm(width);self.output=nn.Linear(width,258)
    def _input(self,x,condition,t,offset=0):
        _time(t)
        if x.ndim!=2 or offset+x.shape[1]>self.max_length:raise ValueError('output capacity exceeded')
        if condition.shape!=(x.shape[0],self.condition_dim) or not torch.isfinite(condition).all():raise ValueError('invalid condition')
        return self.embed(x)+self.pos(torch.arange(offset,offset+x.shape[1],device=x.device))[None]+self.condition(condition)[:,None]+self.time(torch.full((x.shape[0],1),t,device=x.device,dtype=self.embed.weight.dtype))[:,None]
    def forward(self,x,condition,t=0.,causal=False):
        h=self._input(x,condition,t)
        for block in self.blocks:h,_=block(h,causal)
        return self.output(self.norm(h))
    def ar_step(self,x,condition,cache=None):
        if x.shape[1]!=1:raise ValueError('one input per cached step')
        offset=0 if cache is None else cache[0][0].shape[2]
        h=self._input(x,condition,0.,offset);new=[]
        for i,block in enumerate(self.blocks):
            h,c=block(h,True,None if cache is None else cache[i],True);new.append(c)
        return self.output(self.norm(h)),new


def ar_inputs(clean):
    return torch.cat((torch.full_like(clean[:,:1],MASK),clean[:,:-1]),1)


def masked_loss(head,clean,condition,t,generator):
    _time(t)
    if t==0:raise ValueError('nonzero training time')
    noisy=torch.where(torch.rand(clean.shape,device=clean.device,generator=generator)<t,MASK,clean)
    logits=head(noisy,condition,t)
    ce=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),clean.reshape(-1),reduction='none').reshape_as(clean)
    return (ce*(noisy==MASK)).mean()/t


def _prepare(head,condition,length,ids,seed,active):
    if head.training:raise ValueError('eval mode required')
    if not isinstance(length,int) or not 1<=length<=head.max_length:raise ValueError('invalid output length')
    c=condition.detach().clone();b=c.shape[0]
    if len(ids)!=b:raise ValueError('ID batch mismatch')
    if active is None:active=torch.ones(b,dtype=torch.bool,device=c.device)
    if active.dtype!=torch.bool or active.shape!=(b,):raise ValueError('boolean active batch required')
    return c,active,streams(ids,seed,c.device)


def _sample(logits,gen):
    if not torch.isfinite(logits).all():raise ValueError('nonfinite logits')
    return torch.multinomial(logits.softmax(-1).reshape(-1,258),1,generator=gen).flatten()


@torch.no_grad()
def generate_diffusion(head,condition,length,steps,ids,seed=0,active=None):
    if not isinstance(steps,int) or steps<1:raise ValueError('positive integer steps')
    c,active,rng=_prepare(head,condition,length,ids,seed,active)
    x=torch.full((c.shape[0],length),MASK,device=c.device,dtype=torch.long);x[~active]=PAD
    for k in range(steps,0,-1):
        logits=head(x,c.clone(),k/steps)
        for i in range(x.shape[0]):
            if active[i]:
                candidate=_sample(logits[i],rng[i])
                reveal=torch.rand((length,),device=c.device,generator=rng[i])<1/k
                x[i]=torch.where((x[i]==MASK)&reveal,candidate,x[i])
    if (x==MASK).any():raise RuntimeError('unresolved mask')
    return x,{'head_calls':steps,'position_evaluations':steps*length*len(ids),'kind':'absorbing-mask'}


@torch.no_grad()
def generate_ar(head,condition,length,ids,seed=0,active=None,greedy=False):
    c,active,rng=_prepare(head,condition,length,ids,seed,active)
    token=torch.full((c.shape[0],1),MASK,device=c.device,dtype=torch.long)
    cache=None;alive=active.clone();calls=0;active_positions=0
    result=torch.full((c.shape[0],length),PAD,device=c.device,dtype=torch.long)
    for position in range(length):
        if not alive.any():break
        # Dense batch cache retains finished lanes. Their computation is explicitly
        # charged as padded work, but they consume no RNG and produce no further text.
        logits,cache=head.ar_step(token,c.clone(),cache);token=torch.full_like(token,PAD)
        calls+=1;active_positions+=int(alive.sum())
        for i in range(token.shape[0]):
            if alive[i]:
                if not torch.isfinite(logits[i]).all():raise ValueError('nonfinite logits')
                value=logits[i,0].argmax() if greedy else _sample(logits[i],rng[i])[0]
                token[i,0]=value;result[i,position]=value
                if int(value) in (EOS,PAD):alive[i]=False
    evaluated=calls*len(ids)
    return result,{'head_calls':calls,'position_evaluations':evaluated,
                   'active_position_evaluations':active_positions,
                   'padded_position_evaluations':evaluated-active_positions,
                   'kind':'cached-ar','termination':'first-EOS-or-malformed-PAD'}


def decode_task(raw):
    out=[]
    for value in raw.tolist():
        if value==EOS:return bytes(out),'terminated'
        if not 0<=value<256:return bytes(out),'malformed'
        out.append(value)
    return bytes(out),'unterminated'
