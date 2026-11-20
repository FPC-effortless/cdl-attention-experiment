"""Matched independently trained controllers with exact single-block KV reuse."""
import math
import time
import torch
from torch import nn

torch.set_num_threads(1)
DIMS=(4,3,3)
LENGTHS=(128,512,2048,8192,32768)
WIDTH=32
VOCAB=32
QUERY=31


def data(rng,n,length):
    truth=torch.stack([torch.randint(size,(n,),generator=rng) for size in DIMS],1)
    same=truth[:,1]==truth[:,2]
    truth[same,2]=(truth[same,2]+1)%3
    fields=truth+torch.tensor([0,4,7])
    # The boundary exposes which three field tokens belong to the current
    # frame. The archive contains both earlier field IDs and other tokens.
    archive=torch.randint(30,(n,length-4),generator=rng)
    x=torch.cat((archive,fields,torch.full((n,1),QUERY)),dim=1)
    return x,truth


class Local(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb=nn.Embedding(VOCAB,WIDTH)
        self.transform=nn.Sequential(nn.Linear(WIDTH*4,64),nn.GELU(),nn.Linear(64,10))

    def forward(self,x):
        return self.transform(self.emb(x[:,-4:]).flatten(1))


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb=nn.Embedding(VOCAB,WIDTH)
        self.position=nn.Embedding(5,WIDTH)
        self.query=nn.Linear(WIDTH,WIDTH)
        self.key=nn.Linear(WIDTH,WIDTH)
        self.value=nn.Linear(WIDTH,WIDTH)
        self.bias=nn.Embedding(5,4)
        with torch.no_grad():
            self.bias.weight.zero_()
            self.bias.weight[:, :3].fill_(-12.)
            for h in range(3):
                self.bias.weight[h,h]=3.
        self.head=nn.Sequential(nn.Linear(WIDTH,64),nn.GELU(),nn.Linear(64,10))

    def tokens(self,x,archive):
        position=torch.full((x.shape[1],),4,dtype=torch.long,device=x.device)
        if not archive:
            position[-4:]=torch.arange(4,device=x.device)
        return self.emb(x)+self.position(position)

    def cache(self,x):
        h=self.tokens(x,True)
        k,v=self.key(h),self.value(h)
        tail=k.new_empty((k.shape[0],4,k.shape[-1]))
        return torch.cat((k,tail),dim=1),torch.cat((v,tail),dim=1)

    def read(self,k,v,q):
        batch,n,_=k.shape
        q=self.query(q).reshape(batch,1,4,8).transpose(1,2)
        key=k.reshape(batch,n,4,8).transpose(1,2)
        val=v.reshape(batch,n,4,8).transpose(1,2)
        pos=torch.cat((torch.full((n-4,),4,dtype=torch.long,device=k.device),
                       torch.arange(4,device=k.device)))
        bias=self.bias(pos).T[None,:,None,:]
        h=nn.functional.scaled_dot_product_attention(q,key,val,attn_mask=bias)
        return self.head(h.flatten(1))

    def forward(self,x):
        h=self.tokens(x,False)
        return self.read(self.key(h),self.value(h),h[:,-1:])

    def cached(self,current,cache):
        h=self.tokens(current,False)
        if cache is None:
            return self.read(self.key(h),self.value(h),h[:,-1:])
        k,v=cache
        k[:,-4:]=self.key(h)
        v[:,-4:]=self.value(h)
        return self.read(k,v,h[:,-1:])


def fit(model,seed,scoped=False):
    rng=torch.Generator().manual_seed(seed+751)
    rng.manual_seed(seed+751)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.005)
    start=time.perf_counter_ns()
    for _ in range(480):
        x,y=data(rng,32,128)
        logits=model(x[:,-4:] if scoped else x)
        loss=sum(nn.functional.cross_entropy(part,y[:,i])
                 for i,part in enumerate(torch.split(logits,DIMS,1)))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return time.perf_counter_ns()-start


def decode(logits):
    return tuple(int(r.argmax(-1).item()) for r in torch.split(logits,DIMS,1))
