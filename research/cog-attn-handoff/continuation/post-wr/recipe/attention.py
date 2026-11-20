import re
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class Decoder(nn.Module):
    """Single causal decoder block; final summary token predicts all three factors."""
    def __init__(self,vocabulary,width=80):
        super().__init__()
        self.embedding=nn.Embedding(vocabulary+2,width)
        self.summary=vocabulary+1
        self.qkv=nn.Linear(width,width*3)
        self.project=nn.Linear(width,width)
        self.ff=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width*2),nn.GELU(),nn.Linear(width*2,width))
        self.output=nn.Linear(width,66)

    def forward(self,x,prefix=None):
        h=self.embedding(x)
        q,k,v=self.qkv(h).chunk(3,-1)
        if prefix is None:
            attended=F.scaled_dot_product_attention(q[:,None],k[:,None],v[:,None],is_causal=True)
        else:
            length=prefix[0].shape[1]
            k=torch.cat((prefix[0],k),dim=1)
            v=torch.cat((prefix[1],v),dim=1)
            mask=torch.arange(k.shape[1])[None,:]<=length+torch.arange(q.shape[1])[:,None]
            attended=F.scaled_dot_product_attention(q[:,None],k[:,None],v[:,None],attn_mask=mask)
        z=h+self.project(attended[:,0])
        return self.output(z[:,-1]+self.ff(z[:,-1])),(k,v)


class PrefixDecoder:
    def __init__(self,model):
        self.model=model
        self.cache={}
        self.hits=0

    @torch.inference_mode()
    def __call__(self,ids):
        # Cache first two tokens ("please", verb), shared even by distinct commands.
        key=tuple(ids[:2])
        if key not in self.cache:
            _,self.cache[key]=self.model(torch.tensor([key]))
        else:
            self.hits+=1
        logits,_=self.model(torch.tensor([[*ids[2:],self.model.summary]]),self.cache[key])
        return logits
