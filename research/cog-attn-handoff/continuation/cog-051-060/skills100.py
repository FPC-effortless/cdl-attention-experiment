"""Finite-domain, context-labelled continual acquisition with actual upstream learning components."""
import hashlib
import json
from pathlib import Path
import random
import sys
import numpy as np
import torch
from torch import nn
D=Path('/workspace/cog-next-private')
sys.path.insert(0,str(D))
import bdh,neuraxon

def run(seed):
    rng=np.random.default_rng(seed);random.seed(seed);torch.manual_seed(seed)
    m=bdh.BDH(bdh.BDHConfig(n_layer=2,n_embd=16,n_head=2,mlp_internal_dim_multiplier=2,vocab_size=930))
    opt=torch.optim.AdamW(m.parameters(),lr=.006)
    enc=nn.Embedding(930,16);linear=nn.Linear(16,8)
    optim=torch.optim.AdamW([*enc.parameters(),*linear.parameters()],lr=.03)
    memory={};syn={};targets=[];matrix=[]
    params=neuraxon.NetworkParameters()
    def predict(x):
        return linear(enc(x).sum(1))
    def tokens(ids,k):
        return torch.tensor(np.stack([np.full(len(ids),k),100+ids,108+8*k+ids],axis=1))
    def synread(k,x,time):
        pairs,last=syn[k,x]
        # Native fast/slow synaptic weights decay during inactivity. The readout uses these two channels.
        return np.argmax([0 if s.is_silent else s.w_fast*(1-1/params.tau_fast)**(time-last)+s.w_slow*(1-1/params.tau_slow)**(time-last) for s in pairs])
    for k in range(100):
        xs=np.arange(8)
        if k%4==0: ys=(xs+rng.integers(1,8))%8
        if k%4==1: ys=(xs*int(rng.choice([3,5,7]))+rng.integers(8))%8
        if k%4==2: ys=rng.permutation(8)
        if k%4==3: ys=xs^rng.integers(1,8)
        targets.extend(ys.tolist())
        x=tokens(xs,k);y=torch.tensor(ys)
        for step in range(12):
            loss=nn.functional.cross_entropy(m(x)[0][:,-1,:8],y)
            opt.zero_grad();loss.backward();opt.step()
            loss=nn.functional.cross_entropy(predict(x),y)
            optim.zero_grad();loss.backward();optim.step()
            for j,v in enumerate(ys):
                memory[k,j]=v
                if (k,j) not in syn:syn[k,j]=([neuraxon.Synapse(j,z,params) for z in range(8)],0)
                entries,_=syn[k,j]
                for z,s in enumerate(entries):
                    s.update(1,1 if z==v else -1,{'dopamine':1.0},1.)
                syn[k,j]=entries,k*12+step
        test=torch.cat([tokens(xs,t) for t in range(k+1)])
        with torch.no_grad():
            a=m(test)[0][:,-1,:8].argmax(1).numpy();b=predict(test).argmax(1).numpy()
        c=np.array([synread(t,j,k*12+11) for t in range(k+1) for j in xs])
        d=np.array([memory[t,j] for t in range(k+1) for j in xs])
        matrix.append([(v==targets).reshape(-1,8).sum(1).tolist() for v in (a,b,c,d)])
    return dict(arms=['upstream_bdh','neural_online','neuraxon_synapse_bank','verified_exact'],matrix=matrix,keys=len(memory))
if __name__=='__main__':
    torch.set_num_threads(1)
    out={str(s):run(s) for s in (5101,5107,5113)}
    (Path(__file__).resolve().parent/'results/skills100.json').write_text(json.dumps(out))
