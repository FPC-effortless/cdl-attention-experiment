import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import resource
import time

import numpy as np
import torch
from torch.nn import functional as F
from data import train_offsets
from model import Config, LM


def fit(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    cfg=Config(args.arm,args.scale)
    model=LM(cfg)
    parent=None
    if args.resume:
        parent=torch.load(args.resume,map_location='cpu',weights_only=True)
        if parent['config']!=asdict(cfg) or parent['seed']!=args.seed or args.scale!='scale':
            raise ValueError('continuation requires the identical scale, arm and seed')
        model.load_state_dict(parent['state'])
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01,foreach=False)
    steps={'compact':256,'scale':128}[args.scale]
    if parent:
        steps=896
    tokens,offsets=train_offsets(args.data,args.seed+(1000 if parent else 0),steps)
    lr=.001 if args.scale=='compact' else .0005
    if parent:
        lr=.0003
    warmup=24 if parent else 12
    curve=[]
    started=time.perf_counter()
    for i,starts in enumerate(offsets):
        seq=torch.tensor(np.array(tokens[starts[:,None]+np.arange(129)]),dtype=torch.long)
        gain=min(1.,(i+1)/warmup)*(.2+.8*(1+math.cos(math.pi*i/steps))/2)
        opt.param_groups[0]['lr']=lr*gain
        logp,_=model(seq[:,:-1])
        loss=F.nll_loss(logp.flatten(0,1),seq[:,1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
        opt.step()
        curve.append([i+1,float(loss.detach()),grad])
        if i%32==0:
            print(args.arm,args.scale,args.seed,i+1,round(curve[-1][1],3),round(time.perf_counter()-started,1),flush=True)
    train_s=time.perf_counter()-started
    args.out.mkdir(exist_ok=True,parents=True)
    name=f'{args.scale}-{args.arm}-{args.seed}'
    if parent:
        name+='-continued'
    ckpt=args.out/(name+'.pt')
    torch.save(dict(config=asdict(cfg),seed=args.seed,state=model.state_dict()),ckpt)
    summary=dict(name=name,config=asdict(cfg),torch=torch.__version__,seed=args.seed,steps=steps,
                 tokens=steps*512,total_tokens=(steps+(128 if parent else 0))*512,parent_sha256=hashlib.file_digest(args.resume.open('rb'),'sha256').hexdigest() if parent else None,windows=len(np.unique(offsets)),offsets_sha256=hashlib.sha256(offsets.tobytes()).hexdigest(),
                 train_s=train_s,tokens_per_s=steps*512/train_s,parameters=sum(p.numel() for p in model.parameters()),
                 train_curve=curve,checkpoint_sha256=hashlib.file_digest(ckpt.open('rb'),'sha256').hexdigest(),
                 rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,threads=args.threads)
    (args.out/(name+'.train.json')).write_text(json.dumps(summary)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--arm',required=True)
    p.add_argument('--scale',choices=['compact','scale'],required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--resume',type=Path)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    fit(p.parse_args())
