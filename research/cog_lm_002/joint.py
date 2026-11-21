"""Train the same late-introduced modules with the entire backbone unfrozen."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from adapt import digest
from architecture import Readout, backbone, features, successor_ids
from data import corpus


def fit(a):
    torch.set_num_threads(a.threads)
    m = backbone(a.base).requires_grad_(True)
    if (m.c.arm == 'mamba2') != (a.arm == 'mamba_lookup'):
        raise ValueError('arm and parent must match')
    torch.manual_seed(17)
    head = Readout(m.c.width, a.arm == 'cog_revision')
    tokens, bounds = corpus(a.data,'train')
    candidates = np.concatenate([np.arange(lo,hi-128,128,dtype=np.int64) for lo,hi in zip(bounds,bounds[1:])])
    order = np.random.default_rng(20261121).choice(candidates,12288,replace=False).reshape(-1,8)
    opt = torch.optim.AdamW([{'params':m.parameters()}, {'params':head.parameters()}],foreach=False)
    parameters = [*m.parameters(), *head.parameters()]
    path = a.out / ('joint_'+a.arm)
    a.out.mkdir(parents=True,exist_ok=True)
    start = time.perf_counter()
    curve = []
    elapsed = 0.
    progress = path.with_suffix('.progress.pt')
    if progress.is_file():
        saved = torch.load(progress,map_location='cpu',weights_only=True)
        if saved['parent'] != digest(a.base) or saved['arm'] != a.arm:
            raise ValueError('saved progress belongs to another experiment')
        m.load_state_dict(saved['backbone'])
        head.load_state_dict(saved['head'])
        opt.load_state_dict(saved['optimizer'])
        curve, elapsed = saved['loss'], saved['elapsed']
        del saved
    for step in range(len(curve),len(order)):
        offsets = order[step]
        seq = torch.from_numpy(tokens[offsets[:,None]+np.arange(129)].astype('int64'))
        x, y = seq[:,:-1], seq[:,1:]
        gain = min(1.,(step+1)/32) * (.2+.4*(1+math.cos(math.pi*step/len(order))))
        for group, rate in zip(opt.param_groups,(.00015,.001)):
            group['lr'] = rate * gain
        h,_ = features(m,x)
        logp,_ = head(h,successor_ids(x,m.c.capacity),m.emb.weight)
        loss = F.nll_loss(logp.flatten(0,1),y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
        opt.step()
        curve.append(float(loss.detach()))
        if (step+1)%64 == 0 or step == 0:
            print(step+1,round(curve[-1],4),round(time.perf_counter()-start,1),flush=True)
        if a.save_progress and (step+1)%256 == 0 and step+1<len(order):
            temp = progress.with_suffix('.tmp')
            torch.save(dict(parent=digest(a.base),arm=a.arm,backbone=m.state_dict(),
                head=head.state_dict(),optimizer=opt.state_dict(),loss=curve,
                elapsed=elapsed+time.perf_counter()-start),temp)
            temp.replace(progress)
    from dataclasses import asdict
    bpath, hpath = path.with_suffix('.base.pt'), path.with_suffix('.pt')
    torch.save(dict(config=asdict(m.c),seed=17,state=m.state_dict()),bpath)
    torch.save(head.state_dict(),hpath)
    path.with_suffix('.json').write_text(json.dumps(dict(name=path.name, parent=a.base.name,
        parent_sha256=digest(a.base),adapter_sha256=digest(hpath),base_sha256=digest(bpath),
        trainable=sum(p.numel() for p in parameters),total_targets=2097152,steps=len(order),
        offsets_sha256=hashlib.sha256(order.tobytes()).hexdigest(),
        wall_s=elapsed+time.perf_counter()-start,loss=curve))+'\n')
    progress.unlink(missing_ok=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for k in ('base','data','out'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--arm',choices=('cog_lookup','cog_revision','mamba_lookup'),required=True)
    p.add_argument('--save-progress', action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--threads',type=int,default=2)
    fit(p.parse_args())
