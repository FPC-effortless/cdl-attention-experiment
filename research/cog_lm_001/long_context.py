"""Compare last-128-token NLL with 128 versus 1024 available context, paired by article."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from evaluate import restore
from data import corpus


@torch.no_grad()
def run(a):
    model=restore(a.weights)
    arr,off=corpus(a.data,'test')
    ids=np.flatnonzero(np.diff(off)>=1025)
    rng=np.random.default_rng(411)
    ids=sorted(rng.choice(ids,min(len(ids),24),replace=False))
    rows=[]
    for i in ids:
        start=int(rng.integers(off[i],off[i+1]-1024))
        seq=torch.tensor(arr[start:start+1025].astype('int64'))[None,:]
        x,y=seq[:,:-1],seq[:,-128:]
        modes=['full','no_episode','no_slots'] if model.c.arm=='cog' else ['full']
        row=dict(doc=int(i),start=start,count=128)
        for mode in modes:
            for n in (128,1024):
                p,_=model(x[:,-n:],intervention=mode)
                row[f'{n}/{mode}']=float(-p[:,-128:].gather(-1,y[:,:,None]).sum())
        rows.append(row)
    (a.out/(a.weights.stem+'.context.json')).write_text(json.dumps(rows)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('weights','data','out'):
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2);run(a)
