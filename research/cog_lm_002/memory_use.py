"""Describe learned writes and reads on intact held-out article streams."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from runtime import Runtime
from data import corpus


@torch.inference_mode()
def run(a):
    torch.set_num_threads(2)
    model=Runtime.load(a.base,a.adapter)
    bank=model.head.bank
    arr,offs=corpus(a.data,'test')
    read_ratios, write_rates, write_concentration, key_cosines=[],[],[],[]
    def read(_mod,inputs,result):
        residual=result[0]
        h=inputs[0]
        ratio=residual.norm(dim=-1) / h.norm(dim=-1).clamp_min(1e-12)
        read_ratios.extend(ratio.flatten().tolist())
    def write(_mod,_inputs,result):
        write_rates.extend(result.sigmoid().flatten().tolist())
    rh=bank.register_forward_hook(read)
    wh=bank.strength.register_forward_hook(write)
    try:
        for i in np.flatnonzero(np.diff(offs)>1024):
            start=int(offs[i])
            state=None
            for offset in range(start,start+1024,16):
                before=bank.anchors.unsqueeze(0) if state is None else state[1][0]
                x=torch.tensor(arr[offset:offset+16].astype('int64'))[None]
                _,state=model.advance(x,state,last_only=True)
                keys=state[1][0]
                corr=F.normalize(keys,dim=-1) @ F.normalize(keys,dim=-1).transpose(1,2)
                mask=torch.triu(torch.ones_like(corr,dtype=torch.bool),diagonal=1)
                key_cosines.append(float(corr[mask].mean()))
                # Actual movement in each key. The units include value mismatch as well as the gate.
                magnitude=(keys-before).norm(dim=-1)
                total=float(magnitude.sum())
                if total>0:
                    write_concentration.append(float(magnitude.max())/total)
    finally:
        rh.remove()
        wh.remove()
    def stats(v):
        return dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),p95=float(np.quantile(v,.95)))
    result=dict(articles=len(np.flatnonzero(np.diff(offs)>1024)),read_norm_over_hidden=stats(read_ratios),
        write_strength=stats(write_rates),largest_fraction_of_key_movement=stats(write_concentration),
        mean_pairwise_key_cosine=stats(key_cosines))
    (a.out/(a.adapter.stem+'.usage.json')).write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('base','adapter','data','out'):
        p.add_argument('--'+k,type=Path,required=True)
    run(p.parse_args())
