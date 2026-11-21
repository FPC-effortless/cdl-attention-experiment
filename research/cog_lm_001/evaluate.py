"""Frozen text, transfer and lexical-mixture controls; per-document receipts."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from data import eval_offsets
from model import restore, successor_ids


@torch.no_grad()
def score(model,root,split,mode,alphas=()):
    tokens,rows=eval_offsets(root,split)
    stats=[]
    for start in range(0,len(rows),4):
        batch=rows[start:start+4]
        seq=torch.tensor(np.array(tokens[np.array([offset for _,offset in batch])[:,None]+np.arange(129)]),dtype=torch.long)
        x,y=seq[:,:-1],seq[:,1:]
        if mode=='reset16':
            # Erases all state including the sequence backbone, every 16 inputs.
            pred=torch.cat([model(b)[0] for b in x.split(16,1)],1)
        else:
            pred,_=model(x,intervention=mode)
        ids=successor_ids(x,model.c.capacity)
        selected=pred.gather(-1,y.unsqueeze(-1)).squeeze(-1)
        grids=[]
        for alpha in alphas:
            # Same index as COG. Select only a single scalar on validation.
            if alpha:
                corrected=torch.where(ids>=0,torch.logaddexp(selected+np.log1p(-alpha),
                    torch.where(ids==y, np.log(alpha),-torch.inf)),selected)
            else:
                corrected=selected
            grids.append((-corrected).sum(1).tolist())
        for j,(doc,offset) in enumerate(batch):
            stats.append(dict(doc=doc,offset=offset,count=128,nll=float(-selected[j].sum()),
                              top1=int((pred[j].argmax(-1)==y[j]).sum()),hits=int((ids[j]>=0).sum()),
                              correct_copy=int((ids[j]==y[j]).sum()),mixture_nll=[g[j] for g in grids]))
    return stats


@torch.no_grad()
def evaluate(path,data,out):
    m=restore(path)
    tok=Tokenizer.from_file(str(data/'tokenizer.json'))
    controls=('full','no_slots','no_episode','no_memory','reset16') if m.c.arm=='cog' else ('full',)
    # Calibration grid fixed before reading any test/transfer scores.
    grid=[0.,.05,.1,.2,.35,.5,.7,.9] if m.c.arm in ('transformer','mamba2','local') else []
    receipts={}
    alpha=None
    for split in ('validation','test','transfer'):
        for mode in controls:
            if split=='validation' and mode!='full':
                continue
            alphas=grid if split=='validation' else ([] if alpha is None else [alpha])
            rows=score(m,data,split,mode,alphas)
            if grid and split=='validation':
                alpha=grid[int(np.argmin(np.array([r['mixture_nll'] for r in rows]).sum(0)))]
            receipts[split+'/'+mode]=rows
    samples=[]
    for p in ('The history of the city','When the rain stopped,','The customer asked'):
        x=torch.tensor([tok.encode(p).ids])
        ids=m.generate(x,48,seed=31)
        samples.append(dict(prompt=p,ids=ids,completion=tok.decode(ids)))
    out.mkdir(exist_ok=True,parents=True)
    record=dict(model=path.stem,checkpoint_sha256=hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),
                alpha=alpha,grid=grid,rows=receipts,samples=samples)
    (out/(path.stem+'.eval.json')).write_text(json.dumps(record)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for a in ('weights','data','out'):
        p.add_argument('--'+a,type=Path,required=True)
    p.add_argument('--threads',type=int,default=2)
    a=p.parse_args();torch.set_num_threads(a.threads)
    evaluate(a.weights,a.data,a.out)
