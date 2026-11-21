"""Reload every fitted checkpoint and verify independent token-level NLL and cached logits."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from data import corpus
from model import restore


@torch.inference_mode()
def check(path,raw,data):
    spec=json.loads((raw/(path.stem+'.eval.json')).read_text())
    if hashlib.file_digest(path.open('rb'),'sha256').hexdigest()!=spec['checkpoint_sha256']:
        raise ValueError('checkpoint digest changed')
    m=restore(path)
    errors=[]
    for split in ('test','transfer'):
        tokens,_=corpus(data,split)
        rows=spec['rows'][split+'/full']
        for row in (rows[0],rows[len(rows)//2],rows[-1]):
            s=row['offset']
            seq=torch.tensor(tokens[s:s+129].astype('int64'))[None,:]
            p,_=m(seq[:,:-1])
            # Compare against a fresh framework loss computation, token mean.
            loss=torch.nn.functional.nll_loss(p[0],seq[0,1:]).item()
            errors.append(abs(loss-row['nll']/128))
    seq=torch.tensor(tokens[rows[-1]['offset']:rows[-1]['offset']+128].astype('int64'))[None,:]
    parallel,_=m(seq)
    prefix,state=m(seq[:,:96])
    out=[prefix]
    for token in seq[:,96:].split(1,1):
        p,state=m(token,state)
        out.append(p)
    parallel_error=(torch.cat(out,1)-parallel).abs().max().item()
    if max(errors)>.00002 or parallel_error>.001:
        raise ValueError((path.stem,max(errors),parallel_error))
    return dict(checkpoint=spec['checkpoint_sha256'],rescored_windows=len(errors),
                max_mean_nll_difference=max(errors),max_cached_logp_difference=parallel_error)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('fits','raw','data','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    results={f.stem:check(f,a.raw,a.data) for f in sorted(a.fits.glob('*.pt'))}
    a.out.write_text(json.dumps(results,indent=2)+'\n')
