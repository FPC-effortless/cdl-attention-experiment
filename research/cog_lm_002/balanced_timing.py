"""Rotate and reverse four warmed models through identical 1024-token requests."""
import argparse
import json
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer
from runtime import Runtime
from data import corpus


@torch.inference_mode()
def run(a):
    torch.set_num_threads(2)
    models={}
    for name in ('cog_lookup','cog_revision','cog_revision_off','mamba_lookup'):
        stem='joint_'+name.removesuffix('_off')
        models[name]=Runtime.load(a.joint/(stem+'.base.pt'),a.joint/(stem+'.pt'),name.endswith('_off'))
    tok=Tokenizer.from_file(str(a.data/'tokenizer.json'))
    arr,offs=corpus(a.data,'test')
    start=next(int(lo)+32 for lo,hi in zip(offs,offs[1:]) if hi-lo>17000)
    prompt=tok.decode(arr[start:start+1024].tolist())
    results=[]
    names=list(models)
    for rep in range(9):
        order=names if rep<5 else names[::-1]
        shift=(rep-1)%4
        for name in order[shift:]+order[:shift]:
            begin=time.perf_counter()
            x=torch.tensor([tok.encode(prompt).ids])
            t=time.perf_counter()
            pred,state=models[name].advance(x,last_only=True)
            decode_start=time.perf_counter()
            ids=[]
            for _ in range(32):
                token=pred[:,-1].argmax(-1,keepdim=True)
                ids.append(token.item())
                pred,state=models[name].advance(token,state,last_only=True)
            tok.decode(ids)
            end=time.perf_counter()
            if rep:
                results.append(dict(name=name,rep=rep,prefill_s=decode_start-t,decode_s=end-decode_start,
                                    request_s=end-begin,ids=ids))
    (a.out/'balanced_timing.json').write_text(json.dumps(results)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('joint','data','out'):
        p.add_argument('--'+k,type=Path,required=True)
    run(p.parse_args())
