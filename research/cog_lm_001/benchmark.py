"""Single-stream cached decode including memory access and token selection, FP32 CPU."""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path
import numpy as np
import torch

from data import corpus
from evaluate import restore


def tensor_bytes(s):
    if s is None:
        return 0
    if isinstance(s,torch.Tensor):
        return s.untyped_storage().nbytes()
    if isinstance(s,(list,tuple)):
        return sum(tensor_bytes(x) for x in s)
    return 0


def episode_bytes(states):
    if states is None:
        return 0
    seen=set()
    def deep(x):
        if id(x) in seen:
            return 0
        seen.add(id(x))
        n=sys.getsizeof(x)
        if isinstance(x,dict):
            n+=sum(deep(k)+deep(v) for k,v in x.items())
        elif isinstance(x,(list,tuple)):
            n+=sum(deep(v) for v in x)
        elif hasattr(x,'__dict__'):
            n+=deep(vars(x))
        return n
    return deep(states)


@torch.inference_mode()
def run(a):
    m=restore(a.weights)
    arr,offsets=corpus(a.data,'test')
    start=int(offsets[np.flatnonzero(np.diff(offsets)>1200)[0]])
    records=[]
    for length in (128,512,1024):
        x=torch.tensor(arr[start:start+length].astype('int64'))[None,:]
        for repeat in range(4):
            t0=time.perf_counter()
            p,state=m(x)
            if getattr(a,'compact_mamba',False):
                if m.c.arm!='mamba2':
                    raise ValueError('compaction control requires Mamba-2')
                layers,bank,episodes=state
                state=([type(c)(c.conv_state.clone(),c.ssm_state.clone()) for c in layers],bank,episodes)
            prefill=time.perf_counter()-t0
            history=[]
            t1=time.perf_counter()
            for _ in range(32):
                token=p[:,-1].argmax(-1,keepdim=True)
                history.append(token.item())
                p,state=m(token,state)
            decode=time.perf_counter()-t1
            if repeat:
                records.append(dict(length=length,repeat=repeat,prefill_s=prefill,decode_s=decode,
                    generated_ids=history,tensor_cache_bytes=tensor_bytes(state[:2]),
                    episode_heap_estimate=episode_bytes(state[2]),
                    episodes_entries=len(state[2][0].table) if state[2] else 0))
    suffix='.compact-bench.json' if getattr(a,'compact_mamba',False) else '.bench.json'
    (a.out/(a.weights.stem+suffix)).write_text(json.dumps(records)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('weights','data','out'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--compact-mamba',action='store_true',help='materialize only tensors needed for the next recurrent step after prefill')
    a=p.parse_args();torch.set_num_threads(2);run(a)
