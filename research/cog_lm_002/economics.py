"""Serial two-thread CPU timing. Timed requests include memory updates and output decoding."""
import argparse
import json
import resource
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer
from architecture import Successors
from runtime import Runtime
from benchmark import tensor_bytes, episode_bytes
from data import corpus


@torch.inference_mode()
def measure(a):
    t = time.perf_counter()
    tok = Tokenizer.from_file(str(a.data/'tokenizer.json'))
    token_load = time.perf_counter()-t
    t = time.perf_counter()
    system = Runtime.load(a.base,a.adapter,a.disabled)
    model_load = time.perf_counter()-t
    arr, offs = corpus(a.data,'test')
    start = next(int(lo)+32 for lo,hi in zip(offs,offs[1:]) if hi-lo>17000)
    times = []
    for length in (256,1024,4096,16384):
        raw = tok.decode(arr[start:start+length].tolist())
        for rep in range(4):
            t = time.perf_counter()
            ids = tok.encode(raw).ids
            x = torch.tensor([ids])
            t1 = time.perf_counter()
            p,state = system.advance(x,last_only=True)
            t2 = time.perf_counter()
            emitted = []
            for _ in range(32):
                token = p[:,-1].argmax(-1,keepdim=True)
                emitted.append(token.item())
                p,state = system.advance(token,state,last_only=True)
            decoded = tok.decode(emitted)
            t3 = time.perf_counter()
            if rep:
                times.append(dict(request_length=length, encoded=len(ids),rep=rep,encoding_s=t1-t,
                    prefill_s=t2-t1,decode_s=t3-t2,request_s=t3-t, tokens=emitted, text=decoded,
                    cache_tensor_bytes=tensor_bytes(state[:2]), table_heap_bytes=episode_bytes(state[2]),
                    table_entries=len(state[2][0].table)))
    # Same read/write schedule as production. Attribute time to the modules in a separate pass.
    elapsed = dict(bank_s=0.,lookup_s=0.)
    previous = Successors.apply
    def timed_lookup(table,tokens):
        start=time.perf_counter()
        result=previous(table,tokens)
        elapsed['lookup_s']+=time.perf_counter()-start
        return result
    bank=system.head.bank
    def before(_module,_args):
        before.start=time.perf_counter()
    def after(_module,_args,_result):
        elapsed['bank_s']+=time.perf_counter()-before.start
    handles = [bank.register_forward_pre_hook(before),bank.register_forward_hook(after)] if bank else []
    try:
        Successors.apply=timed_lookup
        _,state=system.advance(torch.tensor([arr[start:start+1024].tolist()]),last_only=True)
        for tkn in arr[start+1024:start+1056]:
            _,state=system.advance(torch.tensor([[int(tkn)]]),state,last_only=True)
    finally:
        Successors.apply=previous
        for handle in handles:
            handle.remove()
    result=dict(model_load_s=model_load,tokenizer_load_s=token_load,profile_1024_plus_32=elapsed,
                rss_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                weight_bytes=a.base.stat().st_size+a.adapter.stat().st_size, measurements=times)
    name=a.adapter.stem+('_off' if a.disabled else '')
    (a.out/(name+'.cost.json')).write_text(json.dumps(result)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('base','adapter','data','out'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--disabled',action='store_true')
    a=p.parse_args()
    torch.set_num_threads(2)
    measure(a)
