"""Resource-matched online mean-read comparison with a loopback verifier."""
import json
import multiprocessing as mp
from multiprocessing.connection import Client,Listener
from pathlib import Path
import resource
from time import perf_counter_ns
import numpy as np
import torch
from torch.nn.functional import scaled_dot_product_attention

def verify(conn):
    with Listener(('127.0.0.1',0)) as listener:
        conn.send(listener.address)
        with listener.accept() as link:
            epochs={}
            while True:
                msg=link.recv()
                if msg is None:break
                group,write=msg
                epochs[group]=epochs.get(group,0)+write
                link.send(epochs[group])

def trial(seq,seed,arm):
    torch.set_num_threads(1)
    q1,q2=mp.Pipe()
    child=mp.Process(target=verify,args=(q2,));child.start()
    net=Client(q1.recv())
    rng=np.random.default_rng(seed)
    raw=rng.integers(0,2,(32,seq//32),dtype=np.uint8)
    trace=[(int(rng.integers(32)),int(rng.integers(seq//32)),i%8==0) for i in range(512)]
    # Counterfactual oracle is prepared outside the timed region for each arm.
    ref=raw.copy();gold=[]
    for g,j,change in trace:
        if change:ref[g,j]^=1
        gold.append(float(ref[g].mean()))
    del ref
    t=perf_counter_ns()
    cached=arm!='scoped_sdpa'
    if cached: sums=raw.sum(axis=1,dtype=np.int64)
    else:
        values=torch.from_numpy(raw).float().reshape(32,1,-1,1)
        keys=torch.zeros_like(values);q=torch.zeros(1,1,1,1)
    model_bytes=raw.nbytes+sums.nbytes if cached else (keys.numel()+values.numel())*4
    build=perf_counter_ns()-t
    epochs=[0]*32
    latencies=[];errors=0
    for (g,j,change),answer in zip(trace,gold):
        t=perf_counter_ns()
        net.send((g,int(change)));v=net.recv()
        assert v==epochs[g]+int(change);epochs[g]=v
        if cached:
            if change:
                old=int(raw[g,j]);raw[g,j]=1-old;sums[g]+=1-2*old
            value=float(sums[g])/raw.shape[1]
        else:
            if change: values[g,0,j,0]=1-values[g,0,j,0]
            value=float(scaled_dot_product_attention(q,keys[g:g+1],values[g:g+1]))
        latencies.append(perf_counter_ns()-t);errors+=abs(value-answer)>1e-5
    net.send(None);net.close();child.join()
    return dict(size=seq,seed=seed,arm=arm,build_ns=build,online_ns=sum(latencies),median_ns=float(np.median(latencies)),
                score_ns=build+12*sum(latencies),requests=len(trace),wrong=int(errors),payload_bytes=model_bytes)

if __name__=='__main__':
    path=Path(__file__).resolve().parent/'results/online-cost.json'
    results=[]
    for seed in (5101,5107,5113):
        for size in (128,2048,32768,131072,1048576):
            arms=np.random.default_rng(seed+size).permutation(['index','scoped_sdpa','scoped_cached'])
            for arm in arms:results.append(trial(size,seed,str(arm)))
    path.write_text(json.dumps(results))
