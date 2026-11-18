import csv
import json
import os
from pathlib import Path
from time import perf_counter_ns

import numpy as np
from pilot import bind, execute, learn_trace
from diagnostic import infer

INITIAL = tuple(range(16)) + (None,)
PAD = 255
SWAP = learn_trace(((8,3,None),(8,3,8),(3,3,8),(3,8,8),(3,8,None)))
ROTATE = infer(((8,3,5,None),(8,3,5,8),(3,3,5,8),(3,5,5,8),(3,5,8,8),(3,5,8,None)))
WEIGHTS = np.left_shift(np.uint32(1), np.arange(32,dtype=np.uint32))


def norm(x):
    return x / np.linalg.norm(x,axis=-1,keepdims=True)


def codes(x):
    return (x[...,12:44]>=0).astype(np.uint32) @ WEIGHTS


def apply(packet):
    op,a,b,c = map(int,packet)
    if not (0<=op<4 and 4<=a<20):
        return INITIAL, False
    a -= 4
    if op==0:
        if not (20<=b<255 and c==PAD):
            return INITIAL, False
        state=list(INITIAL)
        state[a]=1000+b-20
        return tuple(state), True
    if not (4<=b<20):
        return INITIAL, False
    b-=4
    if op in (1,2):
        if c!=PAD or a==b:
            return INITIAL, False
        body=((a,b),) if op==1 else bind(SWAP,(a,b,16))
    else:
        if not (4<=c<20) or len({a,b,c-4})!=3:
            return INITIAL, False
        body=bind(ROTATE,(a,b,c-4,16))
    return execute(INITIAL,body), True


def expected(packet):
    op,a,b,c=map(int,packet)
    state=list(INITIAL)
    a-=4
    if op==0:
        state[a]=1000+b-20
    elif op==1:
        state[a]=INITIAL[b-4]
    elif op==2:
        state[a],state[b-4]=INITIAL[b-4],INITIAL[a]
    else:
        state[a],state[b-4],state[c-4]=INITIAL[b-4],INITIAL[c-4],INITIAL[a]
    return tuple(state)


def recall(keys, signatures, query, arm):
    if arm=="full":
        return np.argmax(query @ keys.T,axis=1)
    result=[]
    for cue,signature in zip(query,codes(query)):
        distances=np.bitwise_count(signatures ^ signature)
        subset=np.argsort(distances,kind="stable")[:32]
        result.append(int(subset[np.argmax(keys[subset] @ cue)]))
    return np.asarray(result)


def batch(keys, signatures, queries, arm):
    results=[]
    for query in queries:
        packet=recall(keys,signatures,query,arm)
        state,valid=apply(packet)
        results.append((tuple(packet),state,valid))
    return results


def main():
    out=Path(os.environ["TRAINLAB_OUTPUT_DIR"])
    with (out/"scores.csv").open("w") as sf, (out/"timing.csv").open("w") as tf:
        scores=csv.writer(sf,lineterminator="\n")
        scores.writerow("seed noise arm episode op packet_exact state_exact rejected wrong_commit gold decoded".split())
        timing=csv.writer(tf,lineterminator="\n")
        timing.writerow("seed noise block position arm ns".split())
        for seed in (101,211,307):
            rng=np.random.default_rng(seed)
            keys=norm(rng.normal(size=(256,64)))
            packets=[]
            for episode in range(256):
                op=episode%4
                a,b,c=rng.choice(16,3,replace=False)+4
                packets.append((op,a,rng.integers(20,255) if op==0 else b,c if op==3 else PAD))
            packets=np.asarray(packets)
            for packet in packets:
                assert apply(packet)==(expected(packet),True)
            # Exhaust single-field symbol substitutions against the declared type grammar.
            for packet in packets[:4]:
                for field in range(4):
                    for symbol in range(256):
                        changed=packet.copy()
                        changed[field]=symbol
                        state,valid=apply(changed)
                        assert state == (expected(changed) if valid else INITIAL)
                        assert state[-1] is None
            signatures=codes(keys)
            noise=rng.normal(size=(256,4,64))
            for scale in (0.,.09,.18):
                queries=norm(keys[packets]+scale*noise)
                for arm in ("full","filtered"):
                    results=batch(keys,signatures,queries,arm)
                    for episode,(packet,state,valid) in enumerate(results):
                        exact=state==expected(packets[episode])
                        scores.writerow((seed,scale,arm,episode,int(packets[episode,0]),
                                         int(packet==tuple(packets[episode])),int(exact),
                                         int(not valid),int(valid and not exact),
                                         json.dumps(packets[episode].tolist()),json.dumps([int(x) for x in packet])))
                for block in range(12):
                    order=("full","filtered") if block%2==0 else ("filtered","full")
                    for position,arm in enumerate(order):
                        start=perf_counter_ns()
                        batch(keys,signatures,queries,arm)
                        timing.writerow((seed,scale,block,position,arm,perf_counter_ns()-start))
    (out/"metrics.json").write_text(json.dumps({"panels":9}))


if __name__=="__main__":
    main()
