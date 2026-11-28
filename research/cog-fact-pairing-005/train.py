"""Matched minibatch composition for learning contextual fact selection."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

for name in ("cog-context-binding-004","cog-fact-diagnosis-003","cog-adaptive-repair-002","cog-balance-001","unified-search-002"):
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent/name))
from experiment import grids
from trace_fault import cases, ids
from scale import build_model
from corpus import load_corpus
from confirm_repair import evaluate_confirmation


def blocks(seed,domain,four_digits=False):
    questions=("#a00=","#b00=") if domain=="symbol" else tuple(
        f"What digit does the log give for {name}?" for name in ("Ada","Ben"))
    buckets=defaultdict(list)
    for first,second,a,b in cases(4096,seed+15917,False,domain,independent=True):
        if a>b:
            first,second,a,b=second,first,b,a
        cells=[(h+q,48+v) for h,labels in ((first,(a,b)),(second,(b,a))) for q,v in zip(questions,labels)]
        # Randomize factual-world orientation across worlds while preserving each grid.
        if hashlib.sha256(first.encode()).digest()[1]%2:
            cells=cells[2:]+cells[:2]
        buckets[a,b].append(cells)
        if four_digits:
            complement=tuple(sorted(set(range(4))-{a,b}))
            if len(buckets[a,b])>=4 and len(buckets[complement])>=4:
                block=[]
                for i in range(4):
                    for pair in ((a,b),complement):
                        cells=buckets[pair].pop(0)
                        if cells[0][1]!=48+pair[0]:
                            cells=cells[2:]+cells[:2]
                        block.append(cells)
                yield block
        elif len(buckets[a,b])==8:
            yield buckets.pop((a,b))


def batches(block,grouped):
    result=[]
    for t in range(4):
        if grouped:
            indices=[(2*t+i,c) for i in range(2) for c in range(4)]
        else:
            # Each update has one cell per world. Each world contributes both
            # query identities and both values exactly twice over four updates.
            indices=[(j,(j//2+t)%4) for j in range(8)]
        result.append([block[j][c] for j,c in indices])
    return result


def run(arm,seed,condition,corpus_dir,target,four_digits=False):
    torch.set_num_threads(1)
    model=build_model(arm,seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0008,betas=(.9,.95),weight_decay=.01)
    corpus=load_corpus(Path(corpus_dir))
    rng=np.random.default_rng(seed+311903)
    streams=[blocks(seed,domain,four_digits) for domain in ("symbol","english")]
    pool=hashlib.sha256(); replay=hashlib.sha256()
    for section in range(64):
        block=next(streams[section%2])
        # Orient a pair of worlds oppositely inside each pair so each minibatch
        # has the same target histogram in either condition.
        for i in range(0,8,2):
            if not four_digits and block[i][0][1]==block[i+1][0][1]:
                block[i+1]=block[i+1][2:]+block[i+1][:2]
        paired,mixed=batches(block,True),batches(block,False)
        assert Counter(sum(paired,[]))==Counter(sum(mixed,[]))
        for g,m in zip(paired,mixed):
            assert Counter(v for _,v in g)==Counter(v for _,v in m)
            if four_digits:
                assert sorted(Counter(v for _,v in g).values())==[2,2,2,2]
        for text,v in sorted(sum(paired,[])):
            pool.update(text.encode()+bytes([v]))
        for t,batch in enumerate(paired if condition=="grouped" else mixed):
            x=ids([text for text,_ in batch]); y=torch.tensor([v for _,v in batch])
            optimizer.zero_grad()
            F.cross_entropy(model(x)[:,-1],y).backward()
            if t==0:
                xy=torch.from_numpy(corpus.batch("train",rng,4,128))
                replay.update(xy.numpy().tobytes())
                F.cross_entropy(model(xy[:,:-1]).flatten(0,1),xy[:,1:].flatten()).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
    model.eval()
    with torch.no_grad():
        rng=np.random.default_rng(781299);loss=[]
        for _ in range(16):
            xy=torch.from_numpy(corpus.batch("dev",rng,8,128))
            loss.append(float(F.cross_entropy(model(xy[:,:-1]).flatten(0,1),xy[:,1:].flatten())/math.log(2)))
        report=dict(arm=arm,seed=seed,condition=condition,steps=256,pool_sha=pool.hexdigest(),
                    replay_sha=replay.hexdigest(),bpb=np.mean(loss),symbol=grids(model,"symbol"),
                    english=grids(model,"english"),transfer=evaluate_confirmation(model,worlds=64,batch_size=8))
    Path(target).write_text(json.dumps(report,indent=2,allow_nan=False))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("arm",choices=("C","GQA","GRU"));p.add_argument("seed",type=int)
    p.add_argument("condition",choices=("grouped","mixed"));p.add_argument("corpus");p.add_argument("out")
    p.add_argument("--four-digits",action="store_true")
    a=p.parse_args();run(a.arm,a.seed,a.condition,a.corpus,a.out,a.four_digits)
