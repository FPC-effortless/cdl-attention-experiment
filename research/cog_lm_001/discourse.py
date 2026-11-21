"""Frozen LAMBADA continuation scoring. Store hashes and scores without corpus text."""
import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch
from tokenizers import Tokenizer
from model import restore, successor_ids, mix


@torch.no_grad()
def run(a):
    source=hashlib.sha256(a.parquet.read_bytes()).hexdigest()
    if source!='99d8a9e54cda761c6340ddd4fa6feda7f846ffb0e716afa52968dea75495eea7':
        raise ValueError('unexpected English test data')
    tok=Tokenizer.from_file(str(a.data/'tokenizer.json'))
    texts=pq.read_table(a.parquet)['text'].to_pylist()
    ordered=sorted((hashlib.sha256(text.encode()).hexdigest(),text) for text in texts)
    tasks=[]
    for key,text in ordered[:512]:
        context,word=text.rstrip().rsplit(' ',1)
        prefix=tok.encode(context).ids
        ids=tok.encode(text.rstrip()).ids
        if ids[:len(prefix)]!=prefix or len(ids)==len(prefix):
            raise ValueError('continuation boundary mismatch')
        tasks.append((key,ids,len(prefix),word.casefold() in {w.casefold().strip('.,!?;:') for w in context.split()}))
    model=restore(a.weights)
    alpha=None
    settings=a.out/(a.weights.stem+'.eval.json')
    if settings.is_file():
        alpha=json.loads(settings.read_text())['alpha']
    rows=[]
    tasks.sort(key=lambda r:len(r[1]))
    for pos in range(0,len(tasks),4):
        batch=tasks[pos:pos+4]
        size=16*((max(len(r[1]) for r in batch)+14)//16)
        x=torch.zeros(len(batch),size,dtype=torch.long)
        for j,(_,ids,_,_) in enumerate(batch):
            x[j,:len(ids)-1]=torch.tensor(ids[:-1])
        p,_=model(x)
        lookups=successor_ids(x,model.c.capacity)
        corrected=None
        if alpha is not None and alpha>0:
            gate=torch.full_like(x,float(torch.logit(torch.tensor(alpha))),dtype=torch.float32)
            corrected=mix(p,lookups,gate)
        for j,(key,ids,n,seen_word) in enumerate(batch):
            scores=p[j,n-1:len(ids)-1]
            target=torch.tensor(ids[n:])
            rows.append(dict(passage_sha256=key,count=len(target),context=n,
                        exact=bool(scores.argmax(-1).eq(target).all()),previous_word=seen_word,
                        exact_copy_available=bool(lookups[j,n-1:len(ids)-1].eq(target).all()),
                        mixture_exact=None if alpha is None else bool((corrected if corrected is not None else p)[j,n-1:len(ids)-1].argmax(-1).eq(target).all()),
                        nll=float(-scores.gather(-1,target[:,None]).sum())))
    (a.out/(a.weights.stem+'.discourse.json')).write_text(json.dumps(rows)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('weights','data','out','parquet'):
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2);run(a)
