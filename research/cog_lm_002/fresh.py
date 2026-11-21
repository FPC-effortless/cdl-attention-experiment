"""Score a disjoint slice of the published English LAMBADA file."""
import hashlib
from pathlib import Path

import pyarrow.parquet as pq
import torch
from tokenizers import Tokenizer
from architecture import successor_ids


def lambada(system, data, parquet):
    if hashlib.sha256(parquet.read_bytes()).hexdigest() != '99d8a9e54cda761c6340ddd4fa6feda7f846ffb0e716afa52968dea75495eea7':
        raise ValueError('unexpected source')
    tok = Tokenizer.from_file(str(data/'tokenizer.json'))
    ordered = sorted((hashlib.sha256(t.encode()).hexdigest(),t) for t in pq.read_table(parquet)['text'].to_pylist())
    tasks = []
    for key, text in ordered[512:1024]:
        context, _ = text.rstrip().rsplit(' ',1)
        prefix, ids = tok.encode(context).ids, tok.encode(text.rstrip()).ids
        if ids[:len(prefix)] != prefix or len(ids)<=len(prefix):
            raise ValueError('invalid final-word boundary')
        tasks.append((key, ids, len(prefix)))
    tasks.sort(key=lambda r:len(r[1]))
    result = []
    for start in range(0,len(tasks),4):
        batch = tasks[start:start+4]
        width = max(len(ids)-1 for _,ids,_ in batch)
        width = 16*((width+15)//16)
        x = torch.zeros((len(batch),width),dtype=torch.long)
        for i,(_,ids,_) in enumerate(batch):
            x[i,:len(ids)-1] = torch.tensor(ids[:-1])
        score,_ = system.advance(x)
        ids_seen = successor_ids(x,system.model.c.capacity)
        for i,(key,ids,pos) in enumerate(batch):
            z = score[i,pos-1:len(ids)-1]
            gold = torch.tensor(ids[pos:])
            result.append(dict(passage=key,count=len(gold),exact=bool((z.argmax(-1)==gold).all()),
                available=bool((ids_seen[i,pos-1:len(ids)-1]==gold).all()),
                nll=float(-z.gather(-1,gold[:,None]).sum())))
    return result
