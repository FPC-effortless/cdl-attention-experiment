"""Train extractive pointers on automatic lexical questions about real WikiText passages."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from corpus import load_corpus
from panels import _batch_bytes, _rng, _world
from pointer_control import Pointer, confirmation, natural
from repair_objective import balanced_english_world


DIGIT = re.compile(rb"(?<![A-Za-z0-9])[0-3](?![A-Za-z0-9])")
WORDS = re.compile(rb"[A-Za-z]{4,16}")
SEED = 181492


def candidates(text):
    """Yield short pieces with two distinct, uniquely attributable nearest digit/word pairs."""
    digits = list(DIGIT.finditer(text))
    for i, (first, second) in enumerate(zip(digits, digits[1:])):
        a, b = first.start(), second.start()
        if first.group() == second.group() or b-a > 120 or b-a < 6:
            continue
        start, end = max(0, a-40), min(len(text), b+41)
        part = text[start:end]
        if b"\n" in part or sum(c >= 128 for c in part) > len(part)//8:
            continue
        all_digits = [m.start() for m in re.finditer(rb"[0-9]",part)]
        result = []
        for index in (a-start, b-start):
            anchors = []
            for word in WORDS.finditer(part):
                token = word.group()
                if part.lower().count(token.lower()) != 1: continue
                center = (word.start()+word.end()-1)/2
                distances = sorted((abs(p-center), p) for p in all_digits)
                if distances[0][1] != index or distances[0][0] > 18: continue
                if distances[0][0] == distances[1][0]: continue
                anchors.append((distances[0][0], token.decode("ascii")))
            if anchors:
                result.append(min(anchors)[1])
        if len(result)==2 and result[0]!=result[1]:
            yield start,end,a-start,b-start,result


def extract(root):
    data=load_corpus(root)
    raw=data.bytes["train"]
    offsets=data.offsets["train"]
    docs=np.random.default_rng(SEED).permutation(len(offsets)-1)
    limits={"train":1536, "hold":192}
    pools={k:[] for k in limits}
    seen=set()
    for rank,doc in enumerate(docs):
        split="hold" if rank%8==0 else "train"
        if len(pools[split])>=limits[split]:continue
        start=int(offsets[doc])
        text=bytes(raw[start:int(offsets[doc+1])])
        got=0
        for lo,hi,a,b,anchors in candidates(text):
            sig=hashlib.sha256(text[lo:hi]).hexdigest()
            if sig in seen:continue
            seen.add(sig)
            pools[split].append(dict(doc=int(doc),lo=start+lo,hi=start+hi,positions=[a,b],anchors=anchors,sha256=sig))
            got+=1
            if got==2 or len(pools[split])==limits[split]:break
        if all(len(pools[k])>=n for k,n in limits.items()):break
    a,b=({x['doc'] for x in pools[k]} for k in ("train","hold"))
    assert not a & b and len(pools['hold'])>=128 and len(pools['train'])>=1000
    return pools


def auto_cases(raw, record, rng):
    text=bytes(raw[record['lo']:record['hi']])
    assert hashlib.sha256(text).hexdigest()==record['sha256']
    edited=bytearray(text)
    p1,p2=record['positions']
    assert text[p1]!=text[p2]
    which=int(rng.integers(2))
    edit_pos=record['positions'][which]
    edited[edit_pos]=int(rng.choice([v for v in b"0123" if v!=text[edit_pos] and v!=text[record['positions'][1-which]]]))
    variants=(text,bytes(edited))
    samples,locs=[],[]
    for passage in variants:
        for anchor,pos in zip(record['anchors'],record['positions']):
            # Nearest means absolute character distance to the word's center.
            samples.append(passage+b" Which digit is nearest "+anchor.encode("ascii")+b"?")
            locs.append(pos)
    return samples,locs


def assess_auto(m,raw,pool,include=None):
    rows=[]
    rng=np.random.default_rng(SEED+827)
    with torch.no_grad():
        for i in range(0,len(pool),4):
            prompts,locs=[],[]
            for record in pool[i:i+4]:
                p,l=auto_cases(raw,record,rng)
                prompts+=p; locs+=l
            answers,positions=m.select(prompts)
            for k in range(0,len(prompts),4):
                value=[prompt[pos] for prompt,pos in zip(prompts[k:k+4],locs[k:k+4])]
                pred=answers[k:k+4]
                rows.append((pred==value,pred[:2]==value[:2],positions[k:k+4]==locs[k:k+4],sum(a==b for a,b in zip(pred,value))))
    if include is not None:
        rows=[row for row,keep in zip(rows,include) if keep]
    return dict(passages=len(rows),all_four_correct=sum(v[0] for v in rows),
                original_query_pair=sum(v[1] for v in rows),all_four_positions=sum(v[2] for v in rows),
                answers_correct=sum(v[3] for v in rows),answers=4*len(rows))


def template_pairs(seed, step):
    rng=_rng(seed,2**38+step,"train")
    samples,locs=[],[]
    for _ in range(12):
        if step%4>=2:
            x,y,_,_=balanced_english_world(rng,step%2,"fact")
        else:
            x,y,_,_=_world(rng,"retrieval" if step%2==0 else "conflict")
        pos=next(k for k,(a,b) in enumerate(zip(x,y)) if a!=b)
        samples.extend((x,y)); locs.extend((pos,pos))
    return samples,locs


def run(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    m=Pointer(wide=True)
    parent=args.root/f"wide-balanced-position-{args.seed}.pt"
    m.load_state_dict(torch.load(parent,weights_only=True))
    pools=json.loads(args.manifest.read_text())
    raw=np.memmap(args.corpus/"train.bin",dtype=np.uint8,mode="r")
    initial=assess_auto(m,raw,pools["hold"])
    opt=torch.optim.AdamW(m.parameters(),lr=0.001)
    for step in range(384):
        if args.arm=="real" and step%4!=0:
            rng=_rng(args.seed,2**40+step,"train")
            samples,locs=[],[]
            for idx in rng.integers(len(pools['train']),size=6):
                p,l=auto_cases(raw,pools['train'][idx],rng)
                samples.extend(p);locs.extend(l)
        else:
            replay_step = step // 4 if step % 4 == 0 else 1000 + step
            samples,locs=template_pairs(args.seed,replay_step)
        ids=_batch_bytes(samples)
        scores=m.score(ids,torch.tensor(list(map(len,samples))))
        loss=F.cross_entropy(scores,torch.tensor(locs))
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    m.eval()
    out=args.root/"real";out.mkdir(exist_ok=True)
    stem=f"{args.arm}-{args.seed}"
    torch.save(m.state_dict(),out/(stem+".pt"))
    record=dict(seed=args.seed,arm=args.arm,replay_schedule="stratified-templates-v2",parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
                manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                initial_auto=initial,final_auto=assess_auto(m,raw,pools["hold"]),
                natural=natural(m,args.corpus),confirmation=confirmation(m))
    (out/(stem+".json")).write_text(json.dumps(record,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("local_result/pointer"))
    p.add_argument("--corpus",type=Path,default=Path("trace_corpus"))
    p.add_argument("--manifest",type=Path,default=Path("local_result/pointer/real-passages.json"))
    p.add_argument("--seed",type=int)
    p.add_argument("--arm",choices=("real","template"))
    p.add_argument("--extract",action="store_true")
    args=p.parse_args()
    if args.extract:
        pools=extract(args.corpus)
        args.manifest.write_text(json.dumps(pools,separators=(",",":"))+"\n")
        print({k:(len(v),len({r['doc'] for r in v})) for k,v in pools.items()})
    else:
        run(args)
