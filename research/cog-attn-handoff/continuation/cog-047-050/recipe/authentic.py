"""Evaluate a typed DST subset with actual MultiWOZ dialogue text."""
import json
import re
import time
from collections import Counter
from pathlib import Path

import torch
from torch import nn

from state_store import World

ROOT=Path(__file__).resolve().parent
SLOT="hotel-area"
VALUES=("", "centre", "east", "north", "south", "west", "dontcare")
ACTION=("hold",*VALUES)
HISTORY=80
torch.set_num_threads(1)


def examples(split):
    path=ROOT/"multiwoz"/split/"dialogues_001.json"
    data=json.loads(path.read_text())
    for dialog in data:
        if "hotel" not in dialog["services"]:
            continue
        prefix=[]
        previous=""
        for turn in dialog["turns"]:
            tokens=re.findall(r"[a-z]+|\d+|[^\w\s]",turn["utterance"].lower())
            if turn["speaker"]=="SYSTEM":
                prefix+=["<system>",*tokens]
                continue
            prefix+=["<user>",*tokens]
            cumulative={k:v for frame in turn["frames"] for k,v in frame.get("state",{}).get("slot_values",{}).items()}
            slot=cumulative.get(SLOT,[""])
            if len(slot)!=1 or slot[0] not in VALUES:
                # Keep text context, and exclude multi-valued or unsupported labels.
                yield dialog["dialogue_id"],turn["turn_id"],None,None,cumulative
                previous=None
                continue
            new=slot[0]
            # A known value after an unsupported label is a fresh set action.
            action="hold" if previous==new else new
            yield dialog["dialogue_id"],turn["turn_id"],prefix[-HISTORY:],action,cumulative
            previous=new


class TextModel(nn.Module):
    def __init__(self,mode,vocab):
        super().__init__()
        self.mode=mode
        self.embed=nn.Embedding(vocab,48,padding_idx=0)
        self.pos=nn.Embedding(HISTORY+1,48)
        if mode=="local":
            self.encoder=nn.GRU(48,48,batch_first=True)
        else:
            self.qkv=nn.Linear(48,144)
            self.project=nn.Linear(48,48)
        self.fc=nn.Sequential(nn.Linear(48,64),nn.GELU(),nn.Linear(64,len(ACTION)))

    def forward(self,x):
        e=self.embed(x)+self.pos(torch.arange(x.shape[1]))
        if self.mode=="local":
            h=self.encoder(e)[0][:,-1]
        else:
            q,k,v=self.qkv(e).chunk(3,-1)
            mask=x[:,None,None,:]!=0
            attention=nn.functional.scaled_dot_product_attention(q[:,None,-1:],k[:,None],v[:,None],attn_mask=mask)
            h=e[:,-1]+self.project(attention[:,0,0])
        return self.fc(h)


def vector(raw,vocab):
    return torch.tensor([[0]*(HISTORY-len(item[2]))+[vocab.get(t,1) for t in item[2]]+[2]
                         for item in raw],dtype=torch.long)


def store_replay(raw):
    current=None
    world=None
    n=correct=0
    previous={}
    for session,turn,_,_,target in raw:
        if session!=current:
            if world:
                world.db.close()
            current=session
            world=World()
            previous={}
        diff={k:target.get(k) for k in set(target)|set(previous) if previous.get(k)!=target.get(k)}
        assert world.commit(turn,world.epoch(),diff)=="committed"
        actual=world.state()
        correct+=actual==target
        previous=actual
        n+=1
    if world:
        world.db.close()
    return dict(cases=n,exact=correct)


def run(seed,arm,out):
    mode="local" if arm=="candidate" else "attention"
    all_raw={split:list(examples(split)) for split in ("train","dev","test")}
    raw={k:[t for t in v if t[2] is not None] for k,v in all_raw.items()}
    train=raw["train"]
    freq=Counter(t for item in train for t in item[2])
    vocab={t:i+3 for i,t in enumerate(sorted(t for t,n in freq.items() if n>=2))}
    x=vector(train,vocab)
    y=torch.tensor([ACTION.index(t[3]) for t in train])
    torch.manual_seed(seed)
    model=TextModel(mode,len(vocab)+3)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003)
    rng=torch.Generator().manual_seed(seed+1024)
    weight=(torch.bincount(y,minlength=len(ACTION)).float()+1).pow(-.5)
    start=time.perf_counter_ns()
    for _ in range(400):
        index=torch.randint(len(x),(48,),generator=rng)
        logits=model(x[index])
        loss=nn.functional.cross_entropy(logits,y[index],weight=weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    fit_ns=time.perf_counter_ns()-start
    torch.save(model.state_dict(),out/f"{mode}.pt")
    result={}
    model.eval()
    for split in ("dev","test"):
        x=vector(raw[split],vocab)
        start=time.perf_counter_ns()
        with torch.no_grad():
            output=torch.cat([model(batch).argmax(1) for batch in x.split(64)]).tolist()
        infer_ns=time.perf_counter_ns()-start
        store=World()
        rows=[]
        for item,label in zip(raw[split],output,strict=True):
            session,turn,_,truth,full=item
            action=ACTION[label]
            if action!="hold":
                assert store.commit(f"{session}:{turn}",0,{session:action})=="committed"
            predicted=store.state().get(session,"")
            rows.append(dict(session=session,turn=turn,action=action,truth=truth,
                             predicted=predicted,state=full.get(SLOT,[""])[0]))
        store.db.close()
        result[split]=dict(predictions=rows,oracle_replay=store_replay(all_raw[split]),infer_ns=infer_ns,
                           exact=sum(r["predicted"]==r["state"] for r in rows),
                           update_correct=sum(r["action"]==r["truth"]!="hold" for r in rows),
                           updated=sum(r["truth"]!="hold" for r in rows),
                           false_updates=sum(r["truth"]=="hold" and r["action"]!="hold" for r in rows))
    return dict(seed=seed,mode=mode,train_examples=len(train),fit_ns=fit_ns,result=result)
