"""Audited replacement bridge: original TAC layer, frozen LRC, retrained CSM reader."""
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from original_lrc.bridge_data import bridge_dataset
from original_lrc.model import ContentAddressedLRC
from state_store import World
from tac_identity import IdentityFieldLayer

torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"original_csm"))
import reader
from reference028 import Op,State,commit,read,wait

FAMILIES=(("ledger","accounts","bookkeeping"),("inventory","stock","supplies"),
          ("support","tickets","helpdesk"),("dispatch","shipping","transport"))
FILL=("review","current","records","approved","today","use","for","in","the","batch")
VOCAB={w:i for i,w in enumerate(sorted(set(FILL+sum(FAMILIES,()))))}
EXTRA=("keep","use","but","idle","prefer","retain","select","discard")
for field in EXTRA:
    reader.V.setdefault(field,len(reader.V)+1)
SEEDS=(1103,1129,1153)
TRAIN=(
    "choose {p} over {q}", "ignore {q} and choose {p}", "{p} rather than {q}",
    "use {p} and ignore {q}", "over {q} use {p}", "discard {q} and retain {p}",
    "select {p} but ignore {q}", "prefer {p} over {q}", "keep {p} discard {q}",
)
HELD=("over {q} choose {p}", "ignore {q} but choose {p}", "choose {p} and discard {q}")
CHALLENGE=("rather than {q} select {p}", "over {q} prefer {p}", "ignore {q} but retain {p}")


def family_batch(seed,n,kind):
    rng=random.Random(seed)
    x,y=[],[]
    for _ in range(n):
        label=rng.randrange(4)
        group=FAMILIES[label]
        cue=rng.choice(group[:2]) if kind!="unseen" else group[2]
        words=rng.sample(FILL,4)
        idx=rng.randrange(5) if kind=="train" else 4
        words.insert(idx,cue)
        x.append([VOCAB[w] for w in words])
        y.append(label)
    return torch.tensor(x),torch.tensor(y)


class TAC(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=nn.Embedding(len(VOCAB),32)
        self.layer=IdentityFieldLayer(32,n_identities=4,identity_energy_budget=4.0)

    def input(self,x):
        return self.embedding(x).mean(1,keepdim=True)

    def affinity(self,x):
        query=nn.functional.normalize(self.layer.query_proj(self.input(x)).squeeze(1),dim=-1)
        codes=nn.functional.normalize(self.layer.identity_embeddings.weight,dim=-1)
        return query@codes.T

    def forward(self,x):
        return self.layer(self.input(x)).active_identity


def train_tac(seed):
    torch.manual_seed(seed)
    tac=TAC()
    optim=torch.optim.AdamW(tac.parameters(),lr=.01)
    start=time.perf_counter_ns()
    for step in range(240):
        batch,target=family_batch(seed+step*883,48,"train")
        affinity=tac.affinity(batch)
        loss=nn.functional.cross_entropy(affinity*16.,target)
        optim.zero_grad()
        loss.backward()
        optim.step()
    tac.eval()
    return tac,time.perf_counter_ns()-start


def csm_data(seed,n,forms):
    rng=random.Random(seed)
    support,questions,answers=[],[],[]
    for _ in range(n):
        order=rng.sample(range(8),8)
        records=[]
        for cls in order:
            for alias in reader.WORDS[cls*2:cls*2+2]:
                records.append(reader.encode(rng.choice(("file for ","entry "))+alias))
        a,b=rng.sample(range(8),2)
        q=[]
        for selected,other in ((a,b),(b,a)):
            p=reader.WORDS[order[selected]*2+rng.randrange(2)]
            reject=reader.WORDS[order[other]*2+rng.randrange(2)]
            q.append(reader.encode(rng.choice(forms).format(p=p,q=reject)))
        support.append(records)
        questions.append(q)
        answers.append([a,b])
    return np.array(support),np.array(questions),np.array(answers)


def train_csm(seed):
    model=reader.Reader(seed,True)
    start=time.perf_counter_ns()
    for step in range(1,701):
        support,query,truth=csm_data(seed+17*step,16,TRAIN)
        model.step((support,query,truth,None),step)
    return model,time.perf_counter_ns()-start


def original_csm(seed):
    path=ROOT/f"original_csm/recurrent-{seed}.npz"
    # The earlier embedding table predates the declared vocabulary extension.
    with np.load(path,allow_pickle=False) as w:
        model=reader.Reader(seed,True)
        model.E=np.pad(w["E"],((0,len(reader.V)+1-len(w["E"])),(0,0)))
        model.a=w["a"];model.b=w["b"]
    return model


def csm_pick(model,documents,query):
    memory=model.forward(documents)[0].reshape(len(documents),8,2,32).sum(2)
    cue=model.forward(query)[0]
    score=cue@memory.transpose(0,2,1)
    return score.argmax(2)


def lrc_module(seed):
    folder=ROOT/"original_lrc"
    file=folder/f"repair_{seed}.pt"
    actual=hashlib.sha256(file.read_bytes()).hexdigest()
    assert actual==json.loads((folder/f"repair_{seed}.receipt.json").read_text())["checkpoint_sha256"]
    model=ContentAddressedLRC().eval()
    model.load_state_dict(torch.load(file,weights_only=True)["model"])
    return model,actual


def held(seed,tac,csm,profile,scm_seed,lrc_seed):
    families,answer=family_batch(seed+890,256,"unseen" if profile=="unseen_family" else "held")
    docs,queries,subjects=csm_data(seed+327,256,CHALLENGE if profile=="challenge" else HELD)
    with torch.inference_mode():
        family=tac(families).tolist()
        assert family==tac.affinity(families).argmax(-1).tolist()
    chosen=csm_pick(csm,docs,queries)[:,0]
    banks=[bridge_dataset(seed+551+i*313,64) for i in range(4)]
    model,digest=lrc_module(lrc_seed)
    lrc_valid=[]
    with torch.inference_mode():
        for b in banks:
            x=model(b["records"],b["query"])
            lrc_valid.append(dict(places=x.logits.argmax(1).tolist(),hops=x.selected.tolist(),expected=b["answer"].tolist(),
                                  true_hops=b["support"].tolist()))
    gate=np.load(ROOT/f"original_csm/commit-{scm_seed}.npz",allow_pickle=False)["predicate"]
    results={}
    for mode in ("integrated","rotate_tac","rotate_lrc","rotate_csm","unresolved"):
        world=World()
        evidence=[]
        for i,where in enumerate(family):
            if i and i%64==0:
                world.advance()
            expected_bank=int(answer[i]);expected_subject=int(subjects[i,0])
            if mode=="rotate_tac":
                where=(where+1)%4
            subject=int(chosen[i])
            if mode=="rotate_csm":
                subject=(subject+1)%8
            # Per-bank documents have the same independent typed contract.
            bank=lrc_valid[where]
            dest=bank["places"][i]
            if mode=="rotate_lrc":
                dest=(dest+1)%4
            cold=[("v",1000*where+10*i+j) for j in range(8)]+[None]*4
            if mode=="unresolved":
                cold[subject]=None
            present=read(cold,subject) is not None
            allow=bool(np.dot([1,1,present,present],gate)>.5)
            visible=commit(wait(State(tuple(cold)),Op("copy",8+dest,subject),required=subject)) if allow else State(tuple(cold))
            value=read(visible.cells,8+dest)
            did_commit=False
            if value is not None and all(h>=0 for h in bank["hops"][i]):
                status=world.commit(str(i),world.epoch(),{f"place:{dest}":value})
                assert status=="committed"
                did_commit=True
            expected_dest=lrc_valid[expected_bank]["expected"][i]
            expected_value=1000*expected_bank+10*i+expected_subject
            correct=did_commit and (where,subject,dest,value)==(expected_bank,expected_subject,expected_dest,expected_value)
            evidence.append(dict(bank=where,subject=subject,place=dest,value=value,admitted=did_commit,exact=correct,
                                 target=[expected_bank,expected_subject,expected_dest,expected_value],t=i))
        world.db.close()
        results[mode]=evidence
    result_lrc=dict(held_answers=sum(sum(a==b for a,b in zip(x["places"],x["expected"],strict=True)) for x in lrc_valid),
                    held_paths=sum(sum(a==b for a,b in zip(x["hops"],x["true_hops"],strict=True)) for x in lrc_valid),
                    trials=1024,checkpoint_sha256=digest)
    return dict(profile=profile,lrc=result_lrc,results=results)


def run(seed,arm,out):
    ix=SEEDS.index(seed)
    csm_model,csm_ns=train_csm(seed) if arm=="candidate" else (original_csm((11,23,37)[ix]),0)
    tac,tac_ns=train_tac(seed)
    torch.save(tac.state_dict(),out/"tac.pt")
    np.savez(out/"csm.npz",E=csm_model.E,a=csm_model.a,b=csm_model.b)
    profiles=[held(seed,tac,csm_model,p,(11,23,37)[ix],901+ix) for p in ("supported","challenge","unseen_family")]
    return dict(seed=seed,arm=arm,profiles=profiles,training_ns={"tac":tac_ns,"csm":csm_ns})
