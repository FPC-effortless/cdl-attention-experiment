"""C slow-state model with an optional learned English clause parser and exact store.

Inference buffers at most 160 bytes up to a sentence boundary. A learned
clause classifier decides write/read/ignore; learned token labels identify
the subject and value. Accepted keys are at most 16 bytes, and values are
single bytes. The 32-entry FIFO store handles replacements in place.
"""
import argparse
import hashlib
import json
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from corpus import load_corpus
from panels import NAMES, PROSE

BOUNDARY = b".!?"
TOKENS = re.compile(rb"[A-Za-z]+|[0-9]+|[^\w\s]", re.ASCII)
CAPACITY = 32
CLAUSE_BYTES = 160
KEY_BYTES = 16
IGNORE, TRUST, WEAK, READ = range(4)
OTHER, SUBJECT, VALUE = range(3)


def tokens(raw):
    return TOKENS.findall(raw)


def shape(word):
    return (word.isalpha(), word.isupper(), word.istitle(), word.isdigit(),
            not word.isalnum(), min(len(word), 16)/16)


class Tagger(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.words = nn.Embedding(len(vocab) + 2, 20, padding_idx=0)
        self.ctx = nn.GRU(26, 24, bidirectional=True, batch_first=True)
        self.role = nn.Linear(48, 3)
        self.action = nn.Linear(48, 4)
        self.direct = nn.Linear(48, 4)

    def forward(self, sentences):
        length = torch.tensor([len(s) for s in sentences])
        size = int(length.max())
        ids = torch.zeros(len(sentences),size,dtype=torch.long)
        feats = torch.zeros(len(sentences),size,6)
        for row, sentence in enumerate(sentences):
            ids[row,:len(sentence)] = torch.tensor([self.vocab.get(w.lower(),1) for w in sentence])
            feats[row,:len(sentence)] = torch.tensor([shape(w) for w in sentence])
        if self.training:
            ids = ids.masked_fill((ids > 1) & (torch.rand(ids.shape) < .15),1)
        encoded = torch.cat((self.words(ids),feats),-1)
        pack = nn.utils.rnn.pack_padded_sequence(encoded,length.cpu(),batch_first=True,enforce_sorted=False)
        seq, h = self.ctx(pack)
        seq, _ = nn.utils.rnn.pad_packed_sequence(seq,batch_first=True,total_length=size)
        clause = torch.cat((h[0],h[1]),-1)
        return self.action(clause),self.role(seq),self.direct(clause)


def train(seed, data, steps=260):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    patterns = PROSE[:2]
    vocab = sorted({w.lower() for form in patterns for fmt in form for n in NAMES for v in range(4)
                    for w in tokens(fmt.format(name=n,v=v).encode())})
    vocab = {w:i+2 for i,w in enumerate(vocab)}
    model = Tagger(vocab)
    opt = torch.optim.AdamW(model.parameters(),lr=.005,weight_decay=.005)
    negatives = []
    while len(negatives) < 768:
        for row in data.batch("train",rng,8,256):
            for raw in re.findall(rb"[^.!?]+[.!?]",row.astype(np.uint8).tobytes()):
                t = tokens(raw)
                if 3 <= len(t) <= 22:
                    negatives.append(t)
    for step in range(steps):
        examples, acts, positions, answers = [],[],[],[]
        # Each type and name is represented equally per batch across source roles.
        for action in (IGNORE,TRUST,WEAK,READ):
            for i in range(8):
                val = int(rng.integers(4)); name=NAMES[i%4]
                if action==IGNORE:
                    words=negatives[int(rng.integers(len(negatives)))]
                    labs=[OTHER]*len(words)
                else:
                    form=patterns[int(rng.integers(2))][action-1]
                    words=tokens(form.format(name=name,v=val).encode())
                    labs=[SUBJECT if w==name.encode() else VALUE if w==str(val).encode()
                          else OTHER for w in words]
                examples.append(words);acts.append(action);positions.append(labs);answers.append(val)
        cls,roles,direct=model(examples)
        labs=torch.full(roles.shape[:2],-100)
        for i,ls in enumerate(positions):labs[i,:len(ls)]=torch.tensor(ls)
        # The direct query-only control sees the same query but no stored history.
        loss=F.cross_entropy(cls,torch.tensor(acts))+F.cross_entropy(
            roles.flatten(0,1),labs.flatten(),weight=torch.tensor([.25,1.,1.])
        )+.15*F.cross_entropy(direct[-8:],torch.tensor(answers[-8:]))
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    return model.eval()


@torch.no_grad()
def parse_many(model, clauses):
    """Batch-independent offline acceleration of the bounded sentence parser."""
    out={}
    for start in range(0,len(clauses),64):
        chunk=clauses[start:start+64]
        words=[tokens(c) or [b"."] for c in chunk]
        act, role, direct=model(words)
        act=act.softmax(-1);role=role.softmax(-1)
        for i,(raw,ws) in enumerate(zip(chunk,words)):
            confidence, action=act[i].max(0)
            pos=role[i,:len(ws)].argmax(-1)
            def extract(tag):
                at=(pos==tag).nonzero().flatten()
                if len(at)!=1:return None
                idx=int(at[0])
                return ws[idx] if float(role[i,idx,tag])>=.9 else None
            subject=extract(SUBJECT);value=extract(VALUE)
            if subject is None or len(subject)>KEY_BYTES or confidence<.95:
                action=IGNORE
            if int(action)==TRUST and (value is None or len(value)!=1):
                action=IGNORE
            out[raw]=(int(action),subject.lower() if subject else None,
                      value[0] if value and len(value)==1 else None,48+int(direct[i].argmax()))
    return out


def segments(raw):
    start=0
    for i,b in enumerate(raw):
        if b in BOUNDARY:
            if i-start+1<=CLAUSE_BYTES:
                clause=raw[start:i+1].strip()
                if clause:yield i,clause
            start=i+1


def actions(raw, parsed, state=None):
    bank=OrderedDict() if state is None else state
    updates={};direct={};writes=0
    for end,c in segments(raw):
        kind,key,value,guess=parsed[c]
        if kind==TRUST:
            if key not in bank and len(bank)==CAPACITY:
                bank.popitem(last=False)
            bank[key]=value;writes+=1
        elif kind==READ:
            direct[end]=guess
            if key in bank:updates[end]=bank[key]
    return updates,direct,writes


class StreamingStore:
    """Consume incoming bytes with bounded memory and dispatch at the current byte."""
    def __init__(self,tagger):
        self.tagger=tagger
        self.bank=OrderedDict()
        self.pending=bytearray()
        self.overflow=False

    def step(self,b):
        if len(self.pending)<CLAUSE_BYTES:self.pending.append(b)
        else:self.overflow=True
        if b not in BOUNDARY:return None
        raw=bytes(self.pending)
        skip=self.overflow
        self.pending.clear();self.overflow=False
        if skip:return None
        clause=raw.strip()
        if not clause:return None
        action,key,value,_=parse_many(self.tagger,[clause])[clause]
        if action==TRUST:
            if key not in self.bank and len(self.bank)==CAPACITY:self.bank.popitem(last=False)
            self.bank[key]=value
        elif action==READ:return self.bank.get(key)
        return None


class StructuredCOG(nn.Module):
    """CPU inference adapter for WholeCOG. Branching a carried session is safe."""
    def __init__(self, base, tagger):
        super().__init__()
        self.base=base
        self.tagger=tagger
        self.disable_read=False

    @torch.inference_mode()
    def forward(self, x, state=None, carry=False):
        prior=None if state is None else state[0]
        z,core=self.base(x,prior,carry=True)
        next_rows=[];fused=[]
        for i,row in enumerate(x.tolist()):
            store=StreamingStore(self.tagger)
            if state is not None:
                bank,pending,overflow=state[1][i]
                store.bank=OrderedDict(bank);store.pending.extend(pending);store.overflow=overflow
            edits={}
            for t,b in enumerate(row):
                if b>255:
                    store.pending.clear();store.overflow=False
                    continue
                read=store.step(b)
                if read is not None:edits[t]=read
            next_rows.append((tuple(store.bank.items()),bytes(store.pending),store.overflow))
            fused.append(z[i] if self.disable_read else fuse(z[i],edits))
        out=torch.stack(fused)
        return (out,(core,tuple(next_rows))) if carry else out


def fuse(logits, edits):
    """Make an authorized byte the top prediction, preserving all other logits."""
    if not edits:return logits
    logits=logits.clone()
    for t,token in edits.items():
        logits[t,token]=logits[t].max()+2
    return logits


def parse_prompts(tagger,texts):
    clauses=sorted({c for raw in texts for _,c in segments(raw)})
    return parse_many(tagger,clauses)


@torch.inference_mode()
def compare(tagger,base,texts,with_base=None):
    parsed=parse_prompts(tagger,texts)
    full=[];direct=[];dispatched=0;disabled=[]
    for at in range(0,len(texts),8):
        group=texts[at:at+8]
        ids=torch.zeros(len(group),max(map(len,group)),dtype=torch.long)
        for i,t in enumerate(group):ids[i,:len(t)]=torch.tensor(list(t))
        logits=base(ids) if with_base is None else None
        for i,t in enumerate(group):
            pos=len(t)-1
            edits,shortcut,_=actions(t,parsed)
            off=int(logits[i,pos].argmax()) if logits is not None else with_base[at+i]
            disabled.append(off)
            if pos in edits:dispatched+=1
            full.append(edits.get(pos,off));direct.append(shortcut.get(pos,off))
    return dict(enabled=full,disabled=disabled,query_tagger_only=direct),dispatched


def evaluate(model,base,seed,cached,worlds):
    from confirm_repair import _episodes,_counts,PANELS,SEED
    result={}
    for name in PANELS:
        cases=_episodes(name,worlds,SEED)
        texts=[x for c in cases for x in c[:2]]
        pred,dispatch=compare(model,base,texts,cached.get(name))
        cached[name]=pred["disabled"]
        result[name]={k:_counts(cases,v) for k,v in pred.items()}
        result[name]["dispatch"]=dispatch
    cases=_episodes("balanced_familiar",worlds,SEED+1)
    # Remove the trusted claim belonging to the queried subject in both worlds.
    trimmed=[]
    for sample in cases:
        for raw in sample[:2]:
            clauses=list(segments(raw))
            subject=tokens(clauses[-1][1])[-2]
            parts=[c for _,c in clauses if not (c.startswith(b"The log lists ") and subject in tokens(c))]
            trimmed.append(b" ".join(parts))
    parsed=parse_prompts(model,trimmed)
    missing=sum(len(x)-1 in actions(x,parsed)[0] for x in trimmed)
    result["missing_trusted"]={"queries":len(trimmed),"dispatched":missing}
    return result


@torch.inference_mode()
def language(model,base,corpus):
    rng=np.random.default_rng(781299)
    rows=corpus.batch("dev",rng,256,128)
    prompts=[r[:-1].astype(np.uint8).tobytes() for r in rows]
    parsed=parse_prompts(model,prompts)
    prime=b"The log lists Ada at 1. The log lists Ben at 2. "
    prime_map=parse_prompts(model,[prime])
    prime_bank=OrderedDict()
    actions(prime,prime_map,prime_bank)
    total=0;false=0;false_primed=0;writes=0;plain=hybrid=0.
    for at in range(0,len(rows),8):
        x=torch.from_numpy(rows[at:at+8]);logits=base(x[:,:-1])
        plain+=float(F.cross_entropy(logits.flatten(0,1),x[:,1:].flatten(),reduction="sum"))
        fused=[]
        for i,t in enumerate(prompts[at:at+8]):
            edits,_,w=actions(t,parsed)
            primed,_,_=actions(t,parsed,prime_bank.copy())
            total+=len(t);false+=len(edits);false_primed+=len(primed);writes+=w
            fused.append(fuse(logits[i],edits))
        hybrid+=float(F.cross_entropy(torch.stack(fused).flatten(0,1),x[:,1:].flatten(),reduction="sum"))
    return {"dev_bytes":total,"disabled_bpb":plain/total/np.log(2),
            "enabled_bpb":hybrid/total/np.log(2),"read_overrides":false,
            "read_overrides_after_synthetic_priming":false_primed,"trusted_writes":writes}


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "unified-search-002"))
    from integrated import WholeCOG
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True)
    p.add_argument("--parent",type=Path,required=True);p.add_argument("--worlds",type=int,default=64)
    p.add_argument("--corpus",type=Path,default=Path("trace_corpus"))
    p.add_argument("--seeds",nargs="+",type=int,default=[9731,9743,9769])
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    data=load_corpus(args.corpus)
    base=WholeCOG("SS_native_native",9731).eval()
    base.load_state_dict(torch.load(args.parent,weights_only=True))
    parent_sha=hashlib.file_digest(open(args.parent,"rb"),"sha256").hexdigest()
    cached={}
    for seed in args.seeds:
        tagger=train(seed,data)
        record={"seed":seed,"params":sum(p.numel() for p in tagger.parameters()),
                "frozen_parent_sha256":parent_sha,"confidence_action":.95,"confidence_roles":.9,
                "typed_memory_slots":32,"key_capacity":16,"clause_capacity":160,
                "confirmation":evaluate(tagger,base,seed,cached,args.worlds),"language":language(tagger,base,data)}
        sample=b"The log lists Ada at 1. A rumor puts Ada at 2. What digit does the log give for Ada?"
        store=StreamingStore(tagger)
        live={i:a for i,b in enumerate(sample) if (a:=store.step(b)) is not None}
        assert live==actions(sample,parse_prompts(tagger,[sample]))[0]
        record["streaming_check"]=True
        torch.save(dict(weights=tagger.state_dict(),vocab={w.decode("ascii"):i for w,i in tagger.vocab.items()}),
                   args.out/f"tagger_{seed}.pt")
        (args.out/f"metrics_{seed}.json").write_text(json.dumps(record,indent=2)+"\n")
        print(seed,{k:v["enabled"]["both_correct"] for k,v in record["confirmation"].items() if 'enabled'in v},
              record["language"],flush=True)


if __name__=="__main__":main()
