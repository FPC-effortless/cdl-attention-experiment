"""Train wording-augmented clause taggers before independent blind evaluation.

All new families below were authored for training. Inference uses the same
word shapes, learned action and role probabilities, confidence thresholds,
bounded store and optional read interface as structured_memory.py.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from corpus import load_corpus
from panels import NAMES, PROSE
from structured_memory import (Tagger, IGNORE, TRUST, WEAK, READ, OTHER, SUBJECT,
                               VALUE, tokens, parse_prompts, actions)

EXTRA_NAMES=("Jorin","Mila","Ravi","Fara","Vera","Zoric","Tala","Bilal",
             "Lira","Kori","Ulma","Daven","Farouk","Rosa","Leon","Tsuri")
NEW_FAMILIES=(
    ("The clerk {yes} {name} with {v}.","The clerk {no} {name} with {v}.","Which number belongs to {name}?"),
    ("For {name} the {good} result was {v}.","For {name} the {bad} result was {v}.","Give me the result for {name}."),
    ("{v} was {good} for {name}.","{v} was {bad} for {name}.","How much did {name} receive?"),
    ("{name} received {v} per the {source}.","{name} received {v} according to {noise}.","Can you tell me what {name} received?"),
    ("{source} shows {v} beside {name}.","{noise} suggests {v} beside {name}.","Look up the number beside {name}."),
    ("For {name}, staff {yes} the number {v}.","For {name}, staff {no} the number {v}.","What did staff assign to {name}?"),
    ("{name} - {v} ({good}).","{name} - {v} ({bad}).","Please give the figure for {name}."),
    ("{good} result: {v} belongs to {name}.","{bad} result: {v} belongs to {name}.","Which figure goes with {name}?"),
    ("Number {v}, {name}: {good}.","Number {v}, {name}: {bad}.","Report the figure assigned to {name}."),
    ("The {source} places {name} under {v}.","An item of {noise} places {name} under {v}.","Under which number can we find {name}?"),
    ("{name} has {v}; the committee {yes} it.","{name} has {v}; a visitor {no} it.","What was the committee's result for {name}?"),
    ("After checking, the desk {yes} {v} against {name}.","Without checking, the desk {no} {v} against {name}.","Tell me which value goes against {name}."),
    ("{name}: the {source} says {v}.","{name}: the {noise} says {v}.","For {name}, what number should I use?"),
    ("The number next to {name} is {v}, marked {good}.","The number next to {name} is {v}, marked {bad}.","Read the number next to {name}."),
    ("{v} corresponds to {name} in the {source}.","{v} corresponds to {name} in the {noise}.","Return the number corresponding to {name}."),
    ("The {good} value is {v} in the case of {name}.","The {bad} value is {v} in the case of {name}.","What value applies in the case of {name}?"),
)
GOOD=("approved","official","confirmed","verified","documented","registered","accepted","audited")
BAD=("provisional","unconfirmed","rumored","unverified","speculative","disputed","alleged","tentative")
YES=("certified","verified","approved","registered","recorded","confirmed","validated","documented")
NO=("speculated","estimated","assumed","imagined","guessed","suggested","suspected","alleged")
SOURCE=("ledger","registry","register","archive","certificate","database","transcript","logbook")
NOISE=("gossip","speculation","hearsay","conjecture","rumour","chatter","supposition","theory")
SEEDS=(9731,9743,9769)
UPDATES=512


def render(fmt,name,val,idx):
    # Coprime strides decorrelate choices across different lexical slots.
    return fmt.format(name=name,v=val,good=GOOD[idx%8],bad=BAD[(3*idx+1)%8],
                      yes=YES[(5*idx+2)%8],no=NO[(7*idx+3)%8],
                      source=SOURCE[(3*idx+4)%8],noise=NOISE[(5*idx+1)%8]).encode()


def training(seed, corpus):
    rng=np.random.default_rng(seed);torch.manual_seed(seed)
    families=PROSE[:2]+NEW_FAMILIES
    names=NAMES+EXTRA_NAMES
    vocab={w.lower() for f in families for fmt in f for i in range(8)
           for w in tokens(render(fmt,names[i],i%4,i))}
    vocab.update(name.lower().encode() for name in names)
    vocab={w:i+2 for i,w in enumerate(sorted(vocab))}
    tagger=Tagger(vocab)
    negatives=[]
    while len(negatives)<1024:
        for row in corpus.batch("train",rng,16,256):
            for raw in re.findall(rb"[^.!?]+[.!?]",row.astype(np.uint8).tobytes()):
                sentence=tokens(raw)
                if 3<=len(sentence)<=26:negatives.append(sentence)
    opt=torch.optim.AdamW(tagger.parameters(),lr=.004,weight_decay=.005)
    for step in range(UPDATES):
        samples=[];labels=[];acts=[];answers=[]
        for action in (IGNORE,TRUST,WEAK,READ):
            for j in range(8):
                n=names[(step*8+j)%len(names)];v=int(rng.integers(4))
                if action==IGNORE:
                    word=negatives[int(rng.integers(len(negatives)))];role=[OTHER]*len(word)
                else:
                    # Keep each original form represented in every training epoch.
                    fam=int(rng.integers(2 if j==0 else len(families)))
                    word=tokens(render(families[fam][action-1],n,v,int(rng.integers(8))))
                    role=[SUBJECT if w==n.encode() else VALUE if w==str(v).encode() else OTHER for w in word]
                samples.append(word);labels.append(role);acts.append(action);answers.append(v)
        action,role,direct=tagger(samples)
        lab=torch.full(role.shape[:2],-100)
        for i,row in enumerate(labels):lab[i,:len(row)]=torch.tensor(row)
        loss=F.cross_entropy(action,torch.tensor(acts))+F.cross_entropy(
            role.flatten(0,1),lab.flatten(),weight=torch.tensor([.25,1.,1.])
        )+.15*F.cross_entropy(direct[-8:],torch.tensor(answers[-8:]))
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    return tagger.eval()


def load_tagger(path):
    ck=torch.load(path,weights_only=True)
    m=Tagger({w.encode('ascii'):i for w,i in ck['vocab'].items()})
    m.load_state_dict(ck['weights']);return m.eval()


def load_predictor(path):
    model=load_tagger(path)
    def predict(raw):
        # Offline clause batching gives identical parser/store decisions to the stream.
        parsed=parse_prompts(model,[raw])
        return actions(raw,parsed)[0].get(len(raw)-1)
    return predict


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--corpus',type=Path,default=Path('trace_corpus'))
    args=p.parse_args();args.out.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);corpus=load_corpus(args.corpus)
    rows=[]
    for seed in SEEDS:
        model=training(seed,corpus)
        path=args.out/f'tagger_{seed}.pt'
        torch.save(dict(weights=model.state_dict(),vocab={w.decode():i for w,i in model.vocab.items()}),path)
        row=dict(seed=seed,updates=UPDATES,learned_parameters=sum(p.numel() for p in model.parameters()),
                 sha256=hashlib.sha256(path.read_bytes()).hexdigest(),filename=path.name)
        rows.append(row);print(row,flush=True)
    manifest=dict(families=len(NEW_FAMILIES)+2,train_names=len(NAMES+EXTRA_NAMES),
                  seeds=list(SEEDS),checkpoints=rows,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.out/'FROZEN.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
