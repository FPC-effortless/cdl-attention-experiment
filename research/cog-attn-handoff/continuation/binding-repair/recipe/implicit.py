"""Same-model comparison of plain-language paraphrase augmentation."""
import os
import re
import random
from pathlib import Path

import numpy as np
import torch

from binding import Roles, evaluate, learn_words, lexical_ids, rows, train_roles, words_for_seed, KNOWN
from language import NAMES, apply, encode

TRAIN_CLAUSES=(
    "The balance sitting on {s} determines the quantity . use {verb} to revise {d} , leave {u} alone .",
    "{u} can wait . let {d} receive the result of {verb} using whatever {s} holds .",
    "The figure recorded under {s} is the reference . {verb} that value into {d} ; do nothing to {u} .",
    "For {d} use the {verb} method with the quantity found on {s} ; {u} is irrelevant .",
    "Whatever is in {s} is the input for {verb} . record the result on {d} and do not change {u} .",
    "Leave {u} alone . the figure on {d} should be revised by {verb} using the value in {s} .",
)

FRESH_CLAUSES={
    "shared_wording":(
        "The figure currently recorded under {s} supplies the input . {u} can wait while {d} receives the {verb} result .",
        "For the revised balance on {d} choose {verb} . the quantity sitting on {s} is the reference ; {u} is unaffected .",
        "{u} is irrelevant here . use whatever {s} holds for {verb} and record the new figure under {d} .",
    ),
    "distant_wording":(
        "Leave the line for {u} untouched . {d} is where the {verb} calculation ends up ; it starts with {s} .",
        "Of {u} , {s} and {d} only the last ledger should move . base its {verb} adjustment on the middle ledger .",
        "Consider {u} out of scope . {d} should wind up with the {verb} result after fetching the operand at {s} .",
    )
}


def clause_rows(seed,count,verbs,clauses):
    rng=random.Random(seed)
    result=[]
    for _ in range(count):
        op=rng.randrange(4)
        s,d,u=rng.sample(NAMES,3)
        command=rng.choice(clauses).format(verb=verbs[op],s=s,d=d,u=u)
        tokens=re.findall(r"[a-z]+|[^\w\s]",command.lower())
        entities=list(dict.fromkeys(t for t in tokens if t in NAMES))
        src,dst=entities.index(s),entities.index(d)
        state=np.array([rng.randrange(8) for _ in entities],dtype=np.int64)
        target=state.copy()
        target[dst]=apply(op,state[src],state[dst])
        result.append((tokens,entities,state,target,op,src,dst))
    return result


def data_for(seed,arm):
    extra=words_for_seed(seed)
    train=rows(seed,extra,4096,"familiar")
    if arm=="candidate":
        train[2048:]=clause_rows(seed+6000,2048,KNOWN,TRAIN_CLAUSES)
    support=rows(seed+3060,extra,96,"support")
    tests={cohort:rows(seed+23000+i,extra,512,cohort)
           for i,cohort in enumerate(("joint","implicit","ambiguous"))}
    tests.update({name:clause_rows(seed+24000+i,512,extra,clauses)
                  for i,(name,clauses) in enumerate(FRESH_CLAUSES.items())})
    return extra,train,support,tests


def run(seed,arm):
    torch.manual_seed(seed)
    extra,train,support,tests=data_for(seed,arm)
    vocabulary={w:i+4 for i,w in enumerate(sorted({w for r in train+support for w in r[0] if w not in NAMES}))}
    names=list(range(len(vocabulary)+4,len(vocabulary)+7))
    verbs=KNOWN+extra
    word_ids=[vocabulary[w] for w in verbs]
    model=Roles(len(vocabulary)+7,names,word_ids,True)
    train_roles(model,encode(train,vocabulary,names),650,torch.Generator().manual_seed(seed+210))
    calibration=train[:384]+support
    lexicon=learn_words(model,encode(calibration,vocabulary,names),lexical_ids(calibration,vocabulary,verbs),word_ids)
    filename=Path(os.environ["TRAINLAB_OUTPUT_DIR"])/"local.npz"
    np.savez_compressed(filename,**{k:v.detach().numpy() for k,v in model.state_dict().items()})
    return [evaluate(model,encode(raw,vocabulary,names),name,lexical_ids(raw,vocabulary,verbs),lexicon)
            for name,raw in tests.items()]
