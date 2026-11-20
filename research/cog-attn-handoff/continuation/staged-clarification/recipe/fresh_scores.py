"""Pin NLI role scores on fresh templates before the clarification-policy comparison."""
import hashlib
import json
import sys
from pathlib import Path

import torch

from binding import KNOWN, lexical_ids, rows, words_for_seed
from language import NAMES, encode
from nli_transfer import Entailment
from paraphrases import generate

ROOT=Path(__file__).resolve().parent
SEEDS=(751,773,797)
PATTERNS={
    "implicit":(
        "The {v} operation draws its quantity from the line headed {s}. Reflect the finished number in {d}, leaving {u} as written.",
        "{d} is the ledger we'll post to. Start {v} with the registered figure at {s}; {u} is still current.",
        "Leave {u} alone. The figure parked at {s} drives this {v}, and the new total is for {d}.",
        "Look at the amount on file under {s}. Once you've done {v}, carry the answer over to {d}, with no change to {u}.",
        "The balance to replace is {d}. Base the {v} calculation on {s}. Leave the number for {u} alone.",
    ),
    "correction":(
        "The draft says the input is {u}; that's out of date. Read {s} instead and use {v} to change {d}.",
        "Please {v} with {s} as the operand. I cancelled the posting to {u}; record the result at {d}.",
        "I chose {s} over {u} for the figure to use. For {v}, put the resulting number into {d}.",
        "{u} was the proposed receiving ledger; make that {d}. The input for {v} stays {s}.",
    ),
    "ambiguous":(
        "{d} receives the {v} output, but I have conflicting instructions about whether to draw the quantity from {s} or {u}.",
        "The value for {v} starts at {s}. I still haven't settled whether {d} or {u} gets the result.",
        "Hold the {v} entry for {d}. Both {s} and {u} are in the source column; the approved source is missing.",
        "The note says to {v} using {s}. Only post to {d} if that ledger is current, otherwise to {u}; I can't establish which is current.",
    ),
}


def dataset(seed):
    extra=words_for_seed(seed)
    support=rows(seed+403,extra,96,"support")
    held={key:generate(seed+509+i,128,extra,templates) for i,(key,templates) in enumerate(PATTERNS.items())}
    return extra,support,held


def generate_scores(model,seed):
    extra,support,held=dataset(seed)
    vocabulary={w:i+4 for i,w in enumerate(sorted({token for row in support for token in row[0] if token not in NAMES}))}
    mentions=list(range(len(vocabulary)+4,len(vocabulary)+7))
    role,_=model.roles_and_clarity(support,KNOWN+extra,"separate")
    _,mask,_,_=encode(support,vocabulary,mentions)
    ids=lexical_ids(support,vocabulary,extra)
    log=(.98*(role[:,None,:]*mask.reshape(-1,4,6)).sum(-1)+.02/512).log()
    lexicon={vocabulary[w]:log[ids==vocabulary[w]].sum(0).softmax(-1) for w in extra}
    items=[]
    for cohort,commands in held.items():
        role,clarity=model.roles_and_clarity(commands,KNOWN+extra,"separate")
        verb_probs=torch.stack([lexicon[int(idx)] for idx in lexical_ids(commands,vocabulary,extra)])
        post=(verb_probs[:,:,None]*role[:,None,:]).flatten(1)
        for idx,(q,p,c) in enumerate(zip(commands,post,clarity,strict=True)):
            items.append(dict(cohort=cohort,index=idx,words=" ".join(q[0]),entities=q[1],state=q[2].tolist(),
                              truth=None if cohort=="ambiguous" else list(q[4:]),posterior=p.tolist(),clarity=float(c)))
    return dict(seed=seed,lexicon={w:lexicon[vocabulary[w]].tolist() for w in extra},items=items)


def main():
    model=Entailment()
    seeds=[int(s) for s in sys.argv[1:]] or SEEDS
    for seed in seeds:
        scores=generate_scores(model,seed)
        path=ROOT/f"scores-{seed}.json"
        serialized=json.dumps(scores,separators=(",",":")).encode()
        path.write_bytes(serialized)
        path.with_suffix(".sha256").write_text(hashlib.sha256(serialized).hexdigest())
        print(seed,len(scores["items"]),flush=True)


if __name__=="__main__":
    main()
