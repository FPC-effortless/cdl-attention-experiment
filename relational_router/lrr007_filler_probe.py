"""LRR-007P: frozen-model, same-position filler lexical interventions.

POST-RESULT DIAGNOSTIC ONLY. Not a new confirmatory holdout or optimizer run.
Changes filler *surface tokens* while keeping every fact value/token position
and query relation untouched. Thus it separates lexical identity from syntax.
"""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import replace
from pathlib import Path

import torch
from torch.nn import functional as F

from .lrr005_register import CausalRoleRegister, ID
from .lrr007_dual_credit import heldout_example


FILLER_MAPS = {
    "original": {},
    "seen_filler_permutation": {
        "now": "quite", "quite": "recently", "recently": "now",
    },
    "untrained_in_vocab": {
        "now": "quickly", "quite": "often", "recently": "truly",
    },
    "out_of_vocab": {
        "now": "novelx", "quite": "novely", "recently": "novelz",
    },
}


def changed_words(case, mapping):
    words=tuple(mapping.get(word,word) for word in case.words)
    if len(words)!=len(case.words) or words[case.answer]!=case.words[case.answer]:
        raise AssertionError("filler transform changed gold value or position")
    return replace(case,words=words)


@torch.no_grad()
def hard_batch(model, examples):
    """Vectorized local encoder, then literal hard-index steps per example.

    Never calls a soft token-pair scorer. Agreement with source route()
    must hold for the built-in sanity examples.
    """
    model.eval()
    b=len(examples)
    length=max(len(ex.words) for ex in examples)
    ids=torch.zeros(b,length,dtype=torch.long)
    for i,ex in enumerate(examples):
        ids[i,:len(ex.words)]=torch.tensor([ID.get(word,1) for word in ex.words])
    h=model.token_features(ids)
    gates=model.role(h).sigmoid().tolist()
    codes=model.code(h).argmax(-1).tolist()
    output=[]
    for i,ex in enumerate(examples):
        entity=relation=None
        postings={}
        for pos,word in enumerate(ex.words[:-1]):
            if gates[i][pos][2]>.5 and entity is not None and relation is not None:
                postings[(entity,relation)]=pos
            if gates[i][pos][0]>.5:
                entity=word
            if gates[i][pos][1]>.5:
                relation=codes[i][pos]
        output.append(postings.get((entity,relation)))
    return output


def probe(checkpoint_dir, *, seeds=(51,52,53,54,55),
          facts=(7,32,128), trials=64, arms=("negative_only","two_sided"),
          original_results=None):
    torch.set_num_threads(1)
    rows=[]
    for arm in arms:
        for seed in seeds:
            model=CausalRoleRegister().eval()
            path=Path(checkpoint_dir)/f"lrr007_{arm}_seed{seed}.pt"
            model.load_state_dict(
                torch.load(path,weights_only=True,map_location="cpu"),
                strict=True)
            for n in facts:
                base=[
                    heldout_example(
                        70_000_000+seed*1_000_000+n*1000+i,
                        n,split="test",layout="regular")
                    for i in range(trials)
                ]
                for label,changes in FILLER_MAPS.items():
                    transformed=[changed_words(ex,changes) for ex in base]
                    predictions=hard_batch(model,transformed)
                    hits=sum(p==ex.answer for p,ex in zip(predictions,transformed))
                    if label=="original" and original_results is not None:
                        old=next(x["outcomes"]["correct"] for x in original_results["results"]
                                 if x["arm"]==arm and x["seed"]==seed
                                 and x["facts"]==n and x["layout"]=="regular")
                        if hits!=old:
                            raise AssertionError(f"frozen count mismatch {arm} {seed} H={n}")
                    rows.append({
                        "arm":arm,"seed":seed,"facts":n,
                        "condition":label,"trials":trials,
                        "correct":hits,"recall_at_1":hits/trials,
                    })
    return {
        "experiment":"LRR-007P, posthoc same-position lexical intervention",
        "scientific_status":"DIAGNOSTIC ONLY, no new model training and no confirmatory gate",
        "models_retrained":0,"optimizer_updates":0,
        "seeds":list(seeds),"lengths":list(facts),
        "same_gold_and_token_positions":True,
        "rows":rows,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoints",type=Path,required=True)
    p.add_argument("--original-results",type=Path,required=True)
    p.add_argument("--output",type=Path,default=Path("/tmp/lrr007p.json"))
    a=p.parse_args()
    original=json.loads(a.original_results.read_text())
    r=probe(a.checkpoints,original_results=original)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n")
    for arm in ("negative_only","two_sided"):
        for n in (7,32,128):
            print(arm,n,{
                key:round(statistics.mean(x["recall_at_1"] for x in r["rows"]
                                         if x["arm"]==arm and x["facts"]==n
                                         and x["condition"]==key),6)
                for key in FILLER_MAPS
            })


if __name__=="__main__":
    main()
