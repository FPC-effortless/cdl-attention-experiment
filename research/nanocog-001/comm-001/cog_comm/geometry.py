"""Verifier-side positional support bounds; never exported to model inputs."""
import argparse
from collections import defaultdict
import json

from .data import ANSWERS, generate, model_view, sentences
from .objective import Tokenizer


def support_row(world, tokenizer, *, block=8, retained_blocks=16):
    if block<1 or retained_blocks<1:
        raise ValueError("invalid route geometry")
    ss=sentences(world.prompt)
    query=len(tokenizer.encode(world.prompt))-1
    lower=max(0,(query//block-retained_blocks+1)*block)
    spans=[]
    for sent in world.support_positions:
        if not 0<=sent<len(ss)-1:
            raise ValueError("support must precede query")
        # Prefix encoding always starts with BOS and ends with an answer marker.
        start=len(tokenizer.encode(" ".join(ss[:sent])))-1
        end=len(tokenizer.encode(" ".join(ss[:sent+1])))-1
        if not 0<start<end<=query:
            raise ValueError("support boundary disagrees with tokenization")
        spans.append((start,end))
    return {"length":query+1,"support_nearest_age":query-max(hi-1 for _,hi in spans),
            "any_support_token_in_age_window":any(hi>lower for _,hi in spans),
            "all_support_tokens_in_age_window":all(lo>=lower for lo,_ in spans),
            "support_spans":spans}


def fixture(*, n=40, seed=77100, retain=(16,64)):
    fam=generate("train",n,seed)
    tok=Tokenizer.from_train(model_view(fam),ANSWERS)
    table={}
    for r in retain:
        grouped=defaultdict(list)
        for f in fam:
            for w in f.worlds:
                grouped[f.delay].append(support_row(w,tok,retained_blocks=r))
        result={}
        for d,v in sorted(grouped.items()):
            result[str(d)]={"worlds":len(v),
                "encoded_length_range":[min(x["length"] for x in v),max(x["length"] for x in v)],
                "nearest_support_age_range":[min(x["support_nearest_age"] for x in v),
                                             max(x["support_nearest_age"] for x in v)],
                "any_age_eligible_worlds":sum(x["any_support_token_in_age_window"] for x in v),
                "full_span_age_eligible_worlds":sum(x["all_support_tokens_in_age_window"] for x in v)}
        table[str(r)]=result
    return {"scope":"inert TRAIN generator fixtures, no admitted panel, model calls, fit or quality scores",
            "n":n,"seed":seed,"vocab":tok.size,"parameter":"retained blocks; block width eight; age only",
            "max_resident_payloads_per_layer_stream":8,"strata":table}


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--fixture",action="store_true",required=True)
    p.parse_args()
    print(json.dumps(fixture(),indent=2,sort_keys=True))
