"""Replay frozen trained LRR-005 checkpoints against cached/full hard inference.

Requires extracted artifact lrr-005-five-seed-register-credit, NOT TRAIN data.
This is a *mechanism parity* replay, not a new TEST performance run.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch

from .lrr005_register import CausalRoleRegister, make_example
from .lrr005_cache import CachedRoleRegister


def replay(checkpoints: Path, seeds=(41,42,43,44,45),
           lengths=(7,32,128), trials=8, chunk_sizes=(7,23,10_000)):
    torch.set_num_threads(1)
    passed=failed=0
    examples=0
    for seed in seeds:
        model=CausalRoleRegister().eval()
        name=checkpoints/f"lrr005_register_local_credit_{seed}.pt"
        model.load_state_dict(torch.load(name,map_location="cpu",weights_only=True))
        for n in lengths:
            for i in range(trials):
                sample=make_example((seed+50_000)*100_000_000+n*1_000+i,
                                    n,split="test",layout="regular")
                marker=sample.words.index("what")
                history=sample.words[:marker]
                question=sample.words[marker:]
                expected=model.route(sample.words)[0]
                examples+=1
                for chunk_size in chunk_sizes:
                    cache=CachedRoleRegister(model)
                    for k in range(0,len(history),chunk_size):
                        cache.append(history[k:k+chunk_size])
                    before=(tuple(cache.tokens),dict(cache.postings),cache.entity,cache.relation)
                    observed=cache.query(question)["token_index"]
                    after=(tuple(cache.tokens),dict(cache.postings),cache.entity,cache.relation)
                    if expected == observed and before == after:
                        passed+=1
                    else:
                        failed+=1
    return {"model_seeds":list(seeds),"lengths":list(lengths),
            "examples":examples,"comparisons":passed+failed,
            "passed":passed,"failed":failed,"status":"PASS" if failed==0 else "FAIL"}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoints",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    result=replay(args.checkpoints)
    output=json.dumps(result,indent=2)
    if args.output:
        args.output.write_text(output+"\n")
    print(output)
    if result["failed"]:
        raise SystemExit(1)


if __name__=="__main__":
    main()
