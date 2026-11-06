"""Post-result diagnostic for LRR-005: hard address, write and collision attribution.

Run against frozen checkpoint bytes only. The diagnostic is NOT used for
training, hyperparameter selection or a reinterpretation of prior gates.
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
import torch

from .lrr005_register import CausalRoleRegister, ID, make_example


def trace(model, example):
    words = example.words
    x = torch.tensor([[ID.get(w,1) for w in words]], dtype=torch.long)
    with torch.no_grad():
        h = model.token_features(x)
        gates = model.role(h).sigmoid()[0].tolist()
        codes = model.code(h)[0].argmax(-1).tolist()
    entity = relation = None
    postings = {}
    gold_key = None
    gold_written = False
    writes = 0
    nonvalue_writes = 0
    for i, word in enumerate(words):
        if i == len(words)-1:
            break
        current_key = (entity, relation)
        if i == example.answer:
            gold_key = current_key
            gold_written = gates[i][2] > 0.5
        if gates[i][2] > 0.5 and entity is not None and relation is not None:
            postings[current_key] = i
            writes += 1
            if not word.startswith("value"):
                nonvalue_writes += 1
        if gates[i][0] > 0.5:
            entity = word
        if gates[i][1] > 0.5:
            relation = codes[i]
    query_key = (entity,relation)
    answer = postings.get(query_key)
    if answer == example.answer:
        failure = "correct"
    elif answer is None:
        failure = "abstained"
    elif gold_key == query_key and gold_written:
        failure = "collision_or_overwrite"
    else:
        failure = "wrong_address_or_gate"
    return {"failure":failure,"address_match":gold_key==query_key,
            "gold_written":gold_written, "writes":writes,
            "nonvalue_writes":nonvalue_writes, "answer":answer}


def inspect(checkpoint_dir: str, seeds=(41,42,43,44,45), n=32, trials=64):
    output = []
    for seed in seeds:
        model=CausalRoleRegister().eval()
        path=Path(checkpoint_dir)/f"lrr005_register_local_credit_{seed}.pt"
        state=torch.load(path,map_location="cpu",weights_only=True)
        model.load_state_dict(state,strict=True)
        samples = [
            make_example((seed+50_000)*100_000_000 + n*1_000+i,
                         n,split="test",layout="regular")
            for i in range(trials)
        ]
        traces=[trace(model,ex) for ex in samples]
        output.append({
            "seed":seed, "facts":n,
            "failure_classes":dict(Counter(t["failure"] for t in traces)),
            "query_matches_gold_address":sum(t["address_match"] for t in traces),
            "gold_write_gate_on":sum(t["gold_written"] for t in traces),
            "nonvalue_writes_total":sum(t["nonvalue_writes"] for t in traces),
            "mean_writes":sum(t["writes"] for t in traces)/trials,
        })
    return output


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir",required=True)
    p.add_argument("--output")
    a=p.parse_args()
    result=inspect(a.checkpoint_dir)
    rendered=json.dumps(result,indent=2,sort_keys=True)
    if a.output:
        Path(a.output).write_text(rendered+"\n")
    print(rendered)


if __name__=="__main__":
    main()
