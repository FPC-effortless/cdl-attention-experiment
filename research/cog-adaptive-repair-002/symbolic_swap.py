"""Frozen-checkpoint symbolic addressing diagnostic with equal-frequency keys.

Base histories contain four distinct official values and one independent rumor
for each key. An identity query reads one value. Variants change the query,
one decisive fact, two official values while keeping a permutation, or one
irrelevant rumor. The single-fact edit can create a duplicated official value.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cog-balance-001"))
from panels import _batch_bytes
from scale import build_model
from repair_model import build_repair_model


VALUES = b"0123"
KEYS = b"abcd"
SEED = 889_415_201
VARIANTS = ("original", "other_question", "relevant_edit", "permuted_relevant_edit", "irrelevant_rumor_edit")


def oracle(prompt):
    history, query = prompt.split(b"#")
    assert len(history) == 24 and query[1:] == b"00="
    facts = {history[p+1]:history[p+2] for p in range(0,24,3) if history[p] == ord("!")}
    return facts[query[0]]


def worlds(n=128):
    rng = np.random.default_rng(SEED)
    prompts = []
    labels = []
    for _ in range(n):
        official = rng.permutation(4)
        weak = rng.integers(4,size=4)
        target, other = rng.choice(4,2,replace=False)
        order = rng.permutation(8)
        def render(facts, rumors, subject):
            data = bytearray()
            for item in order:
                flag,k = divmod(int(item),4)
                data.extend((b"!~"[flag], KEYS[k], VALUES[(rumors if flag else facts)[k]]))
            data.extend(b"#"+bytes([KEYS[subject]])+b"00=")
            return bytes(data)
        old = render(official,weak,target)
        changed = official.copy()
        changed[target] = official[other]
        swapped = official.copy()
        swapped[target],swapped[other] = official[other],official[target]
        rumor = weak.copy()
        rumor[target] = (rumor[target]+rng.integers(1,4)) % 4
        batch = (old,render(official,weak,other),render(changed,weak,target),
                 render(swapped,weak,target),render(official,rumor,target))
        gold = [VALUES[official[target]], VALUES[official[other]], VALUES[official[other]],
                VALUES[official[other]], VALUES[official[target]]]
        assert [sum(a!=b for a,b in zip(old, raw)) for raw in batch] == [0,1,1,2,1]
        assert [oracle(raw) for raw in batch] == gold
        prompts.extend(batch)
        labels.append(gold)
    return prompts,np.asarray(labels)


def score(predictions, gold):
    pred = np.asarray(predictions).reshape(gold.shape)
    correct = pred == gold
    return {
        "worlds": len(gold),
        "original_correct": int(correct[:,0].sum()),
        "paired": {
            k: {"both_correct":int((correct[:,0] & correct[:,i]).sum()),
                "answer_changed":int((pred[:,0]!=pred[:,i]).sum())}
            for i,k in enumerate(VARIANTS[1:],start=1)
        },
        "all_five_correct": int(correct.all(axis=1).sum()),
        "predicted_ids": pred.tolist(),
    }


@torch.inference_mode()
def evaluate(model, prompts, gold):
    all_pred=[]
    for start in range(0,len(prompts),32):
        ids=_batch_bytes(prompts[start:start+32])
        logits=model(ids)
        scores=logits[:,-1]
        assert torch.isfinite(scores).all()
        all_pred.extend(scores.argmax(dim=-1).tolist())
    return score(all_pred,gold)


def evaluate_checkpoint(arm,seed,weights):
    model=build_repair_model(seed) if arm=="R" else build_model(arm,seed)
    model.load_state_dict(torch.load(weights,weights_only=True))
    model.eval()
    prompts,gold=worlds()
    with weights.open("rb") as stream:
        fingerprint=hashlib.file_digest(stream,"sha256").hexdigest()
    record={"arm":arm,"seed":seed,"panel_seed":SEED,"weights_sha256":fingerprint,
            "prompt_sha256":hashlib.sha256(b"".join(prompts)).hexdigest(),
            "enabled":evaluate(model,prompts,gold)}
    if arm=="R":
        model.disable_read=True
        record["disabled"]=evaluate(model,prompts,gold)
    return record


def baselines():
    texts,labels=worlds()
    def fixed_first(raw):
        return next(raw[i+2] for i in range(0,24,3) if raw[i:i+2]==b"!a")
    return {name:score([rule(t) for t in texts],labels)
            for name,rule in {
                "input_grammar": oracle, "fixed_key_a": fixed_first,
                "last_observed_claim":lambda t:t[23],
                "constant_zero":lambda t:48,
                "deterministic_uniform_digit":lambda t:VALUES[hashlib.sha256(t).digest()[0]%4],
            }.items()}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm",choices=("C","R","GQA","GRU","baselines"),required=True)
    p.add_argument("--seed",type=int)
    p.add_argument("--weights",type=Path)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if a.arm=="baselines":
        result=baselines()
    else:
        if a.weights is None or a.seed is None:
            p.error("--weights and --seed are required for a model")
        result=evaluate_checkpoint(a.arm,a.seed,a.weights)
    a.out.parent.mkdir(exist_ok=True,parents=True)
    a.out.write_text(json.dumps(result,separators=(",",":"))+"\n")
