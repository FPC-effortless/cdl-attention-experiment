"""LRR-006: explicitly trained one-candidate energy, separate from frozen router.

Correct and incorrect candidate positions are supplied ONLY for TRAIN energy
contrast. Inference calls the hard LRR index, then scores at most one entry.
No query-to-history attention or softmax over old tokens at inference.
"""
from __future__ import annotations

import math
import random
import time
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F

from .lrr005_register import CausalRoleRegister, make_example, ID, NAMES


def known_absent(words, gold, subject):
    seq = list(words)
    start = gold
    while start > 0 and seq[start - 1] != ".":
        start -= 1
    if seq[start] != subject:
        raise AssertionError("gold clause entity mismatch")
    possible = sorted({w for w in seq[:start] if w in NAMES and w != subject})
    if not possible:
        possible = sorted(w for w in NAMES if w != subject)
    seq[start] = possible[0]
    return tuple(seq)


def unknown_relation(words):
    seq = list(words)
    seq[-2] = "unseenverb"
    return tuple(seq)


@torch.no_grad()
def encode_single(model: CausalRoleRegister, words, *, forced_candidate=None):
    """Build hard causal index, read one address, form local energy features.

    forced_candidate is TRAIN-only teacher supervision and MUST NEVER be
    supplied to validation, test, service, or selectable inference.
    """
    tokens = tuple(words)
    x = torch.tensor([[ID.get(w, 1) for w in tokens]], dtype=torch.long)
    e, r, gates = model(x)
    emb = model.emb(x)
    h = model.norm(emb + F.gelu(model.conv(
        F.pad(emb.transpose(1, 2), (2, 0))
    ).transpose(1, 2)))
    hard_codes = model.code(h)[0].argmax(-1).tolist()
    g = gates[0].tolist()
    ent = rel = None
    postings = {}
    keyed_writes = defaultdict(int)
    nonvalue = querywrite = 0
    qstart = tokens.index("what")
    for pos, word in enumerate(tokens[:-1]):
        if g[pos][2] > .5 and ent is not None and rel is not None:
            key = (ent, rel)
            postings[key] = pos
            keyed_writes[key] += 1
            nonvalue += int(not word.startswith("value"))
            querywrite += int(pos >= qstart)
        if g[pos][0] > .5:
            ent = word
        if g[pos][1] > .5:
            rel = hard_codes[pos]
    key = (ent, rel)
    pos = postings.get(key) if forced_candidate is None else forced_candidate
    if pos is None:
        return None, None, {"candidate_absent": True, "writes": sum(keyed_writes.values())}
    if not (0 <= pos < len(tokens) - 1):
        raise ValueError("future or invalid candidate token")
    enq, enc = e[0, -1], e[0, pos]
    rnq, rnc = r[0, -1], r[0, pos]
    entropy = lambda p: float(-(p * p.clamp_min(1e-9).log()).sum())
    es = float((enq * enc).sum())
    rs = float((rnq * rnc).sum())
    w = float(gates[0, pos, 2])
    features = [
        *rnq.tolist(), *rnc.tolist(),
        es, rs, es*rs*w,
        entropy(enq), entropy(enc), entropy(rnq), entropy(rnc),
        float(enq.max()), float(enc.max()), float(rnq.max()), float(rnc.max()),
        *gates[0, pos].tolist(), *gates[0, -2].tolist(),
        float(ID.get(tokens[-2], 1) == 1),
        float(ID.get(tokens[pos], 1) == 1),
        (len(tokens)-1-pos)/len(tokens),
        min(1., keyed_writes[key]/20.),
        min(1., nonvalue/max(1,len(tokens))),
        min(1., querywrite/5.),
        min(1., len(postings)/max(1,len(tokens))),
    ]
    return pos, torch.tensor(features, dtype=torch.float32), {
        "candidate_absent": False, "similarity": es*rs*w,
        "writes": sum(keyed_writes.values()),
    }


class EnergySelector(nn.Module):
    def __init__(self, feature_dim, width=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feature_dim,width), nn.SiLU(),
                                 nn.Linear(width,1))

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def margin_loss(self, x, y, margin=0.5):
        # E_valid=-score; E_null=0. y=1 correct candidate, else y=0.
        return F.softplus(margin-(2*y-1)*self(x)).mean()


def make_conditions(example):
    return (
        ("positive", example.words, example.answer),
        ("known_absent", known_absent(example.words, example.answer,
                                     example.subject), -1),
        ("unknown_relation", unknown_relation(example.words), -1),
    )


def build_rows(model, seed, *, split="train", length=32, n=256, layout="regular"):
    populations = {"train": 2_000_000, "dev": 5_000_000, "test": 9_000_000}
    start = populations[split] + seed*1_000_000 + length*1_000
    rows = []
    for i in range(n):
        ex = make_example(start+i,length,split=split,layout=layout)
        for cls, words, gold in make_conditions(ex):
            pred,features,info = encode_single(model, words)
            rows.append({
                "class_name": cls, "feature": features,
                "score": float(info.get("similarity",float("-inf"))),
                "predicted": pred,
                "correct": cls == "positive" and pred == gold,
                "valid": cls == "positive", "gold": gold,
            })
        if split == "train":
            # Privileged candidate injections are limited to TRAIN.
            _,right_feature,right_info = encode_single(
                model, ex.words, forced_candidate=ex.answer)
            wrong = [fact[-1] for fact in ex.facts
                     if fact[0] == ex.subject and fact[-1] != ex.answer]
            if not wrong:
                raise AssertionError("no same-entity wrong-relation candidate")
            wrong_pos = wrong[i % len(wrong)]
            _,wrong_feature,wrong_info = encode_single(
                model, ex.words, forced_candidate=wrong_pos)
            for cls,feat,inf,pos,correct in (
                ("oracle_correct",right_feature,right_info,ex.answer,True),
                ("counterfactual_wrong",wrong_feature,wrong_info,wrong_pos,False),
            ):
                rows.append({
                    "class_name":cls, "feature":feat,
                    "score":float(inf.get("similarity",float("-inf"))),
                    "predicted":pos, "correct":correct,
                    "valid":True, "gold":ex.answer,
                })
    return rows


def fit_energy(rows, seed, steps=600):
    candidates = [r for r in rows if r["feature"] is not None]
    X = torch.stack([r["feature"] for r in candidates])
    y = torch.tensor([float(r["correct"]) for r in candidates])
    positives = (y==1).nonzero()[:,0]
    negatives = (y==0).nonzero()[:,0]
    torch.manual_seed(seed+23456)
    model = EnergySelector(X.shape[1])
    receipt = {
        "feature_dim":X.shape[1],
        "head_parameters":sum(p.numel() for p in model.parameters()),
        "candidate_examples":len(candidates),
        "positive_train_candidates":len(positives),
        "negative_train_candidates":len(negatives),
        "updates":steps,
    }
    if not len(positives) or not len(negatives):
        return model, {**receipt,"trainable":False}
    opt = torch.optim.AdamW(model.parameters(),lr=.003)
    start = time.perf_counter()
    model.train()
    for _ in range(steps):
        a=positives[torch.randint(len(positives),(64,))]
        b=negatives[torch.randint(len(negatives),(64,))]
        ids=torch.cat((a,b))
        opt.zero_grad(set_to_none=True)
        loss=model.margin_loss(X[ids],y[ids],margin=.5)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),2.0)
        opt.step()
    model.eval()
    return model, {**receipt,"trainable":True,
                   "last_loss":float(loss.detach()),
                   "training_head_wall_seconds":time.perf_counter()-start}


@torch.no_grad()
def score_rows(rows, *, head=None, scalar=False):
    scored = []
    for row in rows:
        if row["feature"] is None:
            s=float("-inf")
        elif scalar:
            s=float(row["score"])
        else:
            s=float(head(row["feature"].unsqueeze(0))[0])
        scored.append((row,s))
    return scored


def dev_threshold(scored, max_absent_fpr=.1):
    correct=[(r,s) for r,s in scored if r["class_name"]=="positive"]
    absent=[s for r,s in scored if r["class_name"]=="known_absent"]
    if not correct or not absent:
        raise ValueError("missing DEV labels")
    ordered=sorted({s for _,s in scored if math.isfinite(s)})
    if not ordered:
        return {"threshold":float("inf"),"correct_coverage":0.,
                "known_absent_fpr":0.}
    choices=[ordered[0]-1e-4]+[s+1e-6 for s in ordered]+[ordered[-1]+1]
    admissible=[]
    for threshold in choices:
        fpr=sum(s>threshold for s in absent)/len(absent)
        coverage=sum(r["correct"] and s>threshold for r,s in correct)/len(correct)
        if fpr<=max_absent_fpr+1e-12:
            admissible.append((coverage,-fpr,threshold))
    coverage,minus_fpr,threshold=max(admissible)
    return {"threshold":threshold,"correct_coverage":coverage,
            "known_absent_fpr":-minus_fpr}


def eval_scored(scored, threshold):
    names=("positive","known_absent","unknown_relation")
    partitions={name:[] for name in names}
    accepted=right=wrong=0
    for row,score in scored:
        approval=row["predicted"] is not None and score>threshold
        accepted+=int(approval)
        right+=int(approval and row["correct"])
        wrong+=int(approval and not row["correct"])
        partitions[row["class_name"]].append((row,approval))
    result={"accepted":accepted,"accepted_correct":right,
            "accepted_wrong":wrong,
            "accepted_error_risk":wrong/max(1,accepted)}
    for cls,entries in partitions.items():
        n=len(entries)
        result[cls]={
            "total":n,
            "raw_candidate_available":sum(row["predicted"] is not None
                                          for row,_ in entries),
            "accepted":sum(a for _,a in entries),
            "accept_rate":sum(a for _,a in entries)/n,
            "correct_coverage":sum(row["correct"] and a for row,a in entries)/n,
        }
    return result


def descriptive_risk_curve(scored):
    candidates=sorted((s for _,s in scored if math.isfinite(s)),reverse=True)
    if not candidates:
        return []
    outputs=[]
    for fraction in (0.,.05,.1,.25,.5,.75,1.):
        if fraction==0: threshold=candidates[0]+1
        elif fraction==1:threshold=candidates[-1]-1
        else:threshold=candidates[max(0,min(len(candidates)-1,
                                           round(len(candidates)*fraction)-1))]-1e-6
        m=eval_scored(scored,threshold)
        outputs.append({
            "candidate_fraction_target":fraction,
            "accepted":m["accepted"],
            "accepted_error_risk":m["accepted_error_risk"],
            "correct_positive_coverage":m["positive"]["correct_coverage"],
        })
    return outputs
