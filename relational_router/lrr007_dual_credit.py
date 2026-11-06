"""LRR-007 — matched same-answer AND different-answer address credit.

Reuses the unmodified LRR-005 CausalRoleRegister (4,748 parameters).
Training sees gold ANSWER TOKEN POSITIONS, not canonical relation codes.
Hard inference stays a one-address dictionary lookup without token attention.
"""
from __future__ import annotations

import random
import time
from dataclasses import replace

import torch
from torch.nn import functional as F

from .lrr005_register import (
    CausalRoleRegister, Example, FAMILIES, RESERVED, make_example, vectorize
)

# This experiment explicitly withholds two *new* ordered lexical
# combinations as well as both historically protected combinations.
TRAIN_FORBIDDEN = frozenset(RESERVED | {(0, 3), (2, 0)})
DEV_PAIR = (2, 0)
TEST_PAIR = (0, 3)


def actual_alias_pair(example: Example) -> tuple[int, int]:
    """Inspect the actual gold fact, never a guessed relation-class shortcut."""
    gold = [fact for fact in example.facts if fact[-1] == example.answer]
    if len(gold) != 1:
        raise AssertionError("unique answer-token position required")
    f = gold[0]
    if f[1] != example.family or f[0] != example.subject:
        raise AssertionError("gold fact metadata inconsistent")
    aliases = FAMILIES[example.family]
    return aliases.index(f[2]), aliases.index(example.question_alias)


def _allowed_aliases(fact) -> tuple[str, ...]:
    family = fact[1]
    aliases = FAMILIES[family]
    fact_number = aliases.index(fact[2])
    return tuple(word for qindex, word in enumerate(aliases)
                 if (fact_number, qindex) not in TRAIN_FORBIDDEN)


def training_triple(seed: int, step: int, index: int) -> tuple[Example, Example, Example]:
    """Paired questions over IDENTICAL history; no test relation pair exposure."""
    original = make_example(seed, n_facts=7, split="train", layout="regular")
    rng = random.Random(seed ^ (step << 13) ^ (index << 7) ^ 0x1957)
    target = next(f for f in original.facts if f[-1] == original.answer)
    permitted = _allowed_aliases(target)
    if len(permitted) < 2:
        raise AssertionError("not enough legal same-answer paraphrases")
    anchor_word, positive_word = rng.sample(permitted, 2)
    def variant(word: str, family: str, answer: int):
        words = list(original.words)
        words[-2] = word
        return replace(original, words=tuple(words), answer=answer,
                       family=family, question_alias=word)
    anchor = variant(anchor_word, target[1], target[-1])
    positive = variant(positive_word, target[1], target[-1])
    candidates = [f for f in original.facts
                  if f[0] == original.subject and f[1] != original.family]
    if not candidates:
        raise AssertionError("missing same-entity different-relation negative")
    different = rng.choice(candidates)
    negative_word = rng.choice(_allowed_aliases(different))
    negative = variant(negative_word, different[1], different[-1])
    if anchor.answer != positive.answer or anchor.answer == negative.answer:
        raise AssertionError("invalid positive/negative answer identity")
    if anchor.question_alias == positive.question_alias:
        raise AssertionError("positive query must change lexical predicate")
    if not (anchor.words[:-2] == positive.words[:-2] == negative.words[:-2]):
        raise AssertionError("query triple not grounded in the same history")
    for example in (anchor, positive, negative):
        if actual_alias_pair(example) in TRAIN_FORBIDDEN:
            raise AssertionError("protected pair leaked to TRAIN")
    return anchor, positive, negative


def heldout_example(seed: int, facts: int, *, split: str,
                    layout: str = "regular") -> Example:
    """New DEV and TEST pairings; no direct TEST question injected in TRAIN."""
    if split not in ("dev","test"):
        raise ValueError("held-out split must be dev or test")
    original = make_example(seed, facts, split="test", layout=layout)
    pair = DEV_PAIR if split == "dev" else TEST_PAIR
    aliases = FAMILIES[original.family]
    facts_meta = list(original.facts)
    gold = next(f for f in facts_meta if f[-1] == original.answer)
    words = list(original.words)
    # Find only the target fact's existing predicate span. This uses
    # generator annotations on the EVALUATION side, not the router.
    clause_start = original.answer
    while clause_start > 0 and words[clause_start-1] != ".":
        clause_start -= 1
    predicate_positions = [
        i for i in range(clause_start, original.answer)
        if words[i] == gold[2]
    ]
    if len(predicate_positions) != 1:
        raise AssertionError("target predicate position not unique")
    words[predicate_positions[0]] = aliases[pair[0]]
    words[-2] = aliases[pair[1]]
    facts_meta = [
        (f[0], f[1], aliases[pair[0]], f[3], f[4])
        if f[-1] == original.answer else f
        for f in facts_meta
    ]
    modified = replace(original, words=tuple(words),
                       question_alias=aliases[pair[1]],
                       facts=tuple(facts_meta))
    if actual_alias_pair(modified) != pair:
        raise AssertionError("held-out pair corrupted")
    return modified


def paired_objective(model: CausalRoleRegister, x, answer, query,
                     triplets: list[tuple[int,int,int]],
                     *, arm: str) -> tuple[torch.Tensor,dict]:
    """Single-forward reproduction of LRR-005 negative-pair baseline.

    A: original negative-only CE+code credit on a common triplet stream.
    B: A + positive same-answer synonym alignment.
    C: auxiliary ranked-energy arm (secondary hypothesis only).
    """
    if arm not in ("negative_only","two_sided","energy_rank"):
        raise ValueError("unknown experimental arm")
    e, r, g = model(x)
    rows = torch.arange(len(x), device=x.device)
    qe, qr = e[rows,query], r[rows,query]
    match = ((e*qe[:,None,:]).sum(-1) *
             (r*qr[:,None,:]).sum(-1) * g[:,:,2]).clamp_min(1e-9)
    valid = torch.arange(x.shape[1],device=x.device)[None,:] < query[:,None]
    scores = torch.where(valid,match.log(),torch.full_like(match,-1e9))
    scores = torch.cat((scores,model.null_logit.expand(len(x),1)),-1)
    ce = F.cross_entropy(scores,answer)
    regularizer = .018*(g*valid[:,:,None]).mean()
    pos = g[rows,answer,2].clamp_min(1e-9)
    positive_write_loss = F.softplus(-torch.log(pos)).mean()
    baseline = ce + regularizer + .06*positive_write_loss
    triple_ix = torch.tensor(triplets,dtype=torch.long,device=x.device)
    a,b,c = qr[triple_ix[:,0]],qr[triple_ix[:,1]],qr[triple_ix[:,2]]
    same=(a*b).sum(-1)
    different=(a*c).sum(-1)
    neg = -torch.log1p(-different.clamp(max=1-1e-6)).mean()
    pos = -torch.log(same.clamp(min=1e-6)).mean()
    if arm == "negative_only":
        loss=baseline+0.2*neg
    elif arm == "two_sided":
        loss=baseline+0.2*neg+0.2*pos
    else:
        loss=baseline+0.2*F.softplus(0.25+different-same).mean()
    return loss,{
        "ce":float(ce.detach()),
        "same_relation_soft_agreement":float(same.detach().mean()),
        "different_relation_soft_agreement":float(different.detach().mean()),
        "positive_alignment_penalty":float(pos.detach()),
        "negative_separation_penalty":float(neg.detach()),
    }


def fit_arm(seed: int, arm: str, *, steps: int = 500,
            triples_per_update: int = 16):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = CausalRoleRegister()
    optimizer = torch.optim.AdamW(model.parameters(),lr=.004)
    start=time.perf_counter()
    tested_pairs=0
    last={}
    for step in range(steps):
        examples=[]
        indices=[]
        for i in range(triples_per_update):
            triple=training_triple(
                seed*1_000_000+step*1_000+i,step,i)
            indices.append((len(examples),len(examples)+1,len(examples)+2))
            examples.extend(triple)
            tested_pairs+=3
        x,answer,query=vectorize(examples)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss,metrics=paired_objective(
            model,x,answer,query,indices,arm=arm)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),2.0)
        optimizer.step()
        last=metrics
    model.eval()
    return model,{
        "seed":seed,"arm":arm,"steps":steps,
        "questions_per_step":triples_per_update*3,
        "source_histories":steps*triples_per_update,
        "train_questions":tested_pairs,
        "trainable_parameters":sum(p.numel() for p in model.parameters()),
        "last_loss":float(loss.detach()),
        "train_wall_seconds":time.perf_counter()-start,
        **last,
    }
