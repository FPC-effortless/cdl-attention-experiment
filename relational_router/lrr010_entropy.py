"""LRR-010: ONE marginal code-use term, unchanged hard inference and base loss.

Ground-truth relation families are used by diagnostic evaluators only.
Neither entropy regularization nor routing receives semantic family labels.
"""
from __future__ import annotations
import math
from collections import Counter, defaultdict
import torch
from torch.nn import functional as F
from .lrr005_register import FAMILIES, vectorize
from .lrr008_invariance import InvariantRegister, triple, heldout, alias_pair, FORBIDDEN
from .lrr009_curriculum import predictions, summarize

ARMS = ('interleaved','clean_first','augmentation_first','interleaved_entropy')
CONDITIONS = ('regular','seen_permutation','untrained_in_vocab','oov','syntax_shift')
MILESTONES = (0,100,250,500,750,1000)


def entropy_objective(model,x,y,q,mask,noise,mode,weight=0.):
    """Algebraically identical LRR-008/009 objective when weight=0."""
    if mode not in ('clean','augmentation') or weight<0:
        raise ValueError('invalid treatment')
    used=mask if mode=='augmentation' else torch.zeros_like(mask)
    e,r,g=model.augmented_forward(x,used,noise)
    rows=torch.arange(len(x)); qe,qr=e[rows,q],r[rows,q]
    scores=((e*qe[:,None]).sum(-1)*(r*qr[:,None]).sum(-1)*g[:,:,2]).clamp_min(1e-9)
    valid=torch.arange(x.shape[1])[None]<q[:,None]
    logits=torch.where(valid,scores.log(),torch.full_like(scores,-1e9))
    logits=torch.cat((logits,model.null_logit.expand(len(x),1)),-1)
    ce=F.cross_entropy(logits,y)
    gate=.018*(g*valid[:,:,None]).mean()+.06*F.softplus(-g[rows,y,2].clamp_min(1e-9).log()).mean()
    anchor,positive,negative=qr[0::3],qr[1::3],qr[2::3]
    same=(anchor*positive).sum(-1); diff=(anchor*negative).sum(-1)
    relational=-.2*same.clamp_min(1e-6).log().mean()-.2*torch.log1p(-diff.clamp(max=1-1e-6)).mean()
    marginal=qr.mean(0)
    usage_entropy=-(marginal*marginal.clamp_min(1e-9).log()).sum()
    sample_entropy=-(qr*qr.clamp_min(1e-9).log()).sum(-1).mean()
    loss=ce+gate+relational-weight*usage_entropy
    return loss,{'ce':ce.detach().item(),'same':same.detach().mean().item(),
                 'different':diff.detach().mean().item(),
                 'marginal_entropy':usage_entropy.detach().item(),
                 'sample_entropy':sample_entropy.detach().item()}


def semantic_examples(seed,per_cell=8):
    """DEV-only independent histories, balanced 4 families x 4 aliases."""
    counts=Counter(); out=[]
    for draw in range(100000):
        candidates=triple(210_000_000_000+seed*1_000_000+draw,draw,0)
        for ex in candidates:
            key=(ex.family,FAMILIES[ex.family].index(ex.question_alias))
            if counts[key]<per_cell:
                assert alias_pair(ex) not in FORBIDDEN
                out.append(ex);counts[key]+=1
                break  # no two observations from the same history
        if len(out)==16*per_cell:break
    if len(out)!=16*per_cell:raise AssertionError('unbalanced semantic diagnostic')
    assert set(counts.values())=={per_cell}
    return out


def hard_entropy(counts):
    n=sum(counts.values())
    return -sum((v/n)*math.log(v/n) for v in counts.values() if v) if n else 0.


def semantic_statistics(records):
    """Task-specific collision threshold is NOT a generic collapse theorem."""
    n=len(records); byfam=defaultdict(Counter); allcodes=Counter(); family_n=Counter()
    for row in records:
        f=row['true_family'];c=row['query_hard_relation_code'];family_n[f]+=1
        if c is not None:byfam[f][c]+=1;allcodes[c]+=1
    missing=n-sum(allcodes.values())
    denom=(n*n-sum(v*v for v in family_n.values()))//2
    numerator=0
    for code,total in allcodes.items():
        numerator+=(total*total-sum(ct[code]**2 for ct in byfam.values()))//2
    cross=numerator/denom if denom else 0.
    H=hard_entropy(allcodes); neff=math.exp(H) if allcodes else 0.
    largest=max(allcodes.values())/sum(allcodes.values()) if allcodes else 1.
    conditional=sum(sum(c.values())*hard_entropy(c) for c in byfam.values())/max(1,sum(allcodes.values()))
    return {'n':n,'code_counts':dict(allcodes),'family_code_counts':dict(byfam),
            'hard_effective_codes':neff,'largest_code_share':largest,'missing_code_rate':missing/n,
            'occupied_codes':len(allcodes),'cross_family_pairs':denom,
            'cross_family_same_code_pairs':numerator,'cross_family_collision_rate':cross,
            'semantic_collapse':cross>=.10,'severe_starvation':neff<2 or largest>=.90,
            'conditional_code_entropy':conditional,'hard_code_family_MI':H-conditional,
            'noncollapsed_stratum':cross<.10 and missing/n<=.10,
            'query_gold_address_agreement':sum(d['gold_address_match'] for d in records)/n,
            'mean_record_collision_pairs':sum(d['collision_pairs'] for d in records)/n,
            'mean_nonvalue_writes':sum(d['nonvalue_writes'] for d in records)/n,
            'mean_writes':sum(d['writes'] for d in records)/n,
            'recall':sum(d['correct'] for d in records)/n}


def dev_panels(seed,trials=128,per_cell=8):
    return {cond:[heldout(310_000_000_000+seed*1_000_000+i,32,cond,split='dev')
                  for i in range(trials)] for cond in ('regular','untrained_in_vocab')} | {
                      'semantic':semantic_examples(seed,per_cell)}


def dev_evaluate(model,panels,step):
    summaries=[]; rows=[]
    for condition,examples in panels.items():
        diag=predictions(model,examples)
        current=[{'condition':condition,'case':i,'step':step,'gold':ex.answer,
                  'true_family':ex.family,'query_alias':ex.question_alias,**d}
                 for i,(ex,d) in enumerate(zip(examples,diag))]
        if condition=='semantic': summary=semantic_statistics(current)
        else: summary=summarize(diag)
        summaries.append({'step':step,'condition':condition,**summary})
        rows.extend(current)
    return summaries,rows
