"""LRR-011: absolute READ/WRITE agreement vs self-selected confidence.

All hard inference, train bank, parameters and base loss are inherited unchanged.
Only an auxiliary TRAIN term receives the existing gold answer-position label.
Neither objective receives relation-family annotations or a hard writer mask.
"""
from __future__ import annotations
import torch
from torch.nn import functional as F

ARMS = ('control', 'self_confidence', 'read_write')
WEIGHT = .2

def auxiliary_terms(query, gold):
    """Agreement=1 iff the two categorical vectors are the same point mass."""
    agreement=(query*gold).sum(-1)
    binding=-agreement.clamp_min(1e-9).log().mean()
    confidence=-.5*(query.max(-1).values.clamp_min(1e-9).log()+
                      gold.max(-1).values.clamp_min(1e-9).log()).mean()
    return binding, confidence, agreement

def binding_objective(model,x,y,q,mask,noise,mode,arm,weight=WEIGHT):
    if mode not in ('clean','augmentation') or arm not in ARMS or weight<0:
        raise ValueError('invalid treatment')
    used=mask if mode=='augmentation' else torch.zeros_like(mask)
    e,r,g=model.augmented_forward(x,used,noise)
    rows=torch.arange(len(x),device=x.device)
    qe,qr=e[rows,q],r[rows,q]
    scores=((e*qe[:,None]).sum(-1)*(r*qr[:,None]).sum(-1)*g[:,:,2]).clamp_min(1e-9)
    valid=torch.arange(x.shape[1],device=x.device)[None]<q[:,None]
    logits=torch.where(valid,scores.log(),torch.full_like(scores,-1e9))
    logits=torch.cat((logits,model.null_logit.expand(len(x),1)),-1)
    ce=F.cross_entropy(logits,y)
    gate=.018*(g*valid[:,:,None]).mean()+.06*F.softplus(-g[rows,y,2].clamp_min(1e-9).log()).mean()
    anchor,positive,negative=qr[0::3],qr[1::3],qr[2::3]
    same=(anchor*positive).sum(-1); diff=(anchor*negative).sum(-1)
    relational=-.2*same.clamp_min(1e-6).log().mean()-.2*torch.log1p(-diff.clamp(max=1-1e-6)).mean()
    gold_r=r[rows,y]
    binding,confidence,agreement=auxiliary_terms(qr,gold_r)
    addition=binding if arm=='read_write' else confidence if arm=='self_confidence' else binding*0.
    loss=ce+gate+relational+weight*addition
    marginal=qr.mean(0)
    metrics={
        'ce':ce.detach().item(),'same':same.detach().mean().item(),
        'different':diff.detach().mean().item(),
        'marginal_entropy':-(marginal*marginal.clamp_min(1e-9).log()).sum().detach().item(),
        'sample_entropy':-(qr*qr.clamp_min(1e-9).log()).sum(-1).mean().detach().item(),
        'gold_sample_entropy':-(gold_r*gold_r.clamp_min(1e-9).log()).sum(-1).mean().detach().item(),
        'read_write_soft_agreement':agreement.detach().mean().item(),
        'binding_penalty':binding.detach().item(),'confidence_penalty':confidence.detach().item(),
        'query_maxprob':qr.max(-1).values.detach().mean().item(),
        'gold_maxprob':gold_r.max(-1).values.detach().mean().item(),
    }
    return loss,metrics
