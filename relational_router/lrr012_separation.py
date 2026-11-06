"""LRR-012: extra query repulsion vs direct incompatible WRITE-state repulsion.

Hard inference and parameters are unchanged. The only labels accessed by the
new loss are the existing answer-token indices. Gold family names are NOT
available here. All triplets share a history/entity; the generator validates
latest-value target identity and different-answer semantics before fitting.
"""
from __future__ import annotations
import torch
from torch.nn import functional as F

ARMS = ('control', 'query_separation', 'record_separation')
WEIGHT = .2


def negative_agreement(p: torch.Tensor, q: torch.Tensor):
    """Soft collision proxy. Its loss is not a hard uniqueness guarantee."""
    overlap = (p * q).sum(-1)
    penalty = -torch.log1p(-overlap.clamp(max=1-1e-6)).mean()
    return penalty, overlap


def separation_objective(model, x, y, q, mask, noise, mode, arm,
                         weight=WEIGHT, *, return_parts=False):
    if mode not in ('clean', 'augmentation') or arm not in ARMS or weight < 0:
        raise ValueError('invalid treatment')
    if len(x) % 3 or (y < 0).any():
        raise ValueError('positive answer-labeled triplets required')
    if not torch.equal(y[0::3], y[1::3]) or (y[0::3] == y[2::3]).any():
        raise ValueError('anchor/positive/negative target labels invalid')
    used = mask if mode == 'augmentation' else torch.zeros_like(mask)
    e, r, g = model.augmented_forward(x, used, noise)
    rows = torch.arange(len(x), device=x.device)
    qe, qr = e[rows, q], r[rows, q]
    scores = ((e * qe[:, None]).sum(-1) *
              (r * qr[:, None]).sum(-1) * g[:, :, 2]).clamp_min(1e-9)
    valid = torch.arange(x.shape[1], device=x.device)[None] < q[:, None]
    logits = torch.where(valid, scores.log(), torch.full_like(scores, -1e9))
    logits = torch.cat((logits, model.null_logit.expand(len(x), 1)), -1)
    ce = F.cross_entropy(logits, y)
    gate = (.018 * (g * valid[:, :, None]).mean() +
            .06 * F.softplus(-g[rows, y, 2].clamp_min(1e-9).log()).mean())
    anchor, positive, negative = qr[0::3], qr[1::3], qr[2::3]
    same = (anchor * positive).sum(-1)
    query_penalty, query_overlap = negative_agreement(anchor, negative)
    relational = -.2 * same.clamp_min(1e-6).log().mean() + .2 * query_penalty
    base = ce + gate + relational
    # y identifies the PRE-WRITE state inherited from the original causal
    # register. Only gold positions, not canonical relation labels, are read.
    writes = r[rows, y]
    record_penalty, record_overlap = negative_agreement(writes[0::3], writes[2::3])
    extra = query_penalty if arm == 'query_separation' else (
        record_penalty if arm == 'record_separation' else query_penalty * 0.)
    loss = base + weight * extra
    marginal = qr.mean(0)
    metrics = {
        'ce': ce.detach().item(), 'same': same.detach().mean().item(),
        'different': query_overlap.detach().mean().item(),
        'record_negative_overlap': record_overlap.detach().mean().item(),
        'query_negative_penalty': query_penalty.detach().item(),
        'record_negative_penalty': record_penalty.detach().item(),
        'marginal_entropy': -(marginal * marginal.clamp_min(1e-9).log()).sum().detach().item(),
        'sample_entropy': -(qr * qr.clamp_min(1e-9).log()).sum(-1).mean().detach().item(),
        'gold_sample_entropy': -(writes * writes.clamp_min(1e-9).log()).sum(-1).mean().detach().item(),
        'read_write_soft_agreement': (qr * writes).sum(-1).detach().mean().item(),
        'query_maxprob': qr.max(-1).values.detach().mean().item(),
        'gold_maxprob': writes.max(-1).values.detach().mean().item(),
    }
    if return_parts:
        return loss, metrics, {'base': base, 'query_negative': query_penalty,
                               'record_negative': record_penalty}
    return loss, metrics


def gradient_diagnostics(parts, model):
    """Read-only all-parameter gradient probe, does not fill .grad or use RNG."""
    params = tuple(model.parameters())
    grads = {}
    for name, term in parts.items():
        gg = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
        grads[name] = torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten()
                                 for p, g in zip(params, gg)])
    norms = {k: float(g.norm()) for k, g in grads.items()}
    result = {'unweighted_gradient_norm': norms}
    for name in ('query_negative', 'record_negative'):
        denominator = norms['base'] * norms[name]
        result[f'base_vs_{name}_cosine'] = (
            float(torch.dot(grads['base'], grads[name])) / denominator
            if denominator > 1e-15 else None)
        result[f'weighted_{name}_to_base_norm_ratio'] = (
            WEIGHT * norms[name] / norms['base'] if norms['base'] > 1e-15 else None)
    denominator = norms['query_negative'] * norms['record_negative']
    result['query_vs_record_cosine'] = (
        float(torch.dot(grads['query_negative'], grads['record_negative'])) / denominator
        if denominator > 1e-15 else None)
    return result
