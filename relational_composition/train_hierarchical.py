"""Selective discrete credit for page-addressed relational models.

For an ACTIVE binary page bit, the base and one-bit counterfactual enumerate
both possible bit values. Holding other policy Gumbel draws fixed gives an
exact *conditional Bernoulli* loss difference for that intervention. The
surrogate `p(bit=1)*(L_1-L_0)` differentiates that conditional expectation.
It is not an unbiased gradient of the full joint hard-routing objective,
because alternative downstream trajectories depend on the intervened bit.
For other multiway actions we retain a biased pairwise local surrogate.
"""
from __future__ import annotations
import random
import torch
from torch.nn import functional as F


def _loss(output, targets, mask):
    if mask is not None:
        if mask.shape != targets.shape:
            raise ValueError('answer supervision shape mismatch')
        targets = torch.where(mask,targets,0)
    if not (targets!=0).any():
        raise ValueError('no valid training targets')
    return F.cross_entropy(output.logits.flatten(0,1),targets.flatten(),ignore_index=0)


def focused_positions(x, answer_mask, *, writer):
    """Focus within each document's own answer window, never another row's."""
    if answer_mask.shape != x.shape:
        raise ValueError('answer supervision shape mismatch')
    result = []
    for row in range(x.shape[0]):
        positions = answer_mask[row].nonzero().flatten().tolist()
        if not positions:
            continue
        first, last = positions[0], positions[-1]
        for t in (x[row] != 0).nonzero().flatten().tolist():
            eligible = t < first if writer else first - 2 <= t <= last
            if eligible:
                result.append((row, t))
    return result


def hierarchical_credit_step(model, optimizer, x, y, *, answer_mask=None,
                             seed: int, focus_prob: float=.85,
                             page_priority: float=.5, credit_weight: float=1.0):
    if not 0 <= focus_prob <= 1 or not 0 <= page_priority <= 1:
        raise ValueError('probabilities out of range')
    model.train()
    optimizer.zero_grad(set_to_none=True)
    base=model(x,sample=True)
    loss=_loss(base,y,answer_mask)
    rng=random.Random(seed)
    eligible=[]
    def positions_for(key):
        action=base.decisions[key]
        coords=[]
        position_list=(x!=0).nonzero().tolist()
        if (answer_mask is not None and rng.random()<focus_prob
                and bool(answer_mask.any())):
            position_list = focused_positions(x, answer_mask,
                writer=key[1].startswith('write_'))
        for row,t in position_list:
            if action.ndim==2:
                coords.append((row,t))
            else:
                if key[1].startswith('page_'):
                    active=base.active_page_bits[key][row,t]
                    for bit in active.nonzero().flatten().tolist():
                        coords.append((row,t,bit))
                else:
                    coords.extend((row,t,j) for j in range(action.shape[-1]))
        return coords
    for key,policies in base.policies.items():
        if policies.shape[-1]>1:
            candidates=positions_for(key)
            if candidates:
                eligible.append((key,candidates))
    policy_loss=loss.new_zeros(())
    result={'loss':float(loss.detach()),'policy_loss':0.0,'field':None,
            'interventions':0,'advantage':0.0,'binary_conditional':False}
    if credit_weight and eligible:
        page_fields=[z for z in eligible if z[0][1].startswith('page_')]
        pool=page_fields if page_fields and rng.random()<page_priority else eligible
        key,coords=rng.choice(pool)
        coord=rng.choice(coords)
        logits=base.policies[key][coord]
        chosen=int(base.decisions[key][coord])
        alt=rng.randrange(logits.numel()-1)
        alt+=int(alt>=chosen)
        changed=base.decisions[key].clone()
        changed[coord]=alt
        with torch.no_grad():
            alternative=model(x,sample=True,noise=base.noise,overrides={key:changed})
            alternative_loss=_loss(alternative,y,answer_mask)
        advantage=loss.detach()-alternative_loss
        if key[1].startswith('page_'):
            # For a binary action these two trajectories exhaust its choices.
            # This computes the derivative of p_1*L1 + (1-p_1)*L0 with
            # L0/L1 treated as constants w.r.t. the intervened policy logits.
            diff=(alternative_loss-loss.detach()) if chosen==0 else (
                  loss.detach()-alternative_loss)
            policy_loss=diff*logits.softmax(-1)[1]
            result['binary_conditional']=True
        else:
            diff=advantage.clamp(-1.,1.)
            logp=logits.log_softmax(-1)
            policy_loss=diff*(logp[chosen]-logp[alt])
        result.update(field=str(key),coordinate=coord,chosen=chosen,alternative=alt,
                      interventions=1,advantage=float(advantage),
                      alternative_loss=float(alternative_loss))
    objective=loss+credit_weight*policy_loss
    objective.backward()
    gradient_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    if not torch.isfinite(gradient_norm):
        raise FloatingPointError('nonfinite gradient; step cancelled')
    optimizer.step()
    result.update(policy_loss=float(policy_loss.detach()),
                  gradient_norm=float(gradient_norm),
                  forward_passes=2 if result['interventions'] else 1)
    return result
