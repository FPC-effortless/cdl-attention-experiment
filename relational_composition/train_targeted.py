"""Optional answer-focused counterfactual training experiment (not yet validated).

Answer masks are *training-side supervision* only. No future token, answer
boundary or oracle edge endpoint is supplied to either neural forward pass.
One uniformly random action/coordinate is preserved 15% of the time as a
coverage diagnostic; the remaining proposals focus on query-answer positions,
while writer interventions are allowed anywhere in the preceding prefix.

This objective is a BIASED, one-coordinate heuristic. It is not an unbiased
score-function estimator or an inference competence result.
"""
import random
import torch
from torch.nn import functional as F


def _masked_ce(output, y, answer_mask):
    targets = torch.where(answer_mask, y, 0) if answer_mask is not None else y
    if not (targets != 0).any():
        raise ValueError('no non-padding target positions')
    return F.cross_entropy(output.logits.flatten(0, 1), targets.flatten(), ignore_index=0)


def targeted_counterfactual_step(model, optimizer, x, y, *, answer_mask,
                                 seed: int, focus_prob: float = .85,
                                 credit_weight: float = 1.0,
                                 decision_key=None):
    if answer_mask is not None and answer_mask.shape != y.shape:
        raise ValueError('answer mask must match labels')
    if not 0 <= focus_prob <= 1:
        raise ValueError('invalid focus probability')
    model.train()
    optimizer.zero_grad(set_to_none=True)
    base = model(x, sample=True)
    nll = _masked_ce(base, y, answer_mask)
    rng = random.Random(seed)
    keys = [k for k,p in base.policies.items() if p.shape[-1] > 1]
    receipt = {'nll': float(nll.detach()), 'advantage': 0.0,
               'interventions': 0, 'focused': False}
    policy_loss = nll.new_zeros(())
    if credit_weight and keys:
        key = rng.choice(keys) if decision_key is None else decision_key
        if key not in keys:
            raise ValueError('invalid decision field')
        coordinates = (x != 0).nonzero().tolist()
        focused = False
        if answer_mask is not None and rng.random() < focus_prob:
            selected = []
            for row,t in coordinates:
                answer_positions = answer_mask[row].nonzero().flatten().tolist()
                if not answer_positions:
                    continue
                first_answer_pred = answer_positions[0]
                if key[1].startswith('write_'):
                    # A write can causally alter answer reads only if it precedes them.
                    if t < first_answer_pred:
                        selected.append([row,t])
                else:
                    # Include the end of query and teacher-forced answer decisions.
                    if first_answer_pred - 2 <= t <= answer_positions[-1]:
                        selected.append([row,t])
            if selected:
                coordinates = selected
                focused = True
        row,t = rng.choice(coordinates)
        base_actions = base.decisions[key]
        coord = ((row,t) if base_actions.ndim==2 else
                 (row,t,rng.randrange(base_actions.shape[-1])))
        chosen = int(base_actions[coord])
        logits = base.policies[key][coord]
        alternative = rng.randrange(logits.numel()-1)
        alternative += int(alternative >= chosen)
        changed = base_actions.clone()
        changed[coord] = alternative
        with torch.no_grad():
            replay = model(x,sample=True,noise=base.noise,overrides={key:changed})
            alternative_loss = _masked_ce(replay,y,answer_mask)
        advantage = (nll.detach()-alternative_loss).clamp(-1,1)
        logp = logits.log_softmax(-1)
        policy_loss = advantage * (logp[chosen]-logp[alternative])
        receipt.update(advantage=float(advantage),interventions=1,
                       focused=focused, field=str(key), coordinate=coord,
                       chosen=chosen, alternative=alternative,
                       alternative_nll=float(alternative_loss))
    objective = nll + credit_weight * policy_loss
    objective.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    if not torch.isfinite(norm):
        raise FloatingPointError('nonfinite gradient')
    optimizer.step()
    receipt.update(policy_loss=float(policy_loss.detach()),grad_norm=float(norm),
                   full_forward_passes=2 if receipt['interventions'] else 1)
    return receipt
