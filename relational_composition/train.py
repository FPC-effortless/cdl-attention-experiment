"""Bounded one-coordinate counterfactual training, including writer decisions.

This is a biased local policy-improvement surrogate, not an unbiased estimator
of end-to-end discrete LM gradients. Other actions are recomputed with the same
Gumbel noise, allowing continuation changes to affect later discrete queries.
It supplies no graph/identity/route labels and claims no CASM reproduction.
"""
import random

import torch

from .model import language_loss


def counterfactual_step(model, optimizer, x, y, *, seed, credit_weight=1.0,
                        decision_key=None):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    base = model(x, sample=True)
    nll = language_loss(base, y)
    rng = random.Random(seed)
    keys = [k for k, logits in base.policies.items() if logits.shape[-1] > 1]
    policy_loss = nll.new_zeros(())
    cost = dict(base.metrics)
    receipt = dict(forward_passes=1, interventions=0, advantage=0.0, decision=None)
    if keys and credit_weight:
        key = decision_key if decision_key is not None else rng.choice(keys)
        if key not in keys:
            raise ValueError("unknown or singleton decision")
        row, t = rng.choice((x != 0).nonzero().tolist())
        action_tensor = base.decisions[key]
        coord = (row, t) if action_tensor.ndim == 2 else (row, t, rng.randrange(action_tensor.shape[2]))
        logits = base.policies[key][coord]
        chosen = int(action_tensor[coord])
        alternative = rng.randrange(logits.shape[-1] - 1)
        alternative += int(alternative >= chosen)
        changed = action_tensor.clone()
        changed[coord] = alternative
        with torch.no_grad():
            counter = model(x, sample=True, noise=base.noise, overrides={key: changed})
            alternative_nll = language_loss(counter, y)
        advantage = (nll.detach() - alternative_nll).clamp(-1.0, 1.0)
        logp = logits.log_softmax(-1)
        # Better base loss => negative advantage => increase base log-probability.
        policy_loss = advantage * (logp[chosen] - logp[alternative])
        cost = {k: v + counter.metrics[k] for k, v in cost.items()}
        receipt.update(forward_passes=2, interventions=1, advantage=float(advantage),
                       decision=[key[0], key[1]], coordinate=list(coord),
                       chosen=chosen, alternative=alternative,
                       alternative_nll=float(alternative_nll))
    objective = nll + credit_weight * policy_loss
    objective.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(norm):
        raise FloatingPointError("nonfinite gradient; optimizer step cancelled")
    optimizer.step()
    return dict(nll=float(nll.detach()), policy_loss=float(policy_loss.detach()),
                gradient_norm=float(norm), cost=cost, **receipt)
