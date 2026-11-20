import json
import numpy as np
from run import (ROOT, action_mass, compatible, dataset, features, noop_gradient,
                 predict, proposals, shortlist)


def gradient(probs, actions, mask):
    mass = action_mass(probs, actions, 32)*mask
    total = mass.sum(1,keepdims=True)
    posterior = np.divide(mass,total,out=np.zeros_like(mass),where=total>0)
    gradients = []
    for p, idx in zip(probs,(actions//32%32,actions%32,actions//1024)):
        delta = p * (total>0)
        np.add.at(delta,(np.arange(len(p))[:,None],idx),-posterior)
        gradients.append(delta)
    return np.concatenate(gradients,axis=1)


rows = []
for seed in (13,37,61):
    spec = json.loads((ROOT/f'evidence/data-32-{seed}.json').read_text())
    raw,state,target,_ = dataset(seed+8600,1024,32,np.array(spec['aliases']),'held_pair')
    keep = (state==target).all(1)
    x = features(raw,spec['vocabulary'])[0][keep]
    state = state[keep]
    with np.load(ROOT/f'evidence/weighted16-32-{seed}.npz') as ck:
        probs = predict([ck[k] for k in ck.files],x)
    exact = np.concatenate(noop_gradient(probs,state),axis=1)
    dense = np.broadcast_to(np.arange(2048),(len(x),2048))
    ref = gradient(probs,dense,compatible(dense,state,state))
    diff = float(np.max(abs(exact-ref)))
    assert diff < 1e-12
    top = shortlist(probs,32)
    draws = []
    rng = np.random.default_rng(seed+9900)
    for _ in range(64):
        actions,mask,_ = proposals(rng,'weighted16',top,compatible(top,state,state),state,state)
        draws.append(gradient(probs,actions,mask))
    rows.append(dict(seed=seed,n=len(x),max_error=diff,
                     sampled_variance_trace=float(np.var(draws,axis=0,ddof=1).sum(1).mean()),
                     analytic_variance_trace=0.0))
(ROOT/'gradient.json').write_text(json.dumps(rows,indent=2))
