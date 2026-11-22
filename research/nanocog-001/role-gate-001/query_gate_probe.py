"""Prospective inference-only gate intervention for native ROLE decoder.

The caller must own exact checkpoint and input custody. Do not interpret
synthetic hooks as learned performance or trainable architecture evidence.
"""
import torch

def paired_query_mask(tokens):
    if tokens.ndim!=2 or tokens.shape[1]!=7 or tokens.shape[0]%2:
        raise ValueError("expected even batch of seven-token prompts")
    if not (torch.equal(tokens[0::2,:6],tokens[1::2,:6]) and
            bool((tokens[0::2,6]!=tokens[1::2,6]).all())):
        raise ValueError("only final query token can differ within a row pair")
    return torch.arange(tokens.shape[0],device=tokens.device)^1

def capture(model,tokens):
    if model.training:
        raise ValueError("frozen eval model required")
    if len(model.mixers)!=2 or any(m.local.window!=4 for m in model.mixers):
        raise ValueError("expected exact two-local-layer mixer geometry")
    scores=[None,None]
    def bind(i):
        def remember(_,args,out):
            if out.shape != (*tokens.shape,4) or not torch.isfinite(out).all():
                raise ValueError("invalid gate scores")
            scores[i]=out.detach().clone()
        return remember
    loops=[m.local.offset_gate.register_forward_hook(bind(i)) for i,m in enumerate(model.mixers)]
    try:
        with torch.inference_mode():
            native,_=model(tokens,state=None)
    finally:
        for h in loops:h.remove()
    if any(s is None for s in scores) or not torch.isfinite(native).all():
        raise ValueError("incomplete native capture")
    return native,scores

def replace_query_gate(model,tokens,scores,layers, donor, override=None):
    """Replay original query embeddings/values; replace pre-softmax gate at t=6."""
    if not isinstance(layers,tuple) or not set(layers)<=set((0,1)) or len(set(layers))!=len(layers):
        raise ValueError("unknown layer branch")
    def bind(i):
        def swap(_,args,out):
            update=out.clone()
            update[:,-1]=scores[i][donor,-1] if override is None else override
            return update
        return swap
    hooks=[model.mixers[i].local.offset_gate.register_forward_hook(bind(i)) for i in layers]
    try:
        with torch.inference_mode():
            logits,_=model(tokens,state=None)
    finally:
        for h in hooks:h.remove()
    if not torch.isfinite(logits).all():
        raise ValueError("nonfinite intervention")
    return logits

def centered_sham(native,opposite):
    delta=opposite-native
    delta=delta-delta.mean(-1,keepdim=True)
    rolls=torch.stack([delta.roll(k,dims=-1) for k in (1,2,3)],dim=1)
    norm_sq=(delta.square()).sum(-1)
    cosine=(rolls*delta[:,None]).sum(-1).abs()/norm_sq[:,None].clamp_min(1e-20)
    ix=cosine.argmin(-1)
    sham=rolls[torch.arange(delta.size(0),device=delta.device),ix]
    torch.testing.assert_close(sham.norm(dim=-1),delta.norm(dim=-1),atol=1e-7,rtol=1e-6)
    return native+sham,delta,cosine.min(-1).values

def probe(model,tokens):
    opposite=paired_query_mask(tokens)
    self_rows=torch.arange(tokens.shape[0],device=tokens.device)
    native,scores=capture(model,tokens)
    control=replace_query_gate(model,tokens,scores,(1,),self_rows)
    if not torch.equal(control,native):
        raise ValueError("same-row donor no-op lost exact parity")
    original=scores[1][:,-1]
    donor=original[opposite]
    fake,dose,cos=centered_sham(original,donor)
    branches={"native":native[:,-1]}
    for name,override in (("opposite",donor),("sham",fake)):
        changed=replace_query_gate(model,tokens,scores,(1,),opposite,override=override)
        if not torch.equal(changed[:,:6],native[:,:6]):
            raise ValueError("intervention changed earlier causal outputs")
        branches[name]=changed[:,-1]
    with torch.inference_mode():
        replay,_=model(tokens,state=None)
    if not torch.equal(replay,native):
        raise ValueError("hook removal changed native output")
    return dict(logits=branches, diagnostics=dict(
        score_l2=dose.norm(dim=-1), sham_abs_cos=cos,
        opposite_prob_l2=(donor.softmax(-1)-original.softmax(-1)).norm(dim=-1),
        sham_prob_l2=(fake.softmax(-1)-original.softmax(-1)).norm(dim=-1),
    ))
