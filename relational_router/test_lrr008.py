import torch
from .lrr005_register import CausalRoleRegister, FILLERS, ID, vectorize
from .lrr008_invariance import *

def test_all_training_queries_heldout_safe_and_latest():
    for n in range(500):
        a,p,b=triple(81000000+n,n%20,n%8)
        assert a.answer==p.answer!=b.answer
        assert a.words[:-2]==p.words[:-2]==b.words[:-2]
        for ex in (a,p,b):
            assert alias_pair(ex) not in FORBIDDEN
            assert latest_records(ex)[(ex.subject,ex.family)][-1]==ex.answer

def test_noise_touches_only_train_fillers():
    examples,x,y,q,mask,noise=augmented_batch(0,0)
    assert x.shape[0]==48
    assert not mask[:24].any() and mask[24:].any()
    for i,ex in enumerate(examples):
        for j in mask[i].nonzero().flatten().tolist():assert ex.words[j] in FILLERS[:3]
    assert not mask[torch.arange(48),y].any()
    assert torch.equal(noise[24],noise[25]) and torch.equal(noise[25],noise[26])

def test_no_noise_equivalent_to_original_forward_and_gradients():
    torch.manual_seed(3);m=InvariantRegister()
    _,x,y,q,mask,noise=augmented_batch(0,1,histories=1)
    a=CausalRoleRegister.forward(m,x);b=m.augmented_forward(x,torch.zeros_like(mask),noise)
    for v,w in zip(a,b):assert torch.allclose(v,w,atol=1e-7)
    ga=torch.autograd.grad(a[1].square().sum(),m.code.weight,retain_graph=True)[0]
    gb=torch.autograd.grad(b[1].square().sum(),m.code.weight)[0]
    assert torch.allclose(ga,gb,atol=1e-7)
    assert sum(p.numel() for p in m.parameters())==4748

def test_paired_inputs_and_exact_added_penalty():
    torch.manual_seed(2);m=InvariantRegister()
    _,x,y,q,mask,noise=augmented_batch(0,2,histories=1)
    a,_=objective(m,x,y,q,mask,noise,"augmentation")
    b,met=objective(m,x,y,q,mask,noise,"invariance")
    assert abs((b-a).item()-.5*met["invariance"])<1e-5
    assert met["invariance"]>=0
    b.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())

def test_transform_preserves_values_positions_and_heldout_pair():
    for s in range(25):
        a=heldout(9900000+s,32)
        for c in MAPS:
            b=heldout(9900000+s,32,c)
            assert len(a.words)==len(b.words) and a.answer==b.answer
            assert a.words[a.answer]==b.words[b.answer] and alias_pair(b)==TEST_PAIR
            assert a.facts==b.facts
        assert alias_pair(heldout(9900000+s,32,split="dev"))==DEV_PAIR

def test_batch_route_matches_source_and_causal_prefix():
    torch.manual_seed(8);m=InvariantRegister().eval()
    cases=[heldout(9100+i,n,c) for i,(n,c) in enumerate(((7,"regular"),(32,"oov"),(128,"syntax_shift")))]
    pred=hard_batch(m,cases)
    for ex,r in zip(cases,pred):assert r["pred"]==m.route(ex.words)[0]
    p=torch.tensor([[ID.get(w,1) for w in cases[0].words[:12]]]);q=torch.cat((p,torch.tensor([[3,4,5]])),1)
    with torch.no_grad():
        a=m(p);b=m(q)
        for v,w in zip(a,b):assert torch.allclose(v,w[:,:12],atol=1e-6)

def test_symbolic_latest_record_ceiling():
    s=SymbolicIndex()
    for n in range(200):
        for ex in triple(781000+n,n,0):s.observe(ex)
    for i in range(10):
        for c in (*MAPS,"syntax_shift"):
            ex=heldout(801000+i,32,c);assert s.route(ex)==ex.answer

def test_training_two_steps_preserves_actual_optimizer_and_counts(tmp_path):
    for arm in ARMS:
        m,r=fit(0,arm,steps=2,histories=1,outdir=tmp_path)
        assert r["optimizer_step_max"]==r["optimizer_step_min"]==2
        assert r["question_view_exposures"]==12
