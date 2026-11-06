"""Prospective LRR-012 algebra, source parity and no-leakage tests."""
import copy
import math
import pytest
import torch
from .lrr008_invariance import InvariantRegister, augmented_batch, heldout, alias_pair, FORBIDDEN, state_digest
from .lrr009_curriculum import predictions, make_bank, payload_hash
from .lrr010_entropy import entropy_objective
from .lrr012_separation import ARMS, separation_objective, negative_agreement, gradient_diagnostics

@pytest.mark.parametrize('mode',['clean','augmentation'])
@pytest.mark.parametrize('arm',ARMS)
def test_zero_coefficient_all_parameter_value_and_gradient_parity(mode,arm):
    torch.set_num_threads(1);torch.manual_seed(774)
    m=InvariantRegister();_,x,y,q,mask,noise=augmented_batch(713,0,2)
    ref,_=entropy_objective(m,x,y,q,mask,noise,mode,0.)
    new,_=separation_objective(m,x,y,q,mask,noise,mode,arm,0.)
    assert torch.allclose(ref,new,rtol=1e-6,atol=1e-6)
    ga=torch.autograd.grad(ref,tuple(m.parameters()),retain_graph=True)
    gb=torch.autograd.grad(new,tuple(m.parameters()))
    for a,b in zip(ga,gb):assert torch.allclose(a,b,atol=1e-6,rtol=1e-5)


def test_negative_agreement_targets_separation_not_uniform_usage():
    a=torch.tensor([[1.,0.,0.,0.]])
    same=a.clone();other=torch.tensor([[0.,1.,0.,0.]])
    big,s=negative_agreement(a,same);small,d=negative_agreement(a,other)
    assert s==1 and d==0 and big>13 and small==0
    uniform=torch.full((1,4),.25)
    u,ss=negative_agreement(uniform,uniform)
    assert abs(u.item()+math.log(.75))<1e-6

@pytest.mark.parametrize('mode',['clean','augmentation'])
def test_record_penalty_extracts_correct_pre_write_states(mode):
    torch.manual_seed(14);m=InvariantRegister()
    examples,x,y,q,mask,noise=augmented_batch(194,2,2)
    used=mask if mode=='augmentation' else torch.zeros_like(mask)
    e,r,g=m.augmented_forward(x,used,noise)
    rows=torch.arange(len(x));gold=r[rows,y]
    direct,_=negative_agreement(gold[0::3],gold[2::3])
    loss,metrics,parts=separation_objective(m,x,y,q,mask,noise,mode,'record_separation',return_parts=True)
    assert torch.allclose(parts['record_negative'],direct,atol=1e-6)
    assert abs(float(loss.detach()-parts['base'].detach())-.2*metrics['record_negative_penalty'])<1e-6
    for i in range(0,len(examples),3):
        a,p,n=examples[i:i+3]
        assert a.words[:-2]==p.words[:-2]==n.words[:-2]
        assert a.subject==n.subject and a.answer==p.answer!=n.answer
        assert a.family!=n.family
        assert all(alias_pair(t) not in FORBIDDEN for t in (a,p,n))
        for ex in (a,p,n):
            latest=max(f[-1] for f in ex.facts if f[0]==ex.subject and f[1]==ex.family)
            assert latest==ex.answer


def test_strength_control_doubles_only_query_negative_term():
    torch.manual_seed(19);m=InvariantRegister()
    _,x,y,q,mask,noise=augmented_batch(191,0,2)
    base,_=separation_objective(m,x,y,q,mask,noise,'augmentation','control')
    new,metrics=separation_objective(m,x,y,q,mask,noise,'augmentation','query_separation')
    assert abs(float((new-base).detach())-.2*metrics['query_negative_penalty'])<1e-6

@pytest.mark.parametrize('arm',ARMS)
def test_gradient_logging_does_not_change_training_update_or_rng(arm):
    torch.manual_seed(222);first=InvariantRegister();second=copy.deepcopy(first)
    op1=torch.optim.AdamW(first.parameters(),lr=.004)
    op2=torch.optim.AdamW(second.parameters(),lr=.004)
    _,x,y,q,mask,noise=augmented_batch(277,1,2)
    f,_,parts=separation_objective(first,x,y,q,mask,noise,'augmentation',arm,return_parts=True)
    rng=torch.get_rng_state().clone()
    stat=gradient_diagnostics(parts,first)
    assert torch.equal(rng,torch.get_rng_state())
    assert all(p.grad is None for p in first.parameters())
    for name,v in stat.items():
        if 'cosine' in name and v is not None:assert -1.000001<=v<=1.000001
    f.backward();torch.nn.utils.clip_grad_norm_(first.parameters(),2.);op1.step()
    l,_=separation_objective(second,x,y,q,mask,noise,'augmentation',arm)
    l.backward();torch.nn.utils.clip_grad_norm_(second.parameters(),2.);op2.step()
    assert state_digest(first)==state_digest(second)


def test_record_and_query_repulsion_need_not_have_same_gradient():
    torch.manual_seed(31);m=InvariantRegister()
    _,x,y,q,mask,noise=augmented_batch(444,0,2)
    _,_,parts=separation_objective(m,x,y,q,mask,noise,'clean','control',return_parts=True)
    a=torch.autograd.grad(parts['query_negative'],m.code.weight,retain_graph=True)[0]
    b=torch.autograd.grad(parts['record_negative'],m.code.weight)[0]
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    assert not torch.allclose(a,b,atol=1e-8)


def test_runtime_and_parameters_unchanged():
    torch.manual_seed(289);m=InvariantRegister().eval()
    assert sum(p.numel() for p in m.parameters())==4748
    exs=[heldout(889_881+i,7,'regular') for i in range(4)]
    assert [r['pred'] for r in predictions(m,exs)]==[m.route(e.words)[0] for e in exs]
    for ex in exs:
        pred,_=m.route(ex.words)
        assert pred is None or 0<=pred<len(ex.words)-1


def test_inputs_unchanged_and_invalid_pairs_fail_closed():
    bank,hashes=make_bank(113,2)
    for examples,x,y,q,mask,noise in bank:
        assert not mask[torch.arange(len(x)),y].any()
        for arm in ARMS:
            m=InvariantRegister();loss,_=separation_objective(m,x,y,q,mask,noise,'augmentation',arm)
            loss.backward();assert torch.isfinite(loss)
    assert [payload_hash(b) for b in bank]==hashes
    examples,x,y,q,mask,noise=bank[0];bad=y.clone();bad[2]=bad[0]
    with pytest.raises(ValueError):separation_objective(m,x,bad,q,mask,noise,'clean','control')
