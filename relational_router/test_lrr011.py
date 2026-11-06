import copy,math
import pytest
import torch
from .lrr008_invariance import InvariantRegister,augmented_batch,heldout,alias_pair,FORBIDDEN
from .lrr009_curriculum import predictions,make_bank,payload_hash,schedule
from .lrr010_entropy import entropy_objective
from .lrr011_binding import auxiliary_terms,binding_objective,ARMS

@pytest.mark.parametrize('mode',['clean','augmentation'])
@pytest.mark.parametrize('arm',ARMS)
def test_zero_new_coefficient_value_and_every_parameter_gradient_parity(mode,arm):
    torch.set_num_threads(1);torch.manual_seed(783)
    m=InvariantRegister();_,x,y,q,mask,noise=augmented_batch(773,0,2)
    ref,_=entropy_objective(m,x,y,q,mask,noise,mode,0.)
    new,_=binding_objective(m,x,y,q,mask,noise,mode,arm,0.)
    assert torch.allclose(ref,new,rtol=1e-6,atol=1e-6)
    ga=torch.autograd.grad(ref,tuple(m.parameters()),retain_graph=True)
    gb=torch.autograd.grad(new,tuple(m.parameters()))
    for a,b in zip(ga,gb):assert torch.allclose(a,b,atol=1e-6,rtol=1e-5)

def test_agreement_is_not_just_sharpness():
    a=torch.tensor([[1.,0.,0.,0.]],requires_grad=True)
    equal=torch.tensor([[1.,0.,0.,0.]])
    wrong=torch.tensor([[0.,1.,0.,0.]])
    b,c,agr=auxiliary_terms(a,equal);assert b==0 and c==0 and agr==1
    b,c,agr=auxiliary_terms(a,wrong);assert b>20 and c==0 and agr==0
    uniform=torch.full((1,4),.25)
    b,c,agr=auxiliary_terms(uniform,uniform)
    assert abs(b.item()-math.log(4))<1e-6 and abs(c.item()-math.log(4))<1e-6

def test_disjoint_soft_addresses_receive_different_binding_signal():
    x=torch.tensor([[2.,0.,-1.]],requires_grad=True)
    y=torch.tensor([[0.,2.,-1.]],requires_grad=True)
    b,c,_=auxiliary_terms(x.softmax(-1),y.softmax(-1))
    gb=torch.autograd.grad(b,(x,y),retain_graph=True)
    gc=torch.autograd.grad(c,(x,y))
    assert all(torch.isfinite(g).all() for g in gb+gc)
    assert not torch.allclose(gb[0],gc[0])

@pytest.mark.parametrize('arm,metric',[('read_write','binding_penalty'),('self_confidence','confidence_penalty')])
def test_only_declared_term_added(arm,metric):
    torch.manual_seed(997);m=InvariantRegister()
    _,x,y,q,mask,noise=augmented_batch(913,0,2)
    base,_=binding_objective(m,x,y,q,mask,noise,'augmentation','control')
    new,diag=binding_objective(m,x,y,q,mask,noise,'augmentation',arm)
    assert abs(float((new-base).detach())-.2*diag[metric])<1e-6

def test_original_runtime_and_parameter_count():
    torch.manual_seed(88);m=InvariantRegister()
    assert sum(p.numel() for p in m.parameters())==4748
    examples=[heldout(776_000+i,7,'regular') for i in range(4)]
    assert [d['pred'] for d in predictions(m,examples)]==[m.route(e.words)[0] for e in examples]
    for e in examples:
        a,_=m.route(e.words);assert a is None or a<len(e.words)-1

def test_bank_unmodified_and_no_extra_label_paths():
    bank,hashes=make_bank(7987,2)
    for examples,x,y,q,mask,noise in bank:
        assert all(alias_pair(e) not in FORBIDDEN for e in examples)
        assert not mask[torch.arange(len(x)),y].any()
        for arm in ARMS:
            m=InvariantRegister();loss,_=binding_objective(m,x,y,q,mask,noise,'augmentation',arm)
            loss.backward();assert torch.isfinite(loss)
    assert [payload_hash(b) for b in bank]==hashes

def test_binding_diagnostics_reproduce_state_and_original_runtime():
    from .run_lrr011 import read_write_diagnostics
    torch.manual_seed(442);m=InvariantRegister().eval()
    exs=[heldout(990_412+i,7,'regular') for i in range(3)]
    out=read_write_diagnostics(m,exs)
    from .lrr005_register import vectorize
    x,y,q=vectorize(exs)
    with torch.no_grad():
        e,r,g=m(x)
    for i,row in enumerate(out):
        assert abs(row['soft_read_write_agreement']-float((r[i,y[i]]*r[i,q[i]]).sum()))<1e-6
        assert 0<=row['query_maxprob']<=1 and 0<=row['gold_maxprob']<=1
        assert row['soft_pointer'] is None or row['soft_pointer']<q[i]
