"""LRR-007 invariants: known label privileges, disjoint pairs and baseline parity."""
import random
import torch
from torch.nn import functional as F

from .lrr005_register import (
    CausalRoleRegister, FAMILIES, make_example, vectorize
)
from .lrr007_dual_credit import (
    TRAIN_FORBIDDEN, DEV_PAIR, TEST_PAIR,
    actual_alias_pair, training_triple, heldout_example,
    paired_objective, fit_arm,
)


def test_no_reserved_pair_in_any_of_three_training_questions():
    for seed in range(60):
        for step in (0,7,120):
            a,p,n=training_triple(12_000_000+seed*1000+step,step,seed)
            assert a.answer==p.answer!=n.answer
            assert a.question_alias!=p.question_alias
            assert a.words[:-2]==p.words[:-2]==n.words[:-2]
            for ex in (a,p,n):
                assert actual_alias_pair(ex) not in TRAIN_FORBIDDEN
                assert ex.words[ex.answer].startswith("value")


def test_dev_and_test_splits_actual_fact_surface():
    for layout in ("regular","shift","novel_fillers"):
        for seed in range(36):
            for split,wanted in (("dev",DEV_PAIR),("test",TEST_PAIR)):
                e=heldout_example(30_000_000+seed,32,split=split,layout=layout)
                assert actual_alias_pair(e)==wanted
                gold=next(f for f in e.facts if f[-1]==e.answer)
                assert gold[2]==FAMILIES[e.family][wanted[0]]
                assert e.question_alias==FAMILIES[e.family][wanted[1]]
                assert e.words[e.answer]==gold[3]


def test_control_matches_original_lrr005_loss_and_gradient():
    torch.manual_seed(9)
    model=CausalRoleRegister()
    ex=[]
    triples=[]
    for i in range(4):
        triples.append((3*i,3*i+1,3*i+2))
        ex.extend(training_triple(40000+i,4,i))
    x,y,q=vectorize(ex)
    neg_pairs=[(a,c) for a,b,c in triples]
    old,diag=model.answer_loss(x,y,q,pairs=neg_pairs,credit=.2)
    new,metrics=paired_objective(model,x,y,q,triples,arm="negative_only")
    assert torch.allclose(new,old,atol=1e-6,rtol=1e-6)
    a=torch.autograd.grad(old,model.code.weight,retain_graph=True)[0]
    b=torch.autograd.grad(new,model.code.weight)[0]
    assert torch.allclose(a,b,atol=1e-5,rtol=1e-5)
    assert 0<=metrics["same_relation_soft_agreement"]<=1


def test_added_positive_term_is_exact_and_has_gradient():
    torch.manual_seed(3)
    model=CausalRoleRegister()
    triples=[training_triple(50000+i,7,i) for i in range(3)]
    x,y,q=vectorize([e for triplet in triples for e in triplet])
    idx=[(3*i,3*i+1,3*i+2) for i in range(3)]
    control,_=paired_objective(model,x,y,q,idx,arm="negative_only")
    dual,met=paired_objective(model,x,y,q,idx,arm="two_sided")
    expected=.2*met["positive_alignment_penalty"]
    assert abs(float((dual-control).detach())-expected)<1e-5
    assert expected>=0
    dual.backward()
    assert model.code.weight.grad is not None
    assert torch.isfinite(model.code.weight.grad).all()


def test_hard_inference_is_causal_and_does_not_accept_answer_position():
    torch.manual_seed(5)
    model=CausalRoleRegister()
    case=heldout_example(88667,32,split="test",layout="shift")
    pred,meta=model.route(case.words)
    assert pred is None or 0<=pred<len(case.words)-1
    assert meta["index_probes"]==1
    assert "answer" not in model.route.__code__.co_varnames
    assert sum(p.numel() for p in model.parameters())==4748
    # Changing a suffix after an existing token must not change past state.
    prefix=case.words[:15]
    tail=prefix+("mira","owns","something",".")
    ids1=torch.tensor([[__import__("relational_router.lrr005_register",fromlist=["ID"]).ID.get(w,1) for w in prefix]])
    ids2=torch.tensor([[__import__("relational_router.lrr005_register",fromlist=["ID"]).ID.get(w,1) for w in tail]])
    with torch.no_grad():
        e1,r1,g1=model(ids1)
        e2,r2,g2=model(ids2)
    assert torch.allclose(e1,e2[:,:len(prefix)])
    assert torch.allclose(r1,r2[:,:len(prefix)])


def test_two_update_smoke_has_parameter_and_data_parity():
    configs=[]
    for arm in ("negative_only","two_sided","energy_rank"):
        model,meta=fit_arm(4,arm,steps=2,triples_per_update=2)
        assert meta["steps"]==2
        assert meta["train_questions"]==12
        assert meta["trainable_parameters"]==4748
        assert torch.isfinite(torch.tensor(meta["last_loss"]))
        configs.append(meta["source_histories"])
    assert configs==[4,4,4]
