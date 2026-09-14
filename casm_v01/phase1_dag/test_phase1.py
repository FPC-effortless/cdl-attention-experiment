import torch
from .generator import BooleanDAGGenerator
from .model import CASMS, StaticMask
from .oracle import exhaustive_truth_table, locally_nonredundant
from .diagnostics import gate0_viability, gate6_integrity, gate3_structural_sensitivity
from .runner import make_structural_pair

def test_generator_has_real_distractors_and_exhaustive_oracle():
    ep=BooleanDAGGenerator(max_nodes=8,min_nodes=4,seed=7).sample()
    assert len(ep.existence_mask)==8
    assert all(e.src<e.dst for e in ep.candidate_edges)
    assert set(ep.true_edges).issubset(set(ep.candidate_edges))
    assert len(ep.candidate_edges)>len(ep.true_edges)
    assert exhaustive_truth_table(ep)==ep.truth_table
    assert locally_nonredundant(ep)

def test_router_cannot_see_runtime_values():
    ep=BooleanDAGGenerator(max_nodes=8,min_nodes=4,seed=9).sample(); model=CASMS(max_nodes=8,dim=16,temperature=2.0,seed=9)
    x=torch.tensor([ep.input_values],dtype=torch.float32,requires_grad=True)
    _,g,_=model([ep],x)
    assert torch.autograd.grad(g.sum(),x,allow_unused=True)[0] is None

def test_gate_and_integrity_contract():
    eps=BooleanDAGGenerator(max_nodes=8,min_nodes=4,seed=11).sample_batch(8); model=CASMS(max_nodes=8,dim=16,temperature=2.0,seed=11)
    assert gate0_viability(model,eps)["pass"]
    assert gate6_integrity(model,eps)["pass"]
    assert tuple(model.alpha_eta.shape)==(2,)
    assert not hasattr(model,"alpha_ij")

def test_static_and_casm_have_distinct_routing_parameterizations():
    assert hasattr(StaticMask(),"edge_logits")
    assert hasattr(CASMS(),"q") and hasattr(CASMS(),"k")

def test_gate3_pair_is_not_a_rewire_only_pair():
    gen=BooleanDAGGenerator(max_nodes=10,min_nodes=4,seed=20260914)
    train=gen.sample_batch(128)
    a,b=make_structural_pair(train,20260915)
    assert a.active_count==b.active_count
    assert len(a.inputs)==len(b.inputs)
    assert tuple(n.depth for n in a.nodes[:a.active_count])==tuple(n.depth for n in b.nodes[:b.active_count])
    assert tuple(n.op for n in a.nodes[:a.active_count])!=tuple(n.op for n in b.nodes[:b.active_count])
    model=CASMS(max_nodes=10,dim=16,temperature=2.0,seed=1)
    # At initialization, distinct structural inputs must be capable of
    # producing distinct logits; this catches accidental rewire-only tests.
    diff=gate3_structural_sensitivity(model,a,b)["mean_gate_difference"]
    assert diff>1e-6
