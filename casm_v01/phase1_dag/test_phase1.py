import torch
from .generator import BooleanDAGGenerator
from .model import CASMS, StaticMask
from .oracle import exhaustive_truth_table, locally_nonredundant
from .pairs import rewire_episode
from .diagnostics import gate0_viability, gate3_structural_sensitivity, gate6_integrity

def test_generator_has_real_distractors_and_exhaustive_oracle():
    ep=BooleanDAGGenerator(max_nodes=8,min_nodes=4,seed=7).sample()
    assert len(ep.existence_mask)==8
    assert len(ep.parent_slots)==8
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

def test_structural_pair_changes_router_input_and_gates():
    ep=BooleanDAGGenerator(max_nodes=8,min_nodes=4,seed=13).sample(); pair=rewire_episode(ep,seed=17); model=CASMS(max_nodes=8,dim=16,temperature=2.0,seed=13)
    assert pair.parent_slots != ep.parent_slots
    _,g,_=model([ep,pair],torch.tensor([ep.input_values,pair.input_values],dtype=torch.float32))
    assert float((g[0,:min(len(ep.candidate_edges),len(pair.candidate_edges))]-g[1,:min(len(ep.candidate_edges),len(pair.candidate_edges))]).abs().mean())>1e-4

def test_gate_and_integrity_contract():
    eps=BooleanDAGGenerator(max_nodes=8,min_nodes=4,seed=11).sample_batch(8); model=CASMS(max_nodes=8,dim=16,temperature=2.0,seed=11)
    assert gate0_viability(model,eps)["pass"]
    assert gate6_integrity(model,eps)["pass"]
    assert tuple(model.alpha_eta.shape)==(2,)
    assert torch.allclose(model.c*torch.nn.functional.softplus(model.alpha_eta),torch.ones(2))
    assert not hasattr(model,"alpha_ij")

def test_static_and_casm_have_distinct_routing_parameterizations():
    assert hasattr(StaticMask(),"edge_logits")
    assert hasattr(CASMS(),"q") and hasattr(CASMS(),"k")
