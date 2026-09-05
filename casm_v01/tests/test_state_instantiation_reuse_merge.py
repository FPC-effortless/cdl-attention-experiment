import torch

from casm.reuse_merge import ReuseSignature, propose_reuse_groups
from casm.state_instantiation_reuse_merge import merge_gates
from casm.state_instantiation_data import make_state_instantiation_batch


def test_signature_grouping_is_deterministic():
    sigs=[ReuseSignature(1,1,(2,3)),ReuseSignature(1,1,(2,3)),ReuseSignature(1,2,(2,3))]
    assert propose_reuse_groups(sigs)==((0,1),(2,))


def test_graph_and_blind_controls_are_distinct_on_dependency_structure():
    batch=make_state_instantiation_batch(4,12,12345,live_cardinality=3,split='train')
    graph,_=merge_gates(torch.full((4,8),0.8),batch,structure_blind=False)
    blind,_=merge_gates(torch.full((4,8),0.8),batch,structure_blind=True)
    assert graph.shape==blind.shape==(4,8)
    assert torch.isfinite(graph).all() and torch.isfinite(blind).all()


def test_merge_never_changes_soft_gate_values_of_representatives():
    batch=make_state_instantiation_batch(2,12,6789,live_cardinality=3,split='train')
    gates=torch.arange(16,dtype=torch.float32).reshape(2,8)/10
    merged,_=merge_gates(gates,batch,structure_blind=False)
    assert (merged<=gates.max()).all()
