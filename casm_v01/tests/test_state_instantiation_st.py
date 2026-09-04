from __future__ import annotations

from dataclasses import replace

import torch

from casm.state_instantiation_data import NUM_CANDIDATES, make_state_instantiation_batch
from casm.state_instantiation_model import StateInstantiationModel
from casm.state_instantiation_st import (
    SOFT_MODE,
    ST_BLIND_MODE,
    ST_GRAPH_MODE,
    X20_FROZEN_RESULT,
    X20RStateInstantiationModel,
    cloned_x20r_models,
    straight_through_binary,
)


def test_x20r_base_is_exact_frozen_x20_result():
    assert X20_FROZEN_RESULT == "3225172c78ca44ad57a26d64b13ae24f122b96bb"


def test_soft_replication_is_exact_x20_when_weights_match():
    torch.manual_seed(101)
    old = StateInstantiationModel(mode="learned_instantiation", d_model=32)
    new = X20RStateInstantiationModel(mode=SOFT_MODE, d_model=32)
    new.load_state_dict(old.state_dict())
    batch = make_state_instantiation_batch(4, 12, 1801, live_cardinality=3, split="train")
    assert torch.equal(old.gates(batch), new.soft_gates(batch))
    old_parts = old.loss_components(batch)
    new_parts = new.loss_components(batch)
    assert old_parts.keys() == new_parts.keys()
    for key in old_parts:
        assert torch.equal(old_parts[key], new_parts[key]), key


def test_all_learned_x20r_regimes_start_bit_identically_and_match_parameter_count():
    torch.manual_seed(102)
    models = cloned_x20r_models(d_model=32)
    modes = (SOFT_MODE, ST_GRAPH_MODE, ST_BLIND_MODE)
    counts = {models[m].parameter_count() for m in modes}
    trainable = {models[m].trainable_parameter_count() for m in modes}
    assert len(counts) == 1
    assert len(trainable) == 1
    reference = list(models[SOFT_MODE].named_parameters())
    for mode in (ST_GRAPH_MODE, ST_BLIND_MODE):
        candidate = list(models[mode].named_parameters())
        assert [n for n, _ in reference] == [n for n, _ in candidate]
        for (_, a), (_, b) in zip(reference, candidate):
            assert torch.equal(a, b)


def test_matched_soft_and_st_graph_raw_gates_are_identical_before_training():
    torch.manual_seed(103)
    models = cloned_x20r_models(d_model=32)
    batch = make_state_instantiation_batch(5, 12, 1803, live_cardinality=4, split="train")
    a = models[SOFT_MODE].soft_gates(batch)
    b = models[ST_GRAPH_MODE].soft_gates(batch)
    assert torch.equal(a, b)


def test_straight_through_forward_is_exactly_binary_and_matches_half_threshold():
    g = torch.tensor([[0.49, 0.50, 0.51, 0.1, 0.9, 0.2, 0.8, 0.3]], requires_grad=True)
    st = straight_through_binary(g)
    assert st.tolist() == [[0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]]
    assert torch.equal(st.detach().bool(), g.detach() >= 0.5)


def test_straight_through_storage_forward_counts_hard_records_but_gradients_reach_soft_gate():
    g = torch.tensor([[0.2, 0.8, 0.49, 0.51]], requires_grad=True)
    st = straight_through_binary(g)
    storage = st.mean()
    assert storage.item() == 0.5
    storage.backward()
    assert torch.equal(g.grad, torch.full_like(g, 0.25))


def test_st_graph_training_gates_are_binary_in_forward_value():
    torch.manual_seed(104)
    batch = make_state_instantiation_batch(6, 12, 1804, live_cardinality=3, split="train")
    model = X20RStateInstantiationModel(mode=ST_GRAPH_MODE, d_model=32)
    raw = model.soft_gates(batch)
    train = model.training_gates(batch)
    assert ((raw > 0.0) & (raw < 1.0)).all()
    assert torch.equal(train.detach().bool(), raw.detach() >= 0.5)
    assert torch.equal(train.detach(), (raw.detach() >= 0.5).to(train.dtype))


def test_st_answer_and_storage_loss_reach_graph_constructor():
    torch.manual_seed(105)
    batch = make_state_instantiation_batch(4, 12, 1805, live_cardinality=3, split="train")
    model = X20RStateInstantiationModel(mode=ST_GRAPH_MODE, d_model=32)
    parts = model.loss_components(batch)
    parts["total_loss"].backward()
    assert model.constructor is not None
    constructor_grad = sum(
        float(p.grad.abs().sum()) for p in model.constructor.parameters() if p.grad is not None
    )
    executor_grad = sum(
        float(p.grad.abs().sum()) for p in model.executor.parameters() if p.grad is not None
    )
    assert constructor_grad > 0.0
    assert executor_grad > 0.0


def test_hidden_live_mask_cannot_change_x20r_learned_gates():
    torch.manual_seed(106)
    batch = make_state_instantiation_batch(4, 12, 1806, live_cardinality=3, split="train")
    for mode in (SOFT_MODE, ST_GRAPH_MODE, ST_BLIND_MODE):
        model = X20RStateInstantiationModel(mode=mode, d_model=32)
        g1 = model.soft_gates(batch)
        g2 = model.soft_gates(replace(batch, live_mask=~batch.live_mask))
        assert torch.equal(g1, g2)


def test_no_learned_per_candidate_gate_table_in_x20r():
    model = X20RStateInstantiationModel(mode=ST_GRAPH_MODE, d_model=32)
    assert model.constructor is not None
    for name, p in model.constructor.named_parameters():
        if name.startswith("command."):
            continue
        assert not (p.ndim >= 2 and p.shape[0] == NUM_CANDIDATES), (name, tuple(p.shape))
