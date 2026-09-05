from __future__ import annotations

from dataclasses import replace

import torch

from casm.state_instantiation_credit import (
    CANONICAL_MODE,
    DUAL_CREDIT_MODE,
    HARD_ONLY_MODE,
    SOFT_CREDIT_BLIND_MODE,
    SOFT_CREDIT_MODE,
    SOFT_X20_MODE,
    STORAGE_LAMBDA,
    X20S_FROZEN_RESULT,
    X20T_PREREGISTRATION,
    X20T_LEARNED_MODES,
    cloned_x20t_models,
    credit_loss_components,
)
from casm.state_instantiation_data import NUM_CANDIDATES, make_state_instantiation_batch
from casm.state_instantiation_st import (
    SOFT_MODE,
    ST_GRAPH_MODE,
    X20RStateInstantiationModel,
    straight_through_binary,
)


def test_x20t_provenance_constants_are_frozen():
    assert X20S_FROZEN_RESULT == "9d4ebc5805f11e7e6e208de006878508111e201e"
    assert X20T_PREREGISTRATION == "9365e9e1ce3242df6305abe9fd816e66298caa64"


def test_all_learned_x20t_regimes_start_bit_identically_and_match_parameter_count():
    torch.manual_seed(201)
    models = cloned_x20t_models(d_model=32)
    counts = {models[m].parameter_count() for m in X20T_LEARNED_MODES}
    trainable = {models[m].trainable_parameter_count() for m in X20T_LEARNED_MODES}
    assert len(counts) == 1
    assert len(trainable) == 1
    reference = list(models[HARD_ONLY_MODE].named_parameters())
    for mode in X20T_LEARNED_MODES:
        candidate = list(models[mode].named_parameters())
        assert [n for n, _ in reference] == [n for n, _ in candidate]
        for (_, a), (_, b) in zip(reference, candidate):
            assert torch.equal(a, b), mode


def test_graph_regimes_have_identical_raw_gates_before_training():
    torch.manual_seed(202)
    models = cloned_x20t_models(d_model=32)
    batch = make_state_instantiation_batch(5, 12, 2202, live_cardinality=4, split="train")
    ref = models[HARD_ONLY_MODE].soft_gates(batch)
    for mode in (SOFT_X20_MODE, SOFT_CREDIT_MODE, DUAL_CREDIT_MODE):
        assert torch.equal(ref, models[mode].soft_gates(batch)), mode


def test_hard_only_is_exact_x20r_loss_replication():
    torch.manual_seed(203)
    old = X20RStateInstantiationModel(mode=ST_GRAPH_MODE, d_model=32)
    new = X20RStateInstantiationModel(mode=ST_GRAPH_MODE, d_model=32)
    new.load_state_dict(old.state_dict())
    batch = make_state_instantiation_batch(6, 12, 2203, live_cardinality=3, split="train")
    old_parts = old.loss_components(batch)
    new_parts = credit_loss_components(new, batch, mode=HARD_ONLY_MODE)
    assert torch.equal(old_parts["answer_loss"], new_parts["hard_answer_loss"])
    assert torch.equal(old_parts["storage_penalty"], new_parts["storage_penalty"])
    assert torch.equal(old_parts["total_loss"], new_parts["total_loss"])


def test_soft_x20_is_exact_continuous_loss_replication():
    torch.manual_seed(204)
    old = X20RStateInstantiationModel(mode=SOFT_MODE, d_model=32)
    new = X20RStateInstantiationModel(mode=SOFT_MODE, d_model=32)
    new.load_state_dict(old.state_dict())
    batch = make_state_instantiation_batch(6, 12, 2204, live_cardinality=2, split="train")
    old_parts = old.loss_components(batch)
    new_parts = credit_loss_components(new, batch, mode=SOFT_X20_MODE)
    assert torch.equal(old_parts["answer_loss"], new_parts["soft_answer_loss"])
    assert torch.equal(old_parts["storage_penalty"], new_parts["storage_penalty"])
    assert torch.equal(old_parts["total_loss"], new_parts["total_loss"])


def test_soft_credit_objective_is_exact_soft_answer_plus_hard_storage():
    torch.manual_seed(205)
    model = X20RStateInstantiationModel(mode=ST_GRAPH_MODE, d_model=32)
    batch = make_state_instantiation_batch(4, 12, 2205, live_cardinality=3, split="train")
    parts = credit_loss_components(model, batch, mode=SOFT_CREDIT_MODE)
    expected = parts["soft_answer_loss"] + STORAGE_LAMBDA * parts["storage_penalty"]
    assert torch.equal(parts["task_loss"], parts["soft_answer_loss"])
    assert torch.equal(parts["total_loss"], expected)
    raw = model.soft_gates(batch)
    hard = straight_through_binary(raw)
    assert torch.equal(parts["storage_penalty"], hard.mean())


def test_dual_credit_objective_uses_equal_hard_soft_task_weight():
    torch.manual_seed(206)
    model = X20RStateInstantiationModel(mode=ST_GRAPH_MODE, d_model=32)
    batch = make_state_instantiation_batch(4, 12, 2206, live_cardinality=4, split="train")
    parts = credit_loss_components(model, batch, mode=DUAL_CREDIT_MODE)
    task = 0.5 * parts["hard_answer_loss"] + 0.5 * parts["soft_answer_loss"]
    assert torch.equal(parts["task_loss"], task)
    assert torch.equal(parts["total_loss"], task + STORAGE_LAMBDA * parts["storage_penalty"])


def test_hard_storage_forward_counts_thresholded_records_and_keeps_gradient():
    g = torch.tensor([[0.2, 0.8, 0.49, 0.51]], requires_grad=True)
    st = straight_through_binary(g)
    storage = st.mean()
    assert storage.item() == 0.5
    storage.backward()
    assert torch.equal(g.grad, torch.full_like(g, 0.25))


def test_soft_counterfactual_answer_path_reaches_constructor_when_hard_task_is_poor():
    torch.manual_seed(207)
    model = X20RStateInstantiationModel(mode=ST_GRAPH_MODE, d_model=32)
    batch = make_state_instantiation_batch(16, 12, 2207, live_cardinality=3, split="train")
    parts = credit_loss_components(model, batch, mode=SOFT_CREDIT_MODE)
    assert float(parts["hard_answer_loss"].detach()) > 1.0
    parts["soft_answer_loss"].backward()
    assert model.constructor is not None
    constructor_grad = sum(
        float(p.grad.abs().sum()) for p in model.constructor.parameters() if p.grad is not None
    )
    assert constructor_grad > 0.0


def test_hidden_live_mask_cannot_change_learned_gates_or_credit_loss():
    torch.manual_seed(208)
    models = cloned_x20t_models(d_model=32)
    batch = make_state_instantiation_batch(4, 12, 2208, live_cardinality=3, split="train")
    altered = replace(batch, live_mask=~batch.live_mask)
    for mode in X20T_LEARNED_MODES:
        assert torch.equal(models[mode].soft_gates(batch), models[mode].soft_gates(altered)), mode
        a = credit_loss_components(models[mode], batch, mode=mode)["total_loss"]
        b = credit_loss_components(models[mode], altered, mode=mode)["total_loss"]
        assert torch.equal(a, b), mode


def test_no_learned_per_candidate_gate_table_in_x20t():
    models = cloned_x20t_models(d_model=32)
    for mode in X20T_LEARNED_MODES:
        model = models[mode]
        assert model.constructor is not None
        for name, p in model.constructor.named_parameters():
            if name.startswith("command."):
                continue
            assert not (p.ndim >= 2 and p.shape[0] == NUM_CANDIDATES), (mode, name, tuple(p.shape))


def test_canonical_has_no_storage_cost():
    torch.manual_seed(209)
    model = X20RStateInstantiationModel(mode=CANONICAL_MODE, d_model=32)
    batch = make_state_instantiation_batch(4, 12, 2209, live_cardinality=3, split="train")
    parts = credit_loss_components(model, batch, mode=CANONICAL_MODE)
    assert float(parts["storage_cost"].detach()) == 0.0
    assert torch.equal(parts["total_loss"], parts["hard_answer_loss"])


def test_structure_blind_credit_uses_same_objective_formula():
    torch.manual_seed(210)
    models = cloned_x20t_models(d_model=32)
    batch = make_state_instantiation_batch(4, 12, 2210, live_cardinality=3, split="train")
    parts = credit_loss_components(models[SOFT_CREDIT_BLIND_MODE], batch, mode=SOFT_CREDIT_BLIND_MODE)
    expected = parts["soft_answer_loss"] + STORAGE_LAMBDA * parts["storage_penalty"]
    assert torch.equal(parts["total_loss"], expected)
