from __future__ import annotations

from dataclasses import replace

import torch

from casm.state_instantiation_credit import DUAL_CREDIT_MODE, credit_loss_components
from casm.state_instantiation_data import NUM_CANDIDATES, make_state_instantiation_batch
from casm.state_instantiation_local_credit import (
    CANONICAL_MODE,
    DUAL_REPLICATION_MODE,
    GLOBAL_TASK_WEIGHT,
    LOCAL_CREDIT_BLIND_MODE,
    LOCAL_CREDIT_MODE,
    LOCAL_TASK_WEIGHT,
    X20T_FROZEN_RESULT,
    X20U_LEARNED_MODES,
    X20U_PREREGISTRATION,
    cloned_x20u_models,
    counterfactual_answer_losses,
    forced_on_off_gates,
    local_counterfactual_risk,
    local_credit_loss_components,
)
from casm.state_instantiation_st import ST_GRAPH_MODE, X20RStateInstantiationModel, straight_through_binary
from casm.state_instantiation_credit import STORAGE_LAMBDA


def test_x20u_provenance_constants_are_frozen():
    assert X20T_FROZEN_RESULT == "a122c447efc31054b61134b3271cbc282f167ddb"
    assert X20U_PREREGISTRATION == "524b372f89e2b3fd554131782c87278c58bf0552"


def test_all_learned_x20u_regimes_start_bit_identically_and_match_parameter_count():
    torch.manual_seed(301)
    models = cloned_x20u_models(d_model=32)
    counts = {models[m].parameter_count() for m in X20U_LEARNED_MODES}
    trainable = {models[m].trainable_parameter_count() for m in X20U_LEARNED_MODES}
    assert len(counts) == 1
    assert len(trainable) == 1
    reference = list(models[DUAL_REPLICATION_MODE].named_parameters())
    for mode in X20U_LEARNED_MODES:
        candidate = list(models[mode].named_parameters())
        assert [n for n, _ in reference] == [n for n, _ in candidate]
        for (_, a), (_, b) in zip(reference, candidate):
            assert torch.equal(a, b), mode


def test_graph_regimes_have_identical_raw_gates_before_training():
    torch.manual_seed(302)
    models = cloned_x20u_models(d_model=32)
    batch = make_state_instantiation_batch(5, 12, 3302, live_cardinality=4, split="train")
    a = models[DUAL_REPLICATION_MODE].soft_gates(batch)
    b = models[LOCAL_CREDIT_MODE].soft_gates(batch)
    assert torch.equal(a, b)


def test_dual_replication_is_exact_x20t_dual_objective():
    torch.manual_seed(303)
    old = X20RStateInstantiationModel(mode=ST_GRAPH_MODE, d_model=32)
    new = X20RStateInstantiationModel(mode=ST_GRAPH_MODE, d_model=32)
    new.load_state_dict(old.state_dict())
    batch = make_state_instantiation_batch(6, 12, 3303, live_cardinality=3, split="train")
    old_parts = credit_loss_components(old, batch, mode=DUAL_CREDIT_MODE)
    new_parts = local_credit_loss_components(new, batch, mode=DUAL_REPLICATION_MODE)
    assert torch.equal(old_parts["hard_answer_loss"], new_parts["hard_answer_loss"])
    assert torch.equal(old_parts["soft_answer_loss"], new_parts["soft_answer_loss"])
    assert torch.equal(old_parts["task_loss"], new_parts["task_loss"])
    assert torch.equal(old_parts["storage_penalty"], new_parts["storage_penalty"])
    assert torch.equal(old_parts["total_loss"], new_parts["total_loss"])


def test_forced_on_off_gates_change_only_selected_candidate():
    g = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])
    on, off = forced_on_off_gates(g)
    assert on.shape == (1, NUM_CANDIDATES, NUM_CANDIDATES)
    assert off.shape == on.shape
    for i in range(NUM_CANDIDATES):
        expected_on = g.clone()
        expected_off = g.clone()
        expected_on[:, i] = 1.0
        expected_off[:, i] = 0.0
        assert torch.equal(on[:, i], expected_on)
        assert torch.equal(off[:, i], expected_off)


def test_local_counterfactual_risk_has_exact_formula_and_detached_outcomes():
    g = torch.tensor([[0.25, 0.75]], requires_grad=True)
    a_on = torch.tensor([[1.0, 3.0]], requires_grad=True)
    a_off = torch.tensor([[3.0, 1.0]], requires_grad=True)
    risk = local_counterfactual_risk(g, a_on, a_off)
    expected = (0.25 * 1.0 + 0.75 * 3.0 + 0.75 * 3.0 + 0.25 * 1.0) / 2.0
    assert float(risk.detach()) == expected
    risk.backward()
    assert g.grad is not None
    assert torch.allclose(g.grad, torch.tensor([[-1.0, 1.0]]))
    assert a_on.grad is None
    assert a_off.grad is None


def test_local_gradient_direction_matches_counterfactual_helpfulness():
    g = torch.tensor([[0.5, 0.5]], requires_grad=True)
    # Candidate 0 helps when on (lower loss); candidate 1 hurts when on (higher loss).
    a_on = torch.tensor([[0.5, 2.0]])
    a_off = torch.tensor([[2.0, 0.5]])
    local_counterfactual_risk(g, a_on, a_off).backward()
    assert float(g.grad[0, 0]) < 0.0  # gradient descent increases gate 0
    assert float(g.grad[0, 1]) > 0.0  # gradient descent decreases gate 1


def test_counterfactual_losses_are_finite_and_do_not_attach_to_gate_graph():
    torch.manual_seed(304)
    model = X20RStateInstantiationModel(mode=ST_GRAPH_MODE, d_model=32)
    batch = make_state_instantiation_batch(4, 12, 3304, live_cardinality=3, split="train")
    g = model.soft_gates(batch)
    a_on, a_off = counterfactual_answer_losses(model, batch, g)
    assert a_on.shape == g.shape == a_off.shape
    assert torch.isfinite(a_on).all() and torch.isfinite(a_off).all()
    assert not a_on.requires_grad
    assert not a_off.requires_grad


def test_local_objective_matches_preregistered_mixture_and_hard_storage():
    torch.manual_seed(305)
    model = X20RStateInstantiationModel(mode=ST_GRAPH_MODE, d_model=32)
    batch = make_state_instantiation_batch(4, 12, 3305, live_cardinality=3, split="train")
    parts = local_credit_loss_components(model, batch, mode=LOCAL_CREDIT_MODE)
    global_expected = 0.5 * parts["hard_answer_loss"] + 0.5 * parts["soft_answer_loss"]
    task_expected = GLOBAL_TASK_WEIGHT * global_expected + LOCAL_TASK_WEIGHT * parts["local_counterfactual_risk"]
    total_expected = task_expected + STORAGE_LAMBDA * parts["storage_penalty"]
    assert torch.equal(parts["global_task_loss"], global_expected)
    assert torch.equal(parts["task_loss"], task_expected)
    assert torch.equal(parts["total_loss"], total_expected)
    raw = model.soft_gates(batch)
    hard = straight_through_binary(raw)
    assert torch.equal(parts["storage_penalty"], hard.mean())


def test_local_risk_reaches_constructor_and_global_path_reaches_executor():
    torch.manual_seed(306)
    model = X20RStateInstantiationModel(mode=ST_GRAPH_MODE, d_model=32)
    batch = make_state_instantiation_batch(8, 12, 3306, live_cardinality=3, split="train")
    g = model.soft_gates(batch)
    a_on, a_off = counterfactual_answer_losses(model, batch, g)
    local = local_counterfactual_risk(g, a_on, a_off)
    local.backward()
    assert model.constructor is not None
    constructor_grad = sum(float(p.grad.abs().sum()) for p in model.constructor.parameters() if p.grad is not None)
    executor_grad = sum(float(p.grad.abs().sum()) for p in model.executor.parameters() if p.grad is not None)
    assert constructor_grad > 0.0
    assert executor_grad == 0.0

    model.zero_grad(set_to_none=True)
    parts = local_credit_loss_components(model, batch, mode=LOCAL_CREDIT_MODE)
    parts["global_task_loss"].backward()
    executor_grad = sum(float(p.grad.abs().sum()) for p in model.executor.parameters() if p.grad is not None)
    assert executor_grad > 0.0


def test_hidden_live_mask_cannot_change_gates_or_local_credit_loss():
    torch.manual_seed(307)
    models = cloned_x20u_models(d_model=32)
    batch = make_state_instantiation_batch(4, 12, 3307, live_cardinality=3, split="train")
    altered = replace(batch, live_mask=~batch.live_mask)
    for mode in X20U_LEARNED_MODES:
        assert torch.equal(models[mode].soft_gates(batch), models[mode].soft_gates(altered)), mode
        a = local_credit_loss_components(models[mode], batch, mode=mode)["total_loss"]
        b = local_credit_loss_components(models[mode], altered, mode=mode)["total_loss"]
        assert torch.equal(a, b), mode


def test_no_learned_per_candidate_gate_table_in_x20u():
    models = cloned_x20u_models(d_model=32)
    for mode in X20U_LEARNED_MODES:
        model = models[mode]
        assert model.constructor is not None
        for name, p in model.constructor.named_parameters():
            if name.startswith("command."):
                continue
            assert not (p.ndim >= 2 and p.shape[0] == NUM_CANDIDATES), (mode, name, tuple(p.shape))


def test_structure_blind_uses_same_local_objective_formula():
    torch.manual_seed(308)
    models = cloned_x20u_models(d_model=32)
    batch = make_state_instantiation_batch(4, 12, 3308, live_cardinality=3, split="train")
    parts = local_credit_loss_components(models[LOCAL_CREDIT_BLIND_MODE], batch, mode=LOCAL_CREDIT_BLIND_MODE)
    expected = (
        GLOBAL_TASK_WEIGHT * (0.5 * parts["hard_answer_loss"] + 0.5 * parts["soft_answer_loss"])
        + LOCAL_TASK_WEIGHT * parts["local_counterfactual_risk"]
        + STORAGE_LAMBDA * parts["storage_penalty"]
    )
    assert torch.equal(parts["total_loss"], expected)


def test_canonical_has_no_storage_or_local_cost():
    torch.manual_seed(309)
    model = X20RStateInstantiationModel(mode=CANONICAL_MODE, d_model=32)
    batch = make_state_instantiation_batch(4, 12, 3309, live_cardinality=3, split="train")
    parts = local_credit_loss_components(model, batch, mode=CANONICAL_MODE)
    assert float(parts["storage_cost"].detach()) == 0.0
    assert float(parts["local_counterfactual_risk"].detach()) == 0.0
    assert torch.equal(parts["total_loss"], parts["hard_answer_loss"])
