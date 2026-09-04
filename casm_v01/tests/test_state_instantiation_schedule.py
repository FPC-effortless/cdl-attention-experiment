from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from casm.state_instantiation_data import NUM_CANDIDATES, make_state_instantiation_batch
from casm.state_instantiation_schedule import (
    DELAYED_ABRUPT_MODE,
    DELAYED_RAMP_BLIND_MODE,
    DELAYED_RAMP_MODE,
    IMMEDIATE_MODE,
    NO_STORAGE_MODE,
    X20R_FROZEN_RESULT,
    X20S_GRAPH_MODES,
    X20S_LEARNED_MODES,
    X20S_PREREGISTRATION,
    cloned_x20s_models,
    scheduled_loss_components,
    storage_lambda,
)
from casm.state_instantiation_st import ST_GRAPH_MODE, X20RStateInstantiationModel


def test_x20s_frozen_provenance_constants():
    assert X20R_FROZEN_RESULT == "a4b8ee98dd300dc51e4398c84020a2e90c2cccc6"
    assert X20S_PREREGISTRATION == "cc9168011e8a2f578bc7dce9879dc42a6161e201"


def test_x20s_schedule_values_are_exactly_preregistered():
    assert [storage_lambda(IMMEDIATE_MODE, s) for s in (1, 500, 1000, 1001, 1500, 2000, 12000)] == [0.05] * 7
    assert [storage_lambda(NO_STORAGE_MODE, s) for s in (1, 500, 1000, 1001, 1500, 2000, 12000)] == [0.0] * 7
    assert [storage_lambda(DELAYED_ABRUPT_MODE, s) for s in (1, 500, 1000)] == [0.0, 0.0, 0.0]
    assert [storage_lambda(DELAYED_ABRUPT_MODE, s) for s in (1001, 1500, 2000, 12000)] == [0.05] * 4
    assert storage_lambda(DELAYED_RAMP_MODE, 1) == 0.0
    assert storage_lambda(DELAYED_RAMP_MODE, 1000) == 0.0
    assert storage_lambda(DELAYED_RAMP_MODE, 1001) == pytest.approx(0.00005, abs=1e-12)
    assert storage_lambda(DELAYED_RAMP_MODE, 1500) == pytest.approx(0.025, abs=1e-12)
    assert storage_lambda(DELAYED_RAMP_MODE, 2000) == pytest.approx(0.05, abs=1e-12)
    assert storage_lambda(DELAYED_RAMP_MODE, 12000) == pytest.approx(0.05, abs=1e-12)
    for step in (1, 500, 1000, 1001, 1500, 2000, 12000):
        assert storage_lambda(DELAYED_RAMP_BLIND_MODE, step) == storage_lambda(DELAYED_RAMP_MODE, step)


def test_graph_scheduled_regimes_start_bit_identically_and_match_parameter_count():
    torch.manual_seed(201)
    models = cloned_x20s_models(d_model=32)
    counts = {models[m].parameter_count() for m in X20S_LEARNED_MODES}
    trainable = {models[m].trainable_parameter_count() for m in X20S_LEARNED_MODES}
    assert len(counts) == 1
    assert len(trainable) == 1
    reference = list(models[IMMEDIATE_MODE].named_parameters())
    for mode in X20S_LEARNED_MODES:
        candidate = list(models[mode].named_parameters())
        assert [n for n, _ in reference] == [n for n, _ in candidate]
        for (_, a), (_, b) in zip(reference, candidate):
            assert torch.equal(a, b)


def test_graph_scheduled_raw_gates_are_identical_before_training():
    torch.manual_seed(202)
    models = cloned_x20s_models(d_model=32)
    batch = make_state_instantiation_batch(5, 12, 2202, live_cardinality=4, split="train")
    reference = models[IMMEDIATE_MODE].soft_gates(batch)
    for mode in X20S_GRAPH_MODES:
        assert torch.equal(reference, models[mode].soft_gates(batch))


def test_immediate_schedule_is_exact_x20r_loss_when_weights_match():
    torch.manual_seed(203)
    old = X20RStateInstantiationModel(mode=ST_GRAPH_MODE, d_model=32)
    models = cloned_x20s_models(d_model=32)
    new = models[IMMEDIATE_MODE]
    new.load_state_dict(old.state_dict())
    batch = make_state_instantiation_batch(6, 12, 2203, live_cardinality=3, split="train")
    old_parts = old.loss_components(batch)
    new_parts = scheduled_loss_components(new, batch, mode=IMMEDIATE_MODE, step=1)
    assert torch.equal(old_parts["answer_loss"], new_parts["answer_loss"])
    assert torch.equal(old_parts["storage_penalty"], new_parts["hard_record_fraction"])
    assert torch.equal(old_parts["total_loss"], new_parts["total_loss"])


def test_schedule_changes_only_storage_coefficient_for_matched_graph_models():
    torch.manual_seed(204)
    models = cloned_x20s_models(d_model=32)
    batch = make_state_instantiation_batch(5, 12, 2204, live_cardinality=3, split="train")
    immediate = scheduled_loss_components(models[IMMEDIATE_MODE], batch, mode=IMMEDIATE_MODE, step=1)
    no_storage = scheduled_loss_components(models[NO_STORAGE_MODE], batch, mode=NO_STORAGE_MODE, step=1)
    delayed = scheduled_loss_components(models[DELAYED_RAMP_MODE], batch, mode=DELAYED_RAMP_MODE, step=1)
    assert torch.equal(immediate["answer_loss"], no_storage["answer_loss"])
    assert torch.equal(immediate["answer_loss"], delayed["answer_loss"])
    assert torch.equal(immediate["hard_record_fraction"], no_storage["hard_record_fraction"])
    assert torch.equal(immediate["hard_record_fraction"], delayed["hard_record_fraction"])
    assert float(immediate["storage_lambda"]) == pytest.approx(0.05)
    assert float(no_storage["storage_lambda"]) == 0.0
    assert float(delayed["storage_lambda"]) == 0.0


def test_delayed_ramp_storage_gradient_reaches_constructor_without_new_labels():
    torch.manual_seed(205)
    models = cloned_x20s_models(d_model=32)
    model = models[DELAYED_RAMP_MODE]
    batch = make_state_instantiation_batch(4, 12, 2205, live_cardinality=3, split="train")
    parts = scheduled_loss_components(model, batch, mode=DELAYED_RAMP_MODE, step=1500)
    assert float(parts["storage_lambda"]) == pytest.approx(0.025)
    parts["storage_cost"].backward()
    assert model.constructor is not None
    grad = sum(float(p.grad.abs().sum()) for p in model.constructor.parameters() if p.grad is not None)
    assert grad > 0.0


def test_hidden_live_mask_cannot_change_x20s_learned_gates():
    torch.manual_seed(206)
    models = cloned_x20s_models(d_model=32)
    batch = make_state_instantiation_batch(4, 12, 2206, live_cardinality=3, split="train")
    corrupted = replace(batch, live_mask=~batch.live_mask)
    for mode in X20S_LEARNED_MODES:
        assert torch.equal(models[mode].soft_gates(batch), models[mode].soft_gates(corrupted))


def test_x20s_training_forward_is_binary_for_every_learned_regime():
    torch.manual_seed(207)
    models = cloned_x20s_models(d_model=32)
    batch = make_state_instantiation_batch(6, 12, 2207, live_cardinality=3, split="train")
    for mode in X20S_LEARNED_MODES:
        raw = models[mode].soft_gates(batch)
        train = models[mode].training_gates(batch)
        expected = (raw.detach() >= 0.5).to(train.dtype)
        assert torch.equal(train.detach(), expected)


def test_no_new_per_candidate_gate_table_in_x20s():
    models = cloned_x20s_models(d_model=32)
    for mode in X20S_LEARNED_MODES:
        model = models[mode]
        assert model.constructor is not None
        for name, p in model.constructor.named_parameters():
            if name.startswith("command."):
                continue
            assert not (p.ndim >= 2 and p.shape[0] == NUM_CANDIDATES), (mode, name, tuple(p.shape))
