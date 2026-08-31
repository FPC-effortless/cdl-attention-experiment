from __future__ import annotations

from copy import deepcopy

import torch

from casm.contextual_data import make_contextual_batch
from casm.fixed_answer_register import (
    FIXED_REGISTER,
    cloned_fixed_answer_models,
    fixed_query_registers,
    regime_loss,
    sample_random_query_registers,
)
from casm.run_fixed_answer_register import answer_hidden_metrics
from casm.weak_supervision import SoftExplicitTransitionModel


def test_fixed_answer_regimes_start_from_identical_parameters():
    torch.manual_seed(23)
    seed = SoftExplicitTransitionModel(d_model=32)
    models = cloned_fixed_answer_models(seed)
    reference = models["full_final"].state_dict()
    for regime in ("random_register", "fixed_register"):
        for key, value in models[regime].state_dict().items():
            assert torch.equal(reference[key], value)


def test_fixed_query_is_always_register_zero():
    query = fixed_query_registers(1024)
    assert query.dtype == torch.long
    assert torch.equal(query, torch.zeros_like(query))
    assert FIXED_REGISTER == 0


def test_fixed_register_loss_ignores_all_intermediate_and_hidden_final_targets():
    batch = make_contextual_batch(32, 8, 9301, split="train")
    random_query = sample_random_query_registers(32, 9401)
    corrupted = deepcopy(batch)
    corrupted.target_states = batch.target_states.clone()
    corrupted.target_states[:, :-1] = torch.remainder(corrupted.target_states[:, :-1] + 1, 16)
    corrupted.target_states[:, -1, 1:] = torch.remainder(
        corrupted.target_states[:, -1, 1:] + 1, 16
    )
    model = SoftExplicitTransitionModel(d_model=32)
    a = regime_loss(model, batch, "fixed_register", random_query)
    b = regime_loss(model, corrupted, "fixed_register", random_query)
    assert torch.equal(a, b)


def test_fixed_register_loss_changes_when_answer_target_changes():
    torch.manual_seed(29)
    batch = make_contextual_batch(32, 8, 9302, split="train")
    random_query = sample_random_query_registers(32, 9402)
    changed = deepcopy(batch)
    changed.target_states = batch.target_states.clone()
    changed.target_states[:, -1, FIXED_REGISTER] = torch.remainder(
        changed.target_states[:, -1, FIXED_REGISTER] + 1, 16
    )
    model = SoftExplicitTransitionModel(d_model=32)
    a = regime_loss(model, batch, "fixed_register", random_query)
    b = regime_loss(model, changed, "fixed_register", random_query)
    assert not torch.equal(a, b)


def test_fixed_register_loss_ignores_semantic_operator_labels():
    batch = make_contextual_batch(32, 8, 9303, split="train")
    random_query = sample_random_query_registers(32, 9403)
    corrupted = deepcopy(batch)
    corrupted.semantics = torch.randint_like(batch.semantics, 0, 8)
    model = SoftExplicitTransitionModel(d_model=32)
    a = regime_loss(model, batch, "fixed_register", random_query)
    b = regime_loss(model, corrupted, "fixed_register", random_query)
    assert torch.equal(a, b)


def test_fixed_register_loss_has_finite_transition_gradients():
    batch = make_contextual_batch(32, 8, 9304, split="train")
    random_query = sample_random_query_registers(32, 9404)
    model = SoftExplicitTransitionModel(d_model=32)
    loss = regime_loss(model, batch, "fixed_register", random_query)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.transition.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_answer_hidden_metrics_separate_observed_answer_from_hidden_state():
    batch = make_contextual_batch(8, 8, 9305, split="train")
    pred = batch.target_states.clone()
    pred[:, :, 1] = torch.remainder(pred[:, :, 1] + 1, 16)
    metrics = answer_hidden_metrics(pred, batch.target_states)
    assert metrics["answer_final_accuracy"] == 1.0
    assert metrics["answer_step_accuracy"] == 1.0
    assert metrics["hidden_register_accuracy"] < 1.0
    assert metrics["hidden_final_exact"] == 0.0
