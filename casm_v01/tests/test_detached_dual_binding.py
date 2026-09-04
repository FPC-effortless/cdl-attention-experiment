from __future__ import annotations

from dataclasses import replace

import torch

from casm.detached_dual_binding import (
    LEARNED_X18R_MODES,
    X18RBindingModel,
    cloned_x18r_models,
    primal_dual_binding_backward_mode,
    x16_forward_reference,
)
from casm.explicit_compute import VALUE_MODULUS
from casm.variable_cardinality_binding import NUM_CANDIDATE_SLOTS, variable_descriptor
from casm.variable_contextual_data import make_variable_contextual_batch


def test_fullgrad_forward_is_exact_x16_reference():
    torch.manual_seed(18101)
    for n in (2, 3, 4):
        logits = torch.randn(n, NUM_CANDIDATE_SLOTS)
        actual_b, actual_p = primal_dual_binding_backward_mode(logits, detach_prices=False)
        expected_b, expected_p = x16_forward_reference(logits)
        assert torch.equal(actual_b, expected_b)
        assert torch.equal(actual_p, expected_p)


def test_detached_and_fullgrad_are_bit_identical_forward_random_and_adversarial():
    torch.manual_seed(18102)
    cases = []
    for n in (2, 3, 4):
        cases.append(torch.randn(n, NUM_CANDIDATE_SLOTS))
        cases.append(torch.zeros(n, NUM_CANDIDATE_SLOTS))
        overloaded = torch.full((n, NUM_CANDIDATE_SLOTS), -8.0)
        overloaded[:, 0] = 12.0
        cases.append(overloaded)
        tied = torch.zeros(n, NUM_CANDIDATE_SLOTS)
        tied[:, :2] = 4.0
        cases.append(tied)
    for logits in cases:
        full_b, full_p = primal_dual_binding_backward_mode(logits, detach_prices=False)
        det_b, det_p = primal_dual_binding_backward_mode(logits, detach_prices=True)
        assert torch.equal(full_b, det_b)
        assert torch.equal(full_p, det_p)
        assert torch.isfinite(det_b).all() and torch.isfinite(det_p).all()
        assert (det_p >= 0).all()
        assert torch.allclose(
            det_b.sum(dim=1),
            torch.ones(det_b.shape[0]),
            atol=1e-7,
            rtol=0.0,
        )


def test_detached_prices_remove_history_but_keep_nonzero_logit_gradient():
    torch.manual_seed(18103)
    logits_full = torch.randn(4, NUM_CANDIDATE_SLOTS, requires_grad=True)
    logits_det = logits_full.detach().clone().requires_grad_(True)
    weights = torch.arange(1, 1 + logits_full.numel(), dtype=logits_full.dtype).reshape_as(logits_full)

    full_b, full_p = primal_dual_binding_backward_mode(logits_full, detach_prices=False)
    det_b, det_p = primal_dual_binding_backward_mode(logits_det, detach_prices=True)
    assert torch.equal(full_b, det_b)
    assert torch.equal(full_p, det_p)
    assert full_p.requires_grad
    assert not det_p.requires_grad

    grad_full = torch.autograd.grad((full_b * weights).sum(), logits_full)[0]
    grad_det = torch.autograd.grad((det_b * weights).sum(), logits_det)[0]
    assert torch.isfinite(grad_full).all() and torch.isfinite(grad_det).all()
    assert float(grad_det.abs().sum()) > 0.0
    assert not torch.allclose(grad_full, grad_det, atol=1e-8, rtol=1e-6)


def test_exact_symmetric_logits_remain_symmetric_and_unrepaired():
    logits = torch.zeros(4, NUM_CANDIDATE_SLOTS)
    binding, prices = primal_dual_binding_backward_mode(logits, detach_prices=True)
    assert torch.equal(binding, torch.full_like(binding, 1.0 / NUM_CANDIDATE_SLOTS))
    assert torch.equal(prices, torch.zeros_like(prices))
    assignment = binding.argmax(dim=1).tolist()
    assert assignment == [0, 0, 0, 0]


def test_learned_models_start_bit_identical_parameter_matched_and_relative_descriptor_only():
    torch.manual_seed(18104)
    models = cloned_x18r_models(d_model=32)
    full = models["dual_fullgrad"]
    detached = models["dual_detached_prices"]
    assert full.parameter_count() == detached.parameter_count()
    assert full.trainable_parameter_count() == detached.trainable_parameter_count()
    fs = full.state_dict()
    ds = detached.state_dict()
    assert fs.keys() == ds.keys()
    assert all(torch.equal(fs[k], ds[k]) for k in fs)

    for n in (2, 3, 4):
        parameter = next(full.binding_generator.parameters())
        indices = torch.arange(n)
        expected = variable_descriptor(indices, n, dtype=parameter.dtype)
        assert expected.shape == (n, 9)
        assert torch.equal(full.base_logits(n), detached.base_logits(n))
        full_b, full_p = full.soft_binding_and_prices(n)
        det_b, det_p = detached.soft_binding_and_prices(n)
        assert torch.equal(full_b, det_b)
        assert torch.equal(full_p, det_p)


def test_detached_model_price_state_has_no_grad_history():
    model = X18RBindingModel(mode="dual_detached_prices", d_model=32)
    binding, prices = model.soft_binding_and_prices(4)
    assert binding.requires_grad
    assert not prices.requires_grad

    full = X18RBindingModel(mode="dual_fullgrad", d_model=32)
    _, full_prices = full.soft_binding_and_prices(4)
    assert full_prices.requires_grad


def test_binding_generator_has_no_id_or_price_parameter_tables():
    for mode in LEARNED_X18R_MODES:
        model = X18RBindingModel(mode=mode, d_model=32)
        generator = model.binding_generator
        assert generator is not None
        for name, module in generator.named_modules():
            assert not isinstance(module, torch.nn.Embedding), name
        for name, _ in model.named_parameters():
            lower = name.lower()
            assert "external_id" not in lower
            assert "cardinality_id" not in lower
            assert "price_state" not in lower
            assert "dual_price" not in lower


def test_hidden_targets_and_semantics_do_not_change_losses():
    torch.manual_seed(18105)
    models = cloned_x18r_models(d_model=32)
    batch = make_variable_contextual_batch(12, 8, 18105, num_registers=4, split="train")
    for mode in LEARNED_X18R_MODES:
        model = models[mode]
        base = model.loss_components(batch)
        altered_targets = torch.randint_like(batch.target_states, VALUE_MODULUS)
        altered_targets[:, -1, 0] = batch.target_states[:, -1, 0]
        hidden_changed = replace(batch, target_states=altered_targets, semantics=torch.randint_like(batch.semantics, 8))
        same = model.loss_components(hidden_changed)
        for key in ("answer_loss", "spread_penalty", "barrier_penalty", "total_loss"):
            assert torch.equal(base[key], same[key])

        answer_changed = altered_targets.clone()
        answer_changed[:, -1, 0] = (batch.target_states[:, -1, 0] + 1) % VALUE_MODULUS
        changed = model.loss_components(replace(batch, target_states=answer_changed))
        assert not torch.allclose(base["answer_loss"], changed["answer_loss"], atol=1e-7, rtol=1e-7)
        assert torch.equal(base["spread_penalty"], changed["spread_penalty"])
        assert torch.equal(base["barrier_penalty"], changed["barrier_penalty"])


def test_losses_finite_and_gradients_reach_scorer_and_executor():
    torch.manual_seed(18106)
    models = cloned_x18r_models(d_model=32)
    batch = make_variable_contextual_batch(16, 8, 18106, num_registers=4, split="train")
    for mode, model in models.items():
        model.zero_grad(set_to_none=True)
        parts = model.loss_components(batch)
        for value in parts.values():
            assert torch.isfinite(value)
            assert float(value.detach()) >= 0.0
        parts["total_loss"].backward()
        executor_grad = sum(float(p.grad.abs().sum()) for p in model.executor.parameters() if p.grad is not None)
        assert executor_grad > 0.0
        if mode != "canonical_functional":
            scorer_grad = sum(float(p.grad.abs().sum()) for p in model.binding_generator.parameters() if p.grad is not None)
            assert scorer_grad > 0.0
