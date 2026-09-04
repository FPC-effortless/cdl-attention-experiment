from __future__ import annotations

from dataclasses import replace

import torch

from casm.dual_horizon_binding import (
    DIAGNOSTIC_MODE,
    DUAL_ETA,
    LONG_DUAL_ROUNDS,
    SHORT_DUAL_ROUNDS,
    cloned_x17_models,
    primal_dual_binding_horizon,
    short_eval64_view,
)
from casm.explicit_compute import VALUE_MODULUS
from casm.primal_dual_capacity_binding import primal_dual_binding
from casm.variable_contextual_data import make_variable_contextual_batch


def test_horizons_and_eta_are_frozen():
    assert SHORT_DUAL_ROUNDS == 8
    assert LONG_DUAL_ROUNDS == 64
    assert DUAL_ETA == 1.0


def test_short_horizon_is_exact_x16_rule():
    torch.manual_seed(17001)
    logits = torch.randn(6, 8)
    x17_binding, x17_prices = primal_dual_binding_horizon(logits, rounds=8)
    x16_binding, x16_prices = primal_dual_binding(logits, enabled=True)
    assert torch.allclose(x17_binding, x16_binding, atol=0.0, rtol=0.0)
    assert torch.allclose(x17_prices, x16_prices, atol=0.0, rtol=0.0)


def test_both_horizons_preserve_probability_and_price_contracts():
    torch.manual_seed(17002)
    logits = torch.randn(6, 8)
    for rounds in (8, 64):
        binding, prices = primal_dual_binding_horizon(logits, rounds=rounds)
        assert torch.isfinite(binding).all()
        assert torch.isfinite(prices).all()
        assert (binding >= 0.0).all()
        assert (prices >= 0.0).all()
        assert torch.allclose(binding.sum(dim=1), torch.ones(6), atol=1e-6, rtol=0.0)


def test_long_horizon_is_row_and_slot_permutation_equivariant():
    torch.manual_seed(17003)
    logits = torch.randn(6, 8)
    row_perm = torch.tensor([5, 1, 4, 0, 3, 2])
    col_perm = torch.tensor([7, 2, 5, 0, 1, 6, 4, 3])
    base_b, base_p = primal_dual_binding_horizon(logits, rounds=64)
    row_b, row_p = primal_dual_binding_horizon(logits[row_perm], rounds=64)
    col_b, col_p = primal_dual_binding_horizon(logits[:, col_perm], rounds=64)
    assert torch.allclose(row_b, base_b[row_perm], atol=1e-6, rtol=1e-6)
    assert torch.allclose(row_p, base_p, atol=1e-6, rtol=1e-6)
    assert torch.allclose(col_b, base_b[:, col_perm], atol=1e-6, rtol=1e-6)
    assert torch.allclose(col_p, base_p[col_perm], atol=1e-6, rtol=1e-6)


def test_exact_symmetry_is_not_broken_at_either_horizon():
    logits = torch.zeros(6, 8)
    for rounds in (8, 64):
        binding, prices = primal_dual_binding_horizon(logits, rounds=rounds)
        assert torch.allclose(binding, torch.full((6, 8), 1.0 / 8.0), atol=1e-7, rtol=0.0)
        assert torch.equal(prices, torch.zeros(8))
        assert binding.argmax(dim=1).tolist() == [0, 0, 0, 0, 0, 0]


def test_short_and_long_models_begin_bit_identical_and_parameter_matched():
    torch.manual_seed(17004)
    models = cloned_x17_models(d_model=32)
    short = models["dual_short_8"]
    long = models["dual_long_64"]
    assert short.parameter_count() == long.parameter_count()
    assert short.trainable_parameter_count() == long.trainable_parameter_count()
    sstate = short.state_dict()
    lstate = long.state_dict()
    assert sstate.keys() == lstate.keys()
    assert all(torch.equal(sstate[k], lstate[k]) for k in sstate)


def test_structural_and_total_losses_ignore_hidden_targets_and_semantics():
    torch.manual_seed(17005)
    models = cloned_x17_models(d_model=32)
    batch = make_variable_contextual_batch(10, 8, 17005, num_registers=4, split="train")
    for mode in ("dual_short_8", "dual_long_64"):
        model = models[mode]
        base = model.loss_components(batch)
        altered_targets = torch.randint_like(batch.target_states, VALUE_MODULUS)
        altered_targets[:, -1, 0] = batch.target_states[:, -1, 0]
        hidden_changed = replace(batch, target_states=altered_targets, semantics=torch.randint_like(batch.semantics, 8))
        same = model.loss_components(hidden_changed)
        for key in ("answer_loss", "spread_penalty", "barrier_penalty", "total_loss"):
            assert torch.allclose(base[key], same[key], atol=0.0, rtol=0.0)

        answer_changed = altered_targets.clone()
        answer_changed[:, -1, 0] = (batch.target_states[:, -1, 0] + 1) % VALUE_MODULUS
        changed = model.loss_components(replace(batch, target_states=answer_changed))
        assert not torch.allclose(base["answer_loss"], changed["answer_loss"], atol=1e-7, rtol=1e-7)


def test_both_horizons_backpropagate_to_scorer_and_executor():
    torch.manual_seed(17006)
    models = cloned_x17_models(d_model=32)
    batch = make_variable_contextual_batch(8, 8, 17006, num_registers=4, split="train")
    for mode in ("dual_short_8", "dual_long_64"):
        model = models[mode]
        model.zero_grad(set_to_none=True)
        loss = model.loss_components(batch)["total_loss"]
        assert torch.isfinite(loss)
        loss.backward()
        assert sum(float(p.grad.abs().sum()) for p in model.executor.parameters() if p.grad is not None) > 0.0
        assert model.binding_generator is not None
        assert sum(float(p.grad.abs().sum()) for p in model.binding_generator.parameters() if p.grad is not None) > 0.0


def test_inference_only_long_view_copies_short_weights_exactly():
    torch.manual_seed(17007)
    models = cloned_x17_models(d_model=96)
    short = models["dual_short_8"]
    with torch.no_grad():
        for parameter in short.parameters():
            parameter.add_(torch.randn_like(parameter) * 1e-4)
    view = short_eval64_view(short)
    assert view.mode == DIAGNOSTIC_MODE
    assert view.rounds == 64
    sstate = short.state_dict()
    vstate = view.state_dict()
    assert sstate.keys() == vstate.keys()
    assert all(torch.equal(sstate[k], vstate[k]) for k in sstate)
