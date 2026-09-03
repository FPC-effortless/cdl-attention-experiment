from __future__ import annotations

from dataclasses import replace
import math

import torch

from casm.collision_barrier_binding import (
    BARRIER_EPSILON,
    BARRIER_MODES,
    LEARNED_X13_MODES,
    QUADRATIC_MODES,
    collision_barrier,
    cloned_x13_models,
)
from casm.explicit_compute import VALUE_MODULUS
from casm.scarcity_binding import normalized_row_spread, slot_capacity_overflow
from casm.variable_contextual_data import make_variable_contextual_batch


def test_barrier_exact_examples():
    distinct = torch.zeros(4, 8)
    distinct[torch.arange(4), torch.arange(4)] = 1.0
    assert torch.allclose(collision_barrier(distinct), torch.tensor(0.0), atol=1e-7, rtol=0.0)

    identical = torch.zeros(2, 8)
    identical[:, 3] = 1.0
    expected_identical = -math.log(BARRIER_EPSILON)
    assert math.isclose(float(collision_barrier(identical)), expected_identical, rel_tol=1e-5, abs_tol=1e-5)

    uniform = torch.full((4, 8), 1.0 / 8.0)
    expected_uniform = -math.log(1.0 - (1.0 - BARRIER_EPSILON) * (1.0 / 8.0))
    assert math.isclose(float(collision_barrier(uniform)), expected_uniform, rel_tol=1e-6, abs_tol=1e-6)


def test_barrier_is_row_and_column_permutation_invariant():
    torch.manual_seed(13001)
    binding = torch.softmax(torch.randn(6, 8), dim=-1)
    row_perm = torch.tensor([5, 1, 4, 0, 3, 2])
    col_perm = torch.tensor([7, 2, 5, 0, 1, 6, 4, 3])
    base = collision_barrier(binding)
    assert torch.allclose(base, collision_barrier(binding[row_perm]), atol=1e-7, rtol=1e-7)
    assert torch.allclose(base, collision_barrier(binding[:, col_perm]), atol=1e-7, rtol=1e-7)


def test_near_saturated_barrier_gradient_is_at_least_100x_quadratic_capacity():
    logits = torch.zeros(2, 8, requires_grad=True)
    with torch.no_grad():
        logits[:, 0] = 12.0
    binding = torch.softmax(logits, dim=-1)
    quadratic = slot_capacity_overflow(binding)
    barrier = collision_barrier(binding)
    quadratic_grad = torch.autograd.grad(quadratic, logits, retain_graph=True)[0].norm()
    barrier_grad = torch.autograd.grad(barrier, logits)[0].norm()
    assert torch.isfinite(quadratic_grad)
    assert torch.isfinite(barrier_grad)
    assert float(quadratic_grad) > 0.0
    assert float(barrier_grad) >= 100.0 * float(quadratic_grad), (quadratic_grad, barrier_grad)


def test_quadratic_and_barrier_pairs_begin_bit_identical():
    torch.manual_seed(13002)
    models = cloned_x13_models(d_model=32)
    ref_exec = models["canonical_functional"].executor.state_dict()
    for model in models.values():
        current = model.executor.state_dict()
        assert current.keys() == ref_exec.keys()
        assert all(torch.equal(current[k], ref_exec[k]) for k in ref_exec)

    ref_gen = models["relational_independent_quadratic"].binding_generator.state_dict()
    for mode in LEARNED_X13_MODES:
        current = models[mode].binding_generator.state_dict()
        assert current.keys() == ref_gen.keys()
        assert all(torch.equal(current[k], ref_gen[k]) for k in ref_gen)

    assert models["relational_independent_barrier"].parameter_count() == models["relational_coordinated_barrier"].parameter_count()


def test_paired_forward_bindings_are_identical_before_optimization():
    torch.manual_seed(13003)
    models = cloned_x13_models(d_model=32)
    for left, right in (
        ("relational_independent_quadratic", "relational_independent_barrier"),
        ("relational_coordinated_quadratic", "relational_coordinated_barrier"),
    ):
        for n in range(2, 7):
            assert torch.equal(models[left].soft_binding(n), models[right].soft_binding(n))


def test_loss_components_match_frozen_objectives():
    torch.manual_seed(13004)
    models = cloned_x13_models(d_model=32)
    batch = make_variable_contextual_batch(12, 8, 13004, num_registers=4, split="train")
    for mode in QUADRATIC_MODES:
        parts = models[mode].loss_components(batch)
        binding = models[mode].soft_binding(4)
        assert torch.allclose(parts["spread_penalty"], normalized_row_spread(binding))
        assert torch.allclose(parts["capacity_penalty"], slot_capacity_overflow(binding))
        assert torch.allclose(parts["barrier_penalty"], torch.zeros_like(parts["barrier_penalty"]))
        assert torch.allclose(parts["total_loss"], parts["answer_loss"] + parts["spread_penalty"] + parts["capacity_penalty"])
    for mode in BARRIER_MODES:
        parts = models[mode].loss_components(batch)
        binding = models[mode].soft_binding(4)
        assert torch.allclose(parts["spread_penalty"], normalized_row_spread(binding))
        assert torch.allclose(parts["capacity_penalty"], torch.zeros_like(parts["capacity_penalty"]))
        assert torch.allclose(parts["barrier_penalty"], collision_barrier(binding))
        assert torch.allclose(parts["total_loss"], parts["answer_loss"] + parts["spread_penalty"] + parts["barrier_penalty"])


def test_barrier_gradients_reach_binding_generator():
    torch.manual_seed(13005)
    model = cloned_x13_models(d_model=32)["relational_coordinated_barrier"]
    barrier = collision_barrier(model.soft_binding(4))
    barrier.backward()
    total = 0.0
    for parameter in model.binding_generator.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
            total += float(parameter.grad.abs().sum())
    assert total > 0.0


def test_structural_and_total_losses_ignore_hidden_intermediate_and_semantic_targets():
    torch.manual_seed(13006)
    models = cloned_x13_models(d_model=32)
    batch = make_variable_contextual_batch(12, 8, 13006, num_registers=4, split="train")
    for mode in QUADRATIC_MODES + BARRIER_MODES:
        model = models[mode]
        base = model.loss_components(batch)
        altered_targets = torch.randint_like(batch.target_states, VALUE_MODULUS)
        altered_targets[:, -1, 0] = batch.target_states[:, -1, 0]
        hidden_changed = replace(batch, target_states=altered_targets, semantics=torch.randint_like(batch.semantics, 8))
        same = model.loss_components(hidden_changed)
        for key in ("answer_loss", "spread_penalty", "capacity_penalty", "barrier_penalty", "total_loss"):
            assert torch.allclose(base[key], same[key], atol=0.0, rtol=0.0)

        answer_changed_targets = altered_targets.clone()
        answer_changed_targets[:, -1, 0] = (batch.target_states[:, -1, 0] + 1) % VALUE_MODULUS
        changed = model.loss_components(replace(batch, target_states=answer_changed_targets))
        assert not torch.allclose(base["answer_loss"], changed["answer_loss"], atol=1e-7, rtol=1e-7)
        assert torch.allclose(base["spread_penalty"], changed["spread_penalty"], atol=0.0, rtol=0.0)
        assert torch.allclose(base["capacity_penalty"], changed["capacity_penalty"], atol=0.0, rtol=0.0)
        assert torch.allclose(base["barrier_penalty"], changed["barrier_penalty"], atol=0.0, rtol=0.0)


def test_hard_argmax_preserves_collisions_without_repair():
    torch.manual_seed(13007)
    model = cloned_x13_models(d_model=32)["relational_independent_barrier"]
    with torch.no_grad():
        for parameter in model.binding_generator.parameters():
            parameter.zero_()
    hard, assignment = model.independent_argmax_binding(6)
    assert assignment == [0, 0, 0, 0, 0, 0]
    assert hard.argmax(dim=1).tolist() == assignment
    stats = model.binding_stats(6)
    assert stats["independent_argmax_unique_slot_count"] == 1
    assert stats["independent_argmax_collision_count"] == 5


def test_all_learned_losses_are_finite_and_backpropagate():
    torch.manual_seed(13008)
    models = cloned_x13_models(d_model=32)
    batch = make_variable_contextual_batch(16, 8, 13008, num_registers=4, split="train")
    for mode in QUADRATIC_MODES + BARRIER_MODES:
        model = models[mode]
        model.zero_grad(set_to_none=True)
        parts = model.loss_components(batch)
        for value in parts.values():
            assert torch.isfinite(value)
            assert float(value.detach()) >= 0.0
        parts["total_loss"].backward()
        binding_grad = sum(float(p.grad.abs().sum()) for p in model.binding_generator.parameters() if p.grad is not None)
        executor_grad = sum(float(p.grad.abs().sum()) for p in model.executor.parameters() if p.grad is not None)
        assert binding_grad > 0.0
        assert executor_grad > 0.0
