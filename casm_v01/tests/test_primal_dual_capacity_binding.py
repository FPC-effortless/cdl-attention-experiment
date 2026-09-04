from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F

from casm.collision_barrier_binding import collision_barrier
from casm.coordinated_binding import slot_descriptor
from casm.explicit_compute import VALUE_MODULUS
from casm.primal_dual_capacity_binding import (
    DUAL_ETA,
    DUAL_ROUNDS,
    cloned_x16_models,
    primal_dual_binding,
    projected_dual_update,
)
from casm.scarcity_binding import normalized_row_spread
from casm.variable_cardinality_binding import variable_descriptor
from casm.variable_contextual_data import make_variable_contextual_batch


def test_dual_constants_are_frozen():
    assert DUAL_ROUNDS == 8
    assert DUAL_ETA == 1.0


def test_neutral_binding_equals_direct_softmax_and_zero_prices():
    torch.manual_seed(16001)
    logits = torch.randn(6, 8)
    binding, prices = primal_dual_binding(logits, enabled=False)
    assert torch.allclose(binding, F.softmax(logits, dim=-1), atol=1e-7, rtol=1e-7)
    assert torch.equal(prices, torch.zeros(8))


def test_priced_binding_preserves_probability_and_nonnegative_price_contract():
    torch.manual_seed(16002)
    logits = torch.randn(6, 8)
    binding, prices = primal_dual_binding(logits, enabled=True)
    assert torch.isfinite(binding).all()
    assert torch.isfinite(prices).all()
    assert (binding >= 0.0).all()
    assert (prices >= 0.0).all()
    assert torch.allclose(binding.sum(dim=1), torch.ones(6), atol=1e-6, rtol=0.0)


def test_projected_dual_update_examples():
    # All rows prefer slot 0, so the first projected update must raise slot-0 price.
    logits = torch.full((4, 8), -20.0)
    logits[:, 0] = 20.0
    initial = F.softmax(logits, dim=-1)
    initial_occupancy = initial.sum(dim=0)
    first = projected_dual_update(torch.zeros(8), initial_occupancy)
    assert float(first[0]) > 2.9
    assert torch.equal(first[1:], torch.zeros(7))

    # A positive price on an underloaded slot decreases by exactly the occupancy deficit,
    # projected at zero.
    price = torch.tensor([0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    occupancy = torch.tensor([0.5, 0.95, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    updated = projected_dual_update(price, occupancy)
    assert torch.allclose(updated[:2], torch.tensor([0.3, 0.15]), atol=1e-7, rtol=0.0)
    assert (updated >= 0.0).all()

    # Projection prevents negative prices after a large underload.
    projected = projected_dual_update(torch.tensor([0.1] + [0.0] * 7), torch.zeros(8))
    assert torch.equal(projected, torch.zeros(8))


def test_priced_allocator_is_row_and_column_permutation_equivariant():
    torch.manual_seed(16003)
    logits = torch.randn(6, 8)
    row_perm = torch.tensor([5, 1, 4, 0, 3, 2])
    col_perm = torch.tensor([7, 2, 5, 0, 1, 6, 4, 3])
    base_binding, base_prices = primal_dual_binding(logits, enabled=True)
    row_binding, row_prices = primal_dual_binding(logits[row_perm], enabled=True)
    col_binding, col_prices = primal_dual_binding(logits[:, col_perm], enabled=True)
    assert torch.allclose(row_binding, base_binding[row_perm], atol=1e-6, rtol=1e-6)
    assert torch.allclose(row_prices, base_prices, atol=1e-6, rtol=1e-6)
    assert torch.allclose(col_binding, base_binding[:, col_perm], atol=1e-6, rtol=1e-6)
    assert torch.allclose(col_prices, base_prices[col_perm], atol=1e-6, rtol=1e-6)


def test_full_descriptor_to_priced_binding_is_equivariant():
    torch.manual_seed(16004)
    model = cloned_x16_models(d_model=32)["dual_priced"]
    generator = model.binding_generator
    assert generator is not None
    parameter = next(generator.parameters())
    n = 6
    external = variable_descriptor(torch.arange(n), n, dtype=parameter.dtype)
    slots = slot_descriptor(torch.arange(8), dtype=parameter.dtype)
    logits = generator.logits_from_descriptors(external, slots)
    base, _ = primal_dual_binding(logits, enabled=True)

    row_perm = torch.tensor([4, 0, 5, 2, 1, 3])
    col_perm = torch.tensor([3, 7, 0, 4, 1, 6, 5, 2])
    row_logits = generator.logits_from_descriptors(external[row_perm], slots)
    col_logits = generator.logits_from_descriptors(external, slots[col_perm])
    row_result, _ = primal_dual_binding(row_logits, enabled=True)
    col_result, col_prices = primal_dual_binding(col_logits, enabled=True)
    _, base_prices = primal_dual_binding(logits, enabled=True)
    assert torch.allclose(row_result, base[row_perm], atol=1e-6, rtol=1e-6)
    assert torch.allclose(col_result, base[:, col_perm], atol=1e-6, rtol=1e-6)
    assert torch.allclose(col_prices, base_prices[col_perm], atol=1e-6, rtol=1e-6)


def test_other_row_intervention_affects_only_priced_focal_row():
    base = torch.zeros(3, 8)
    base[0, 0] = 2.0
    base[1, 1] = 2.0
    base[2, 2] = 2.0
    altered = base.clone()
    altered[1, 0] = 8.0
    altered[1, 1] = -8.0

    neutral_base, _ = primal_dual_binding(base, enabled=False)
    neutral_altered, _ = primal_dual_binding(altered, enabled=False)
    assert torch.equal(neutral_base[0], neutral_altered[0])

    priced_base, _ = primal_dual_binding(base, enabled=True)
    priced_altered, _ = primal_dual_binding(altered, enabled=True)
    assert not torch.allclose(priced_base[0], priced_altered[0], atol=1e-6, rtol=1e-6)


def test_dual_pair_begins_bit_identical_and_parameter_matched():
    torch.manual_seed(16005)
    models = cloned_x16_models(d_model=32)
    neutral = models["dual_neutral"]
    priced = models["dual_priced"]
    assert neutral.parameter_count() == priced.parameter_count()
    assert neutral.trainable_parameter_count() == priced.trainable_parameter_count()
    nstate = neutral.state_dict()
    pstate = priced.state_dict()
    assert nstate.keys() == pstate.keys()
    assert all(torch.equal(nstate[k], pstate[k]) for k in nstate)


def test_binding_generator_has_no_external_slot_or_cardinality_id_table():
    model = cloned_x16_models(d_model=32)["dual_priced"]
    generator = model.binding_generator
    assert generator is not None
    for name, module in generator.named_modules():
        assert not isinstance(module, torch.nn.Embedding), name
    for name, _ in generator.named_parameters():
        lower = name.lower()
        assert "external_id" not in lower
        assert "slot_id" not in lower
        assert "cardinality_id" not in lower


def test_structural_gradients_flow_through_price_dynamics_to_scorer():
    torch.manual_seed(16006)
    model = cloned_x16_models(d_model=32)["dual_priced"]
    binding = model.soft_binding(4)
    loss = normalized_row_spread(binding) + collision_barrier(binding)
    loss.backward()
    generator = model.binding_generator
    assert generator is not None
    total = 0.0
    for parameter in generator.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
            total += float(parameter.grad.abs().sum())
    assert total > 0.0


def test_structural_and_total_losses_ignore_hidden_targets_and_semantics():
    torch.manual_seed(16007)
    models = cloned_x16_models(d_model=32)
    batch = make_variable_contextual_batch(12, 8, 16007, num_registers=4, split="train")
    for mode in ("dual_neutral", "dual_priced"):
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
        assert torch.allclose(base["spread_penalty"], changed["spread_penalty"], atol=0.0, rtol=0.0)
        assert torch.allclose(base["barrier_penalty"], changed["barrier_penalty"], atol=0.0, rtol=0.0)


def test_exact_symmetry_remains_symmetric_and_hard_collisions_are_unrepaired():
    logits = torch.zeros(6, 8)
    binding, prices = primal_dual_binding(logits, enabled=True)
    assert torch.allclose(binding, torch.full((6, 8), 1.0 / 8.0), atol=1e-7, rtol=0.0)
    assert torch.equal(prices, torch.zeros(8))

    model = cloned_x16_models(d_model=32)["dual_priced"]
    generator = model.binding_generator
    assert generator is not None
    with torch.no_grad():
        for parameter in generator.parameters():
            parameter.zero_()
    hard, assignment = model.independent_argmax_binding(6)
    assert assignment == [0, 0, 0, 0, 0, 0]
    assert hard.argmax(dim=1).tolist() == assignment
    stats = model.binding_stats(6)
    assert stats["independent_argmax_unique_slot_count"] == 1
    assert stats["independent_argmax_collision_count"] == 5


def test_all_regime_losses_are_finite_and_backpropagate():
    torch.manual_seed(16008)
    models = cloned_x16_models(d_model=32)
    batch = make_variable_contextual_batch(16, 8, 16008, num_registers=4, split="train")
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
            generator = model.binding_generator
            assert generator is not None
            binding_grad = sum(float(p.grad.abs().sum()) for p in generator.parameters() if p.grad is not None)
            assert binding_grad > 0.0
