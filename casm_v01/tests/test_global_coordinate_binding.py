from __future__ import annotations

from dataclasses import replace

import torch

from casm.coordinated_binding import slot_descriptor
from casm.explicit_compute import VALUE_MODULUS
from casm.global_coordinate_binding import (
    LEARNED_X18_MODES,
    X18BindingModel,
    cloned_x18_models,
    global_variable_descriptor,
)
from casm.primal_dual_capacity_binding import X16BindingModel, primal_dual_binding
from casm.variable_cardinality_binding import NUM_CANDIDATE_SLOTS, variable_descriptor
from casm.variable_contextual_data import make_variable_contextual_batch


def test_relative_descriptor_is_exact_existing_descriptor():
    model = X18BindingModel(mode="relative_descriptor", d_model=32)
    for n in range(2, 7):
        indices = torch.arange(n)
        expected = variable_descriptor(indices, n)
        actual = model.external_descriptor(indices, n, dtype=torch.float32)
        assert torch.equal(actual, expected)


def test_global_descriptor_is_n_invariant_and_finite():
    for e in range(2):
        values = []
        for n in range(max(2, e + 1), 7):
            value = global_variable_descriptor(torch.tensor([e]), n)
            assert value.shape == (1, 9)
            assert torch.isfinite(value).all()
            values.append(value)
        assert all(torch.equal(values[0], value) for value in values[1:])

    # Wider indices are checked over every cardinality in which they exist.
    for e in range(2, 6):
        values = [global_variable_descriptor(torch.tensor([e]), n) for n in range(e + 1, 7)]
        assert all(torch.equal(values[0], value) for value in values[1:])


def test_global_descriptors_are_distinct_for_indices_zero_through_five():
    descriptors = global_variable_descriptor(torch.arange(6), 6)
    assert descriptors.shape == (6, 9)
    assert torch.unique(descriptors, dim=0).shape[0] == 6


def test_global_descriptor_uses_fixed_workspace_coordinate():
    descriptors = global_variable_descriptor(torch.arange(6), 6)
    assert torch.equal(descriptors[:, 1], torch.ones(6))
    assert torch.allclose(descriptors[:, 0], torch.arange(6, dtype=torch.float32) / 7.0)


def test_relative_x18_path_is_exact_x16_priced_path_when_weights_match():
    torch.manual_seed(18001)
    x18 = X18BindingModel(mode="relative_descriptor", d_model=32)
    x16 = X16BindingModel(mode="dual_priced", d_model=32)
    x16.core.load_state_dict(x18.core.state_dict())
    for n in range(2, 7):
        assert torch.equal(x18.base_logits(n), x16.base_logits(n))
        a, ap = x18.soft_binding_and_prices(n)
        b, bp = x16.soft_binding_and_prices(n)
        assert torch.equal(a, b)
        assert torch.equal(ap, bp)


def test_relative_and_global_models_begin_bit_identical_and_parameter_matched():
    torch.manual_seed(18002)
    models = cloned_x18_models(d_model=32)
    relative = models["relative_descriptor"]
    global_model = models["global_descriptor"]
    assert relative.parameter_count() == global_model.parameter_count()
    assert relative.trainable_parameter_count() == global_model.trainable_parameter_count()
    rs = relative.state_dict()
    gs = global_model.state_dict()
    assert rs.keys() == gs.keys()
    assert all(torch.equal(rs[key], gs[key]) for key in rs)


def test_descriptor_frame_is_only_learned_treatment_difference():
    torch.manual_seed(18003)
    models = cloned_x18_models(d_model=32)
    relative = models["relative_descriptor"]
    global_model = models["global_descriptor"]
    assert relative.mode != global_model.mode
    assert torch.equal(relative.binding_generator.state_dict()["external_proj.0.weight"], global_model.binding_generator.state_dict()["external_proj.0.weight"])
    # Different deterministic descriptors are allowed to induce different logits.
    assert not torch.equal(relative.base_logits(4), global_model.base_logits(4))


def test_binding_generator_has_no_id_embedding_tables():
    for mode in LEARNED_X18_MODES:
        model = X18BindingModel(mode=mode, d_model=32)
        generator = model.binding_generator
        assert generator is not None
        for name, module in generator.named_modules():
            assert not isinstance(module, torch.nn.Embedding), name
        for name, _ in generator.named_parameters():
            lower = name.lower()
            assert "external_id" not in lower
            assert "slot_id" not in lower
            assert "cardinality_id" not in lower


def test_global_complete_binding_is_row_and_slot_permutation_equivariant():
    torch.manual_seed(18004)
    model = X18BindingModel(mode="global_descriptor", d_model=32)
    generator = model.binding_generator
    assert generator is not None
    parameter = next(generator.parameters())
    n = 6
    external = global_variable_descriptor(torch.arange(n), n, dtype=parameter.dtype)
    slots = slot_descriptor(torch.arange(NUM_CANDIDATE_SLOTS), dtype=parameter.dtype)
    logits = generator.logits_from_descriptors(external, slots)
    base, prices = primal_dual_binding(logits, enabled=True)

    row_perm = torch.tensor([4, 0, 5, 2, 1, 3])
    col_perm = torch.tensor([3, 7, 0, 4, 1, 6, 5, 2])
    row_logits = generator.logits_from_descriptors(external[row_perm], slots)
    col_logits = generator.logits_from_descriptors(external, slots[col_perm])
    row_binding, row_prices = primal_dual_binding(row_logits, enabled=True)
    col_binding, col_prices = primal_dual_binding(col_logits, enabled=True)
    assert torch.allclose(row_binding, base[row_perm], atol=1e-6, rtol=1e-6)
    assert torch.allclose(row_prices, prices, atol=1e-6, rtol=1e-6)
    assert torch.allclose(col_binding, base[:, col_perm], atol=1e-6, rtol=1e-6)
    assert torch.allclose(col_prices, prices[col_perm], atol=1e-6, rtol=1e-6)


def test_hidden_targets_and_semantics_do_not_change_losses():
    torch.manual_seed(18005)
    models = cloned_x18_models(d_model=32)
    batch = make_variable_contextual_batch(12, 8, 18005, num_registers=4, split="train")
    for mode in LEARNED_X18_MODES:
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


def test_losses_are_finite_and_gradients_reach_scorer_and_executor():
    torch.manual_seed(18006)
    models = cloned_x18_models(d_model=32)
    batch = make_variable_contextual_batch(16, 8, 18006, num_registers=4, split="train")
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
