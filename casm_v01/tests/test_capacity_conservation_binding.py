from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F

from casm.capacity_conservation_binding import (
    CAPACITY_DAMPING,
    CAPACITY_EPSILON,
    CAPACITY_ROUNDS,
    capacity_refine_logits,
    cloned_x15_models,
    other_row_occupancy,
    remaining_capacity,
)
from casm.collision_barrier_binding import collision_barrier
from casm.coordinated_binding import slot_descriptor
from casm.explicit_compute import VALUE_MODULUS
from casm.scarcity_binding import normalized_row_spread
from casm.variable_cardinality_binding import variable_descriptor
from casm.variable_contextual_data import make_variable_contextual_batch


def test_capacity_constants_are_frozen():
    assert CAPACITY_ROUNDS == 8
    assert CAPACITY_EPSILON == 1e-3
    assert CAPACITY_DAMPING == 0.5


def test_other_row_occupancy_excludes_focal_row_and_capacity_examples():
    binding = torch.zeros(2, 8)
    binding[0, 0] = 1.0
    binding[1, 1] = 1.0
    other = other_row_occupancy(binding)
    assert torch.equal(other[0], binding[1])
    assert torch.equal(other[1], binding[0])

    capacity = remaining_capacity(binding)
    assert capacity[0, 0] == 1.0
    assert torch.isclose(capacity[0, 1], torch.tensor(CAPACITY_EPSILON))
    assert capacity[1, 1] == 1.0
    assert torch.isclose(capacity[1, 0], torch.tensor(CAPACITY_EPSILON))

    one_row = torch.zeros(1, 8)
    one_row[0, 3] = 1.0
    assert torch.equal(other_row_occupancy(one_row), torch.zeros_like(one_row))
    assert torch.equal(remaining_capacity(one_row), torch.ones_like(one_row))


def test_capacity_transform_preserves_probability_contract():
    torch.manual_seed(15001)
    logits = torch.randn(6, 8)
    for enabled in (False, True):
        binding = capacity_refine_logits(logits, enabled=enabled)
        assert torch.isfinite(binding).all()
        assert (binding >= 0.0).all()
        assert torch.allclose(binding.sum(dim=1), torch.ones(6), atol=1e-6, rtol=0.0)


def test_neutral_transform_equals_ordinary_row_softmax():
    torch.manual_seed(15002)
    logits = torch.randn(6, 8)
    expected = F.softmax(logits, dim=-1)
    actual = capacity_refine_logits(logits, enabled=False)
    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-7)


def test_capacity_transform_is_row_and_column_permutation_equivariant():
    torch.manual_seed(15003)
    logits = torch.randn(6, 8)
    row_perm = torch.tensor([5, 1, 4, 0, 3, 2])
    col_perm = torch.tensor([7, 2, 5, 0, 1, 6, 4, 3])
    base = capacity_refine_logits(logits, enabled=True)
    row_result = capacity_refine_logits(logits[row_perm], enabled=True)
    col_result = capacity_refine_logits(logits[:, col_perm], enabled=True)
    assert torch.allclose(row_result, base[row_perm], atol=1e-6, rtol=1e-6)
    assert torch.allclose(col_result, base[:, col_perm], atol=1e-6, rtol=1e-6)


def test_full_descriptor_to_capacity_binding_is_equivariant():
    torch.manual_seed(15004)
    model = cloned_x15_models(d_model=32)["capacity_conserving"]
    generator = model.binding_generator
    assert generator is not None
    parameter = next(generator.parameters())
    n = 6
    external = variable_descriptor(torch.arange(n), n, dtype=parameter.dtype)
    slots = slot_descriptor(torch.arange(8), dtype=parameter.dtype)
    logits = generator.logits_from_descriptors(external, slots)
    base = capacity_refine_logits(logits, enabled=True)

    row_perm = torch.tensor([4, 0, 5, 2, 1, 3])
    col_perm = torch.tensor([3, 7, 0, 4, 1, 6, 5, 2])
    row_logits = generator.logits_from_descriptors(external[row_perm], slots)
    col_logits = generator.logits_from_descriptors(external, slots[col_perm])
    assert torch.allclose(capacity_refine_logits(row_logits, enabled=True), base[row_perm], atol=1e-6, rtol=1e-6)
    assert torch.allclose(capacity_refine_logits(col_logits, enabled=True), base[:, col_perm], atol=1e-6, rtol=1e-6)


def test_other_row_intervention_affects_only_capacity_enabled_focal_row():
    base = torch.zeros(3, 8)
    base[0, 0] = 2.0
    base[1, 1] = 2.0
    base[2, 2] = 2.0
    altered = base.clone()
    altered[1, 0] = 8.0
    altered[1, 1] = -8.0

    neutral_base = capacity_refine_logits(base, enabled=False)
    neutral_altered = capacity_refine_logits(altered, enabled=False)
    assert torch.equal(neutral_base[0], neutral_altered[0])

    conserving_base = capacity_refine_logits(base, enabled=True)
    conserving_altered = capacity_refine_logits(altered, enabled=True)
    assert not torch.allclose(conserving_base[0], conserving_altered[0], atol=1e-6, rtol=1e-6)


def test_capacity_pair_begins_bit_identical_and_parameter_matched():
    torch.manual_seed(15005)
    models = cloned_x15_models(d_model=32)
    neutral = models["capacity_neutral"]
    conserving = models["capacity_conserving"]
    assert neutral.parameter_count() == conserving.parameter_count()
    assert neutral.trainable_parameter_count() == conserving.trainable_parameter_count()

    nstate = neutral.state_dict()
    cstate = conserving.state_dict()
    assert nstate.keys() == cstate.keys()
    assert all(torch.equal(nstate[k], cstate[k]) for k in nstate)


def test_binding_generator_has_no_external_or_slot_id_embedding_table():
    model = cloned_x15_models(d_model=32)["capacity_conserving"]
    generator = model.binding_generator
    assert generator is not None
    for name, module in generator.named_modules():
        assert not isinstance(module, torch.nn.Embedding), name
    for name, _ in generator.named_parameters():
        lower = name.lower()
        assert "external_id" not in lower
        assert "slot_id" not in lower
        assert "cardinality_id" not in lower


def test_capacity_structural_gradients_reach_base_scorer():
    torch.manual_seed(15006)
    model = cloned_x15_models(d_model=32)["capacity_conserving"]
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
    torch.manual_seed(15007)
    models = cloned_x15_models(d_model=32)
    batch = make_variable_contextual_batch(12, 8, 15007, num_registers=4, split="train")
    for mode in ("capacity_neutral", "capacity_conserving"):
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


def test_hard_argmax_preserves_exact_symmetry_collision():
    torch.manual_seed(15008)
    model = cloned_x15_models(d_model=32)["capacity_conserving"]
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
    torch.manual_seed(15009)
    models = cloned_x15_models(d_model=32)
    batch = make_variable_contextual_batch(16, 8, 15009, num_registers=4, split="train")
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
