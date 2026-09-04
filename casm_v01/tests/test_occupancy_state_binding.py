from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn as nn

from casm.collision_barrier_binding import collision_barrier
from casm.coordinated_binding import slot_descriptor
from casm.explicit_compute import VALUE_MODULUS
from casm.occupancy_state_binding import (
    ITERATIVE_MODES,
    REFINEMENT_ROUNDS,
    OccupancyRefinementBindingGenerator,
    cloned_x14_models,
)
from casm.scarcity_binding import normalized_row_spread
from casm.variable_cardinality_binding import NUM_CANDIDATE_SLOTS, variable_descriptor
from casm.variable_contextual_data import (
    TRAIN_CARDINALITIES,
    make_variable_contextual_batch,
    training_cardinality_for_step,
)


def _descriptors(n: int, *, dtype=torch.float32):
    external = variable_descriptor(torch.arange(n), n, dtype=dtype)
    slots = slot_descriptor(torch.arange(NUM_CANDIDATE_SLOTS), dtype=dtype)
    return external, slots


def test_iterative_pair_begins_bit_identical_and_parameter_matched():
    torch.manual_seed(14001)
    models = cloned_x14_models(d_model=32)
    left = models["iterative_no_occupancy"]
    right = models["iterative_occupancy"]
    assert left.parameter_count() == right.parameter_count()
    assert left.trainable_parameter_count() == right.trainable_parameter_count()
    left_state = left.binding_generator.state_dict()
    right_state = right.binding_generator.state_dict()
    assert left_state.keys() == right_state.keys()
    assert all(torch.equal(left_state[key], right_state[key]) for key in left_state)
    assert all(
        torch.equal(left.executor.state_dict()[key], right.executor.state_dict()[key])
        for key in left.executor.state_dict()
    )


def test_refinement_round_count_is_exactly_frozen():
    models = cloned_x14_models(d_model=32)
    for mode in ITERATIVE_MODES:
        generator = models[mode].binding_generator
        assert isinstance(generator, OccupancyRefinementBindingGenerator)
        assert generator.refinement_rounds == REFINEMENT_ROUNDS == 8


def test_generated_other_occupancy_is_sum_of_other_rows_only():
    probabilities = torch.tensor(
        [
            [0.7, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    occupancy = OccupancyRefinementBindingGenerator.generated_other_occupancy(probabilities)
    total = probabilities.sum(dim=0, keepdim=True)
    assert torch.allclose(occupancy, total - probabilities, atol=0.0, rtol=0.0)
    for row in range(probabilities.shape[0]):
        assert torch.allclose(
            occupancy[row],
            probabilities[torch.arange(probabilities.shape[0]) != row].sum(dim=0),
            atol=0.0,
            rtol=0.0,
        )


def test_generated_occupancy_is_row_permutation_equivariant():
    torch.manual_seed(14002)
    probabilities = torch.softmax(torch.randn(6, 8), dim=-1)
    perm = torch.tensor([5, 1, 4, 0, 3, 2])
    base = OccupancyRefinementBindingGenerator.generated_other_occupancy(probabilities)
    permuted = OccupancyRefinementBindingGenerator.generated_other_occupancy(probabilities[perm])
    assert torch.allclose(permuted, base[perm], atol=1e-7, rtol=1e-7)


def test_occupancy_allocator_is_external_row_permutation_equivariant():
    torch.manual_seed(14003)
    model = cloned_x14_models(d_model=32)["iterative_occupancy"]
    generator = model.binding_generator
    assert isinstance(generator, OccupancyRefinementBindingGenerator)
    external, slots = _descriptors(6)
    perm = torch.tensor([5, 1, 4, 0, 3, 2])
    base = generator.probabilities_from_descriptors(external, slots)
    permuted = generator.probabilities_from_descriptors(external[perm], slots)
    assert torch.allclose(permuted, base[perm], atol=2e-6, rtol=2e-6)


def test_occupancy_allocator_is_slot_column_permutation_equivariant():
    torch.manual_seed(14004)
    model = cloned_x14_models(d_model=32)["iterative_occupancy"]
    generator = model.binding_generator
    assert isinstance(generator, OccupancyRefinementBindingGenerator)
    external, slots = _descriptors(6)
    perm = torch.tensor([7, 2, 5, 0, 1, 6, 4, 3])
    base = generator.probabilities_from_descriptors(external, slots)
    permuted = generator.probabilities_from_descriptors(external, slots[perm])
    assert torch.allclose(permuted, base[:, perm], atol=2e-6, rtol=2e-6)


def test_only_occupancy_variant_has_cross_row_descriptor_gradient():
    torch.manual_seed(14005)
    models = cloned_x14_models(d_model=32)
    slots = slot_descriptor(torch.arange(NUM_CANDIDATE_SLOTS))
    cross_norms = {}
    for mode in ITERATIVE_MODES:
        generator = models[mode].binding_generator
        assert isinstance(generator, OccupancyRefinementBindingGenerator)
        external = variable_descriptor(torch.arange(4), 4).detach().clone().requires_grad_(True)
        binding = generator.probabilities_from_descriptors(external, slots)
        focal = binding[0, 0]
        gradient = torch.autograd.grad(focal, external)[0]
        cross_norms[mode] = float(gradient[1:].abs().sum())
    assert cross_norms["iterative_occupancy"] > 1e-10, cross_norms
    assert cross_norms["iterative_no_occupancy"] < 1e-12, cross_norms


def test_no_binding_generator_contains_learned_identity_embedding_tables():
    models = cloned_x14_models(d_model=32)
    for mode in ("x13_one_shot_barrier",) + ITERATIVE_MODES:
        generator = models[mode].binding_generator
        assert generator is not None
        assert not any(isinstance(module, nn.Embedding) for module in generator.modules())


def test_one_shot_and_iterative_losses_use_exact_x13_barrier_objective():
    torch.manual_seed(14006)
    models = cloned_x14_models(d_model=32)
    batch = make_variable_contextual_batch(12, 8, 14006, num_registers=4, split="train")
    for mode in ("x13_one_shot_barrier",) + ITERATIVE_MODES:
        model = models[mode]
        parts = model.loss_components(batch)
        binding = model.soft_binding(4)
        assert torch.allclose(parts["spread_penalty"], normalized_row_spread(binding))
        assert torch.allclose(parts["barrier_penalty"], collision_barrier(binding))
        assert torch.allclose(
            parts["total_loss"],
            parts["answer_loss"] + parts["spread_penalty"] + parts["barrier_penalty"],
        )


def test_structural_gradients_reach_iterative_base_and_refiner():
    torch.manual_seed(14007)
    model = cloned_x14_models(d_model=32)["iterative_occupancy"]
    generator = model.binding_generator
    assert isinstance(generator, OccupancyRefinementBindingGenerator)
    binding = model.soft_binding(4)
    structural = normalized_row_spread(binding) + collision_barrier(binding)
    structural.backward()
    base_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in generator.base.parameters()
        if parameter.grad is not None
    )
    refiner_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in generator.refiner.parameters()
        if parameter.grad is not None
    )
    assert base_grad > 0.0
    assert refiner_grad > 0.0


def test_losses_ignore_hidden_intermediate_and_semantic_targets():
    torch.manual_seed(14008)
    models = cloned_x14_models(d_model=32)
    batch = make_variable_contextual_batch(12, 8, 14008, num_registers=4, split="train")
    for mode in ("x13_one_shot_barrier",) + ITERATIVE_MODES:
        model = models[mode]
        base = model.loss_components(batch)
        altered_targets = torch.randint_like(batch.target_states, VALUE_MODULUS)
        altered_targets[:, -1, 0] = batch.target_states[:, -1, 0]
        altered = replace(
            batch,
            target_states=altered_targets,
            semantics=torch.randint_like(batch.semantics, 8),
        )
        same = model.loss_components(altered)
        for key in ("answer_loss", "spread_penalty", "barrier_penalty", "total_loss"):
            assert torch.allclose(base[key], same[key], atol=0.0, rtol=0.0)

        answer_targets = altered_targets.clone()
        answer_targets[:, -1, 0] = (batch.target_states[:, -1, 0] + 1) % VALUE_MODULUS
        changed = model.loss_components(replace(batch, target_states=answer_targets))
        assert not torch.allclose(base["answer_loss"], changed["answer_loss"], atol=1e-7, rtol=1e-7)
        assert torch.allclose(base["spread_penalty"], changed["spread_penalty"], atol=0.0, rtol=0.0)
        assert torch.allclose(base["barrier_penalty"], changed["barrier_penalty"], atol=0.0, rtol=0.0)


def test_hard_argmax_preserves_collisions_without_repair():
    torch.manual_seed(14009)
    model = cloned_x14_models(d_model=32)["iterative_occupancy"]
    assert model.binding_generator is not None
    with torch.no_grad():
        for parameter in model.binding_generator.parameters():
            parameter.zero_()
    hard, assignment = model.independent_argmax_binding(6)
    assert assignment == [0, 0, 0, 0, 0, 0]
    assert hard.argmax(dim=1).tolist() == assignment
    stats = model.binding_stats(6)
    assert stats["independent_argmax_unique_slot_count"] == 1
    assert stats["independent_argmax_collision_count"] == 5


def test_training_schedule_is_only_234_repeated():
    observed = [training_cardinality_for_step(step) for step in range(1, 10001)]
    assert set(observed) == set(TRAIN_CARDINALITIES) == {2, 3, 4}
    assert observed[:9] == [2, 3, 4, 2, 3, 4, 2, 3, 4]
    assert observed.count(2) == 3334
    assert observed.count(3) == 3333
    assert observed.count(4) == 3333


def test_all_learned_losses_are_finite_and_backpropagate():
    torch.manual_seed(14010)
    models = cloned_x14_models(d_model=32)
    batch = make_variable_contextual_batch(16, 8, 14010, num_registers=4, split="train")
    for mode in ("x13_one_shot_barrier",) + ITERATIVE_MODES:
        model = models[mode]
        model.zero_grad(set_to_none=True)
        parts = model.loss_components(batch)
        for value in parts.values():
            assert torch.isfinite(value)
            assert float(value.detach()) >= 0.0
        parts["total_loss"].backward()
        binding_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.binding_generator.parameters()
            if parameter.grad is not None
        )
        executor_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.executor.parameters()
            if parameter.grad is not None
        )
        assert binding_grad > 0.0
        assert executor_grad > 0.0
