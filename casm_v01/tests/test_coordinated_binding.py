from __future__ import annotations

from dataclasses import replace

import torch

from casm.coordinated_binding import (
    LEARNED_MODES,
    DirectIndependentBindingGenerator,
    RelationalBindingGenerator,
    X10BindingModel,
    cloned_x10_models,
    slot_descriptor,
)
from casm.explicit_compute import VALUE_MODULUS
from casm.variable_cardinality_binding import NUM_CANDIDATE_SLOTS, variable_descriptor
from casm.variable_contextual_data import make_variable_contextual_batch


def test_binding_generators_have_no_learned_external_or_slot_id_embeddings():
    direct = DirectIndependentBindingGenerator(d_model=32)
    independent = RelationalBindingGenerator(d_model=32, coordinated=False)
    coordinated = RelationalBindingGenerator(d_model=32, coordinated=True)
    assert not any(isinstance(module, torch.nn.Embedding) for module in direct.modules())
    assert not any(isinstance(module, torch.nn.Embedding) for module in independent.modules())
    assert not any(isinstance(module, torch.nn.Embedding) for module in coordinated.modules())


def test_slot_descriptor_is_deterministic_and_finite():
    slots = torch.arange(NUM_CANDIDATE_SLOTS)
    first = slot_descriptor(slots)
    second = slot_descriptor(slots)
    assert torch.equal(first, second)
    assert first.shape == (NUM_CANDIDATE_SLOTS, 8)
    assert torch.isfinite(first).all()
    assert torch.all(first[:, 0] >= 0.0)
    assert torch.all(first[:, 0] <= 1.0)


def test_external_descriptor_contract_remains_x9_deterministic():
    for n in range(2, 7):
        indices = torch.arange(n)
        first = variable_descriptor(indices, n)
        second = variable_descriptor(indices, n)
        assert torch.equal(first, second)
        assert torch.isfinite(first).all()


def test_relational_external_row_permutation_equivariance():
    torch.manual_seed(11001)
    generator = RelationalBindingGenerator(d_model=32, coordinated=True).eval()
    n = 6
    ext = variable_descriptor(torch.arange(n), n)
    slots = slot_descriptor(torch.arange(NUM_CANDIDATE_SLOTS))
    permutation = torch.tensor([5, 2, 0, 4, 1, 3])
    original = generator.probabilities_from_descriptors(ext, slots)
    permuted = generator.probabilities_from_descriptors(ext[permutation], slots)
    assert torch.allclose(permuted, original[permutation], atol=1e-6, rtol=1e-6)


def test_relational_slot_descriptor_permutation_equivariance():
    torch.manual_seed(11002)
    for coordinated in (False, True):
        generator = RelationalBindingGenerator(d_model=32, coordinated=coordinated).eval()
        ext = variable_descriptor(torch.arange(5), 5)
        slots = slot_descriptor(torch.arange(NUM_CANDIDATE_SLOTS))
        permutation = torch.tensor([7, 3, 5, 0, 2, 6, 1, 4])
        original = generator.probabilities_from_descriptors(ext, slots)
        permuted = generator.probabilities_from_descriptors(ext, slots[permutation])
        assert torch.allclose(permuted, original[:, permutation], atol=1e-6, rtol=1e-6)


def test_independent_has_no_cross_row_gradient_but_coordinated_does():
    torch.manual_seed(11003)
    n = 5
    base = RelationalBindingGenerator(d_model=32, coordinated=False)
    coordinated = RelationalBindingGenerator(d_model=32, coordinated=True)
    coordinated.load_state_dict(base.state_dict())
    slots = slot_descriptor(torch.arange(NUM_CANDIDATE_SLOTS))

    ext_ind = variable_descriptor(torch.arange(n), n).requires_grad_(True)
    independent_score = base.logits_from_descriptors(ext_ind, slots)[0].sum()
    independent_grad = torch.autograd.grad(independent_score, ext_ind)[0]
    assert torch.equal(independent_grad[1:], torch.zeros_like(independent_grad[1:]))

    ext_coord = variable_descriptor(torch.arange(n), n).requires_grad_(True)
    coordinated_score = coordinated.logits_from_descriptors(ext_coord, slots)[0].sum()
    coordinated_grad = torch.autograd.grad(coordinated_score, ext_coord)[0]
    assert coordinated_grad[1:].abs().sum() > 0.0


def test_relational_pair_is_parameter_matched_and_bit_identical():
    torch.manual_seed(11004)
    models = cloned_x10_models(d_model=32)
    independent = models["relational_independent"]
    coordinated = models["relational_coordinated"]
    assert independent.parameter_count() == coordinated.parameter_count()
    left = independent.binding_generator.state_dict()
    right = coordinated.binding_generator.state_dict()
    assert left.keys() == right.keys()
    assert all(torch.equal(left[key], right[key]) for key in left)


def test_all_regimes_start_from_identical_executor_state():
    torch.manual_seed(11005)
    models = cloned_x10_models(d_model=32)
    reference = models["canonical_functional"].executor.state_dict()
    for model in models.values():
        current = model.executor.state_dict()
        assert current.keys() == reference.keys()
        assert all(torch.equal(current[key], reference[key]) for key in reference)


def test_initial_learned_bindings_are_row_stochastic_near_uniform_and_finite():
    torch.manual_seed(11006)
    models = cloned_x10_models(d_model=32)
    for mode in LEARNED_MODES:
        model = models[mode]
        for n in range(2, 7):
            stats = model.binding_stats(n)
            matrix = model.soft_binding(n)
            assert matrix.shape == (n, NUM_CANDIDATE_SLOTS)
            assert torch.isfinite(matrix).all()
            assert torch.allclose(matrix.sum(dim=1), torch.ones(n), atol=1e-6)
            assert abs(stats["total_binding_mass"] - n) <= 1e-5
            assert stats["row_max_mean"] < 0.18


def test_capacity_transport_stays_normalized_under_adversarial_collision():
    model = X10BindingModel(mode="relational_coordinated", d_model=32)
    batch = make_variable_contextual_batch(4, 8, 11007, num_registers=6, split="train")
    collision = torch.zeros(6, NUM_CANDIDATE_SLOTS)
    collision[:, 0] = 1.0
    probs = model.executor.initial_internal_probs(batch.initial, collision)
    assert torch.allclose(probs.sum(dim=-1), torch.ones_like(probs[..., 0]), atol=1e-6)
    decoded = model.executor.decode_external_probs(probs, collision)
    assert torch.allclose(decoded.sum(dim=-1), torch.ones_like(decoded[..., 0]), atol=1e-6)


def test_hard_argmax_preserves_collisions_without_repair():
    model = X10BindingModel(mode="x9_direct_independent", d_model=32)
    generator = model.binding_generator
    assert isinstance(generator, DirectIndependentBindingGenerator)
    with torch.no_grad():
        for parameter in generator.parameters():
            parameter.zero_()
    hard, assignment = model.independent_argmax_binding(6)
    assert assignment == [0, 0, 0, 0, 0, 0]
    assert hard.argmax(dim=1).tolist() == assignment
    stats = model.binding_stats(6)
    assert stats["independent_argmax_unique_slot_count"] == 1
    assert stats["independent_argmax_collision_count"] == 5


def test_fixed_answer_loss_ignores_hidden_intermediate_and_semantic_targets():
    torch.manual_seed(11008)
    model = X10BindingModel(mode="relational_coordinated", d_model=32)
    batch = make_variable_contextual_batch(12, 8, 11008, num_registers=4, split="train")
    base = model.fixed_answer_loss(batch)

    altered_targets = torch.randint_like(batch.target_states, VALUE_MODULUS)
    altered_targets[:, -1, 0] = batch.target_states[:, -1, 0]
    hidden_changed = replace(
        batch,
        target_states=altered_targets,
        semantics=torch.randint_like(batch.semantics, 8),
    )
    same = model.fixed_answer_loss(hidden_changed)
    assert torch.allclose(base, same, atol=0.0, rtol=0.0)

    answer_changed_targets = altered_targets.clone()
    answer_changed_targets[:, -1, 0] = (batch.target_states[:, -1, 0] + 1) % VALUE_MODULUS
    changed = model.fixed_answer_loss(replace(batch, target_states=answer_changed_targets))
    assert not torch.allclose(base, changed, atol=1e-7, rtol=1e-7)


def test_gradients_reach_binding_generator_and_executor_for_all_learned_regimes():
    torch.manual_seed(11009)
    models = cloned_x10_models(d_model=32)
    batch = make_variable_contextual_batch(16, 8, 11009, num_registers=4, split="train")
    for mode in LEARNED_MODES:
        model = models[mode]
        model.zero_grad(set_to_none=True)
        loss = model.fixed_answer_loss(batch)
        loss.backward()
        assert torch.isfinite(loss)
        generator_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.binding_generator.parameters()
            if parameter.grad is not None
        )
        executor_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.executor.parameters()
            if parameter.grad is not None
        )
        assert generator_grad > 0.0
        assert executor_grad > 0.0


def test_all_x10_regimes_roll_out_all_cardinalities_through_depth_96():
    torch.manual_seed(11010)
    models = cloned_x10_models(d_model=16)
    for n in range(2, 7):
        batch = make_variable_contextual_batch(2, 96, 12000 + n, num_registers=n, split="composition")
        for model in models.values():
            soft = model.rollout_soft(batch)
            hard = model.rollout_hard(batch, discrete_binding=(model.mode != "canonical_functional"))
            assert soft.shape == (2, 96, n, VALUE_MODULUS + 1)
            assert hard.shape == (2, 96, n)
            assert torch.isfinite(soft).all()
