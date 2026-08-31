from __future__ import annotations

from dataclasses import replace

import torch

from casm.explicit_compute import VALUE_MODULUS
from casm.resource_competitive_binding import (
    COMPETITIVE_MODES,
    LEARNED_X11_MODES,
    RESOURCE_COMPETITION_LAMBDA,
    binding_overlap,
    cloned_x11_models,
)
from casm.variable_cardinality_binding import NUM_CANDIDATE_SLOTS
from casm.variable_contextual_data import make_variable_contextual_batch


def test_overlap_contract_exact_examples():
    uniform = torch.full((4, NUM_CANDIDATE_SLOTS), 1.0 / NUM_CANDIDATE_SLOTS)
    assert torch.allclose(binding_overlap(uniform), torch.tensor(0.125))

    distinct = torch.zeros(4, NUM_CANDIDATE_SLOTS)
    distinct[torch.arange(4), torch.arange(4)] = 1.0
    assert torch.allclose(binding_overlap(distinct), torch.tensor(0.0))

    same = torch.zeros(2, NUM_CANDIDATE_SLOTS)
    same[:, 3] = 1.0
    assert torch.allclose(binding_overlap(same), torch.tensor(1.0))


def test_overlap_is_row_and_column_permutation_invariant():
    torch.manual_seed(12001)
    logits = torch.randn(6, NUM_CANDIDATE_SLOTS)
    binding = torch.softmax(logits, dim=-1)
    row_perm = torch.tensor([5, 1, 4, 0, 3, 2])
    col_perm = torch.tensor([7, 2, 5, 0, 1, 6, 4, 3])
    base = binding_overlap(binding)
    assert torch.allclose(base, binding_overlap(binding[row_perm]), atol=1e-7, rtol=1e-7)
    assert torch.allclose(base, binding_overlap(binding[:, col_perm]), atol=1e-7, rtol=1e-7)


def test_all_relational_clones_begin_bit_identical():
    torch.manual_seed(12002)
    models = cloned_x11_models(d_model=32)
    ref_exec = models["canonical_functional"].executor.state_dict()
    for model in models.values():
        current = model.executor.state_dict()
        assert current.keys() == ref_exec.keys()
        assert all(torch.equal(current[k], ref_exec[k]) for k in ref_exec)

    ref_gen = models["relational_independent_no_competition"].binding_generator.state_dict()
    for mode in LEARNED_X11_MODES:
        current = models[mode].binding_generator.state_dict()
        assert current.keys() == ref_gen.keys()
        assert all(torch.equal(current[k], ref_gen[k]) for k in ref_gen)

    assert (
        models["relational_independent_competitive"].parameter_count()
        == models["relational_coordinated_competitive"].parameter_count()
    )


def test_competition_changes_only_loss_not_forward_binding_at_initialization():
    torch.manual_seed(12003)
    models = cloned_x11_models(d_model=32)
    for left, right in (
        ("relational_independent_no_competition", "relational_independent_competitive"),
        ("relational_coordinated_no_competition", "relational_coordinated_competitive"),
    ):
        for n in range(2, 7):
            assert torch.equal(models[left].soft_binding(n), models[right].soft_binding(n))


def test_competitive_total_loss_adds_exact_weighted_overlap():
    torch.manual_seed(12004)
    models = cloned_x11_models(d_model=32)
    batch = make_variable_contextual_batch(10, 8, 12004, num_registers=4, split="train")
    for mode in LEARNED_X11_MODES:
        parts = models[mode].loss_components(batch)
        expected_weight = RESOURCE_COMPETITION_LAMBDA if mode in COMPETITIVE_MODES else 0.0
        assert torch.allclose(parts["weighted_overlap"], parts["overlap_penalty"] * expected_weight)
        assert torch.allclose(parts["total_loss"], parts["answer_loss"] + parts["weighted_overlap"])


def test_overlap_gradients_reach_binding_generator():
    torch.manual_seed(12005)
    models = cloned_x11_models(d_model=32)
    model = models["relational_coordinated_competitive"]
    overlap = binding_overlap(model.soft_binding(4))
    overlap.backward()
    total = 0.0
    for parameter in model.binding_generator.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
            total += float(parameter.grad.abs().sum())
    assert total > 0.0


def test_total_loss_ignores_hidden_intermediate_and_semantic_targets():
    torch.manual_seed(12006)
    models = cloned_x11_models(d_model=32)
    batch = make_variable_contextual_batch(12, 8, 12006, num_registers=4, split="train")
    for mode in COMPETITIVE_MODES:
        model = models[mode]
        base = model.loss_components(batch)

        altered_targets = torch.randint_like(batch.target_states, VALUE_MODULUS)
        altered_targets[:, -1, 0] = batch.target_states[:, -1, 0]
        hidden_changed = replace(
            batch,
            target_states=altered_targets,
            semantics=torch.randint_like(batch.semantics, 8),
        )
        same = model.loss_components(hidden_changed)
        assert torch.allclose(base["answer_loss"], same["answer_loss"], atol=0.0, rtol=0.0)
        assert torch.allclose(base["overlap_penalty"], same["overlap_penalty"], atol=0.0, rtol=0.0)
        assert torch.allclose(base["total_loss"], same["total_loss"], atol=0.0, rtol=0.0)

        answer_changed_targets = altered_targets.clone()
        answer_changed_targets[:, -1, 0] = (batch.target_states[:, -1, 0] + 1) % VALUE_MODULUS
        changed = model.loss_components(replace(batch, target_states=answer_changed_targets))
        assert not torch.allclose(base["answer_loss"], changed["answer_loss"], atol=1e-7, rtol=1e-7)
        assert torch.allclose(base["overlap_penalty"], changed["overlap_penalty"], atol=0.0, rtol=0.0)


def test_hard_argmax_preserves_collisions_without_repair():
    torch.manual_seed(12007)
    models = cloned_x11_models(d_model=32)
    model = models["relational_independent_competitive"]
    generator = model.binding_generator
    with torch.no_grad():
        for parameter in generator.parameters():
            parameter.zero_()
    hard, assignment = model.independent_argmax_binding(6)
    assert assignment == [0, 0, 0, 0, 0, 0]
    assert hard.argmax(dim=1).tolist() == assignment
    stats = model.binding_stats(6)
    assert stats["independent_argmax_unique_slot_count"] == 1
    assert stats["independent_argmax_collision_count"] == 5


def test_competitive_loss_is_finite_and_backpropagates_through_executor_and_binding():
    torch.manual_seed(12008)
    models = cloned_x11_models(d_model=32)
    batch = make_variable_contextual_batch(16, 8, 12008, num_registers=4, split="train")
    for mode in COMPETITIVE_MODES:
        model = models[mode]
        model.zero_grad(set_to_none=True)
        parts = model.loss_components(batch)
        assert torch.isfinite(parts["answer_loss"])
        assert torch.isfinite(parts["overlap_penalty"])
        assert torch.isfinite(parts["total_loss"])
        assert float(parts["answer_loss"].detach()) >= 0.0
        assert float(parts["overlap_penalty"].detach()) >= 0.0
        parts["total_loss"].backward()
        binding_grad = sum(
            float(p.grad.abs().sum())
            for p in model.binding_generator.parameters()
            if p.grad is not None
        )
        executor_grad = sum(
            float(p.grad.abs().sum())
            for p in model.executor.parameters()
            if p.grad is not None
        )
        assert binding_grad > 0.0
        assert executor_grad > 0.0
