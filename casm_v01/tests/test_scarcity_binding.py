from __future__ import annotations

from dataclasses import replace

import torch

from casm.explicit_compute import VALUE_MODULUS
from casm.scarcity_binding import (
    LEARNED_X12_MODES,
    SCARCITY_MODES,
    X12_MODES,
    cloned_x12_models,
    normalized_row_spread,
    slot_capacity_overflow,
)
from casm.variable_cardinality_binding import NUM_CANDIDATE_SLOTS
from casm.variable_contextual_data import make_variable_contextual_batch


def test_scarcity_exact_examples():
    uniform = torch.full((4, NUM_CANDIDATE_SLOTS), 1.0 / NUM_CANDIDATE_SLOTS)
    assert torch.allclose(normalized_row_spread(uniform), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(slot_capacity_overflow(uniform), torch.tensor(0.0), atol=1e-7)

    distinct = torch.zeros(4, NUM_CANDIDATE_SLOTS)
    distinct[torch.arange(4), torch.arange(4)] = 1.0
    assert torch.allclose(normalized_row_spread(distinct), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(slot_capacity_overflow(distinct), torch.tensor(0.0), atol=1e-7)

    same = torch.zeros(4, NUM_CANDIDATE_SLOTS)
    same[:, 3] = 1.0
    expected = torch.tensor((4.0 - 1.0) ** 2 / 4.0)
    assert torch.allclose(normalized_row_spread(same), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(slot_capacity_overflow(same), expected, atol=1e-7)


def test_scarcity_is_row_and_column_permutation_invariant():
    torch.manual_seed(13001)
    b = torch.softmax(torch.randn(6, NUM_CANDIDATE_SLOTS), dim=-1)
    rp = torch.tensor([5, 1, 4, 0, 3, 2])
    cp = torch.tensor([7, 2, 5, 0, 1, 6, 4, 3])
    for fn in (normalized_row_spread, slot_capacity_overflow):
        base = fn(b)
        assert torch.allclose(base, fn(b[rp]), atol=1e-7, rtol=1e-7)
        assert torch.allclose(base, fn(b[:, cp]), atol=1e-7, rtol=1e-7)


def test_x12_relational_clones_begin_bit_identical():
    torch.manual_seed(13002)
    models = cloned_x12_models(d_model=32)
    ref_exec = models["canonical_functional"].executor.state_dict()
    for model in models.values():
        current = model.executor.state_dict()
        assert current.keys() == ref_exec.keys()
        assert all(torch.equal(current[k], ref_exec[k]) for k in ref_exec)

    ref_gen = models["relational_independent_overlap"].binding_generator.state_dict()
    for mode in LEARNED_X12_MODES:
        current = models[mode].binding_generator.state_dict()
        assert current.keys() == ref_gen.keys()
        assert all(torch.equal(current[k], ref_gen[k]) for k in ref_gen)

    assert models["relational_independent_scarcity"].parameter_count() == models["relational_coordinated_scarcity"].parameter_count()


def test_overlap_and_scarcity_pair_forward_bindings_match_initially():
    torch.manual_seed(13003)
    models = cloned_x12_models(d_model=32)
    for left, right in (
        ("relational_independent_overlap", "relational_independent_scarcity"),
        ("relational_coordinated_overlap", "relational_coordinated_scarcity"),
    ):
        for n in range(2, 7):
            assert torch.equal(models[left].soft_binding(n), models[right].soft_binding(n))


def test_scarcity_gradients_reach_binding_generator_off_symmetric_point():
    torch.manual_seed(13004)
    models = cloned_x12_models(d_model=32)
    model = models["relational_coordinated_scarcity"]
    b = model.soft_binding(4)
    objective = normalized_row_spread(b) + slot_capacity_overflow(b)
    objective.backward()
    total = sum(float(p.grad.abs().sum()) for p in model.binding_generator.parameters() if p.grad is not None)
    assert total > 0.0


def test_scarcity_total_loss_ignores_hidden_targets_and_semantics():
    torch.manual_seed(13005)
    models = cloned_x12_models(d_model=32)
    batch = make_variable_contextual_batch(12, 8, 13005, num_registers=4, split="train")
    for mode in SCARCITY_MODES:
        model = models[mode]
        base = model.loss_components(batch)
        altered = torch.randint_like(batch.target_states, VALUE_MODULUS)
        altered[:, -1, 0] = batch.target_states[:, -1, 0]
        changed_hidden = replace(batch, target_states=altered, semantics=torch.randint_like(batch.semantics, 8))
        same = model.loss_components(changed_hidden)
        for key in ("answer_loss", "spread_penalty", "capacity_penalty", "total_loss"):
            assert torch.allclose(base[key], same[key], atol=0.0, rtol=0.0)

        answer_changed = altered.clone()
        answer_changed[:, -1, 0] = (batch.target_states[:, -1, 0] + 1) % VALUE_MODULUS
        diff = model.loss_components(replace(batch, target_states=answer_changed))
        assert not torch.allclose(base["answer_loss"], diff["answer_loss"], atol=1e-7, rtol=1e-7)
        assert torch.allclose(base["spread_penalty"], diff["spread_penalty"], atol=0.0, rtol=0.0)
        assert torch.allclose(base["capacity_penalty"], diff["capacity_penalty"], atol=0.0, rtol=0.0)


def test_hard_argmax_still_preserves_collisions():
    torch.manual_seed(13006)
    models = cloned_x12_models(d_model=32)
    model = models["relational_independent_scarcity"]
    with torch.no_grad():
        for parameter in model.binding_generator.parameters():
            parameter.zero_()
    hard, assignment = model.independent_argmax_binding(6)
    assert assignment == [0, 0, 0, 0, 0, 0]
    assert hard.argmax(dim=1).tolist() == assignment
    stats = model.binding_stats(6)
    assert stats["independent_argmax_unique_slot_count"] == 1
    assert stats["independent_argmax_collision_count"] == 5


def test_all_training_losses_are_finite_nonnegative_and_backpropagate():
    torch.manual_seed(13007)
    models = cloned_x12_models(d_model=32)
    batch = make_variable_contextual_batch(16, 8, 13007, num_registers=4, split="train")
    for mode in X12_MODES:
        model = models[mode]
        model.zero_grad(set_to_none=True)
        parts = model.loss_components(batch)
        for key in ("answer_loss", "overlap_penalty", "spread_penalty", "capacity_penalty", "total_loss"):
            assert torch.isfinite(parts[key])
            assert float(parts[key].detach()) >= 0.0
        parts["total_loss"].backward()
        executor_grad = sum(float(p.grad.abs().sum()) for p in model.executor.parameters() if p.grad is not None)
        assert executor_grad > 0.0
        if mode != "canonical_functional":
            binding_grad = sum(float(p.grad.abs().sum()) for p in model.binding_generator.parameters() if p.grad is not None)
            assert binding_grad > 0.0
