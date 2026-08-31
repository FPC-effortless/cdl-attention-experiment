from __future__ import annotations

from dataclasses import replace

import torch

from casm.cardinality_valid_executor import LocalEquivariantTransitionModel
from casm.explicit_compute import ProgramBatch, VALUE_MODULUS
from casm.variable_cardinality_binding import NUM_CANDIDATE_SLOTS
from casm.variable_contextual_data import (
    TRAIN_CARDINALITIES,
    make_variable_contextual_batch,
    training_cardinality_for_step,
)


def _permuted_binding(model: LocalEquivariantTransitionModel, n: int) -> torch.Tensor:
    canonical = model.binding_matrix(n)
    permutation = torch.tensor([6, 4, 7, 1, 5, 0, 3, 2], dtype=torch.long)
    return canonical[:, permutation]


def test_local_executor_has_no_absolute_slot_identity_path():
    model = LocalEquivariantTransitionModel(d_model=32)
    assert not hasattr(model, "slot")
    assert not hasattr(model, "state_proj")
    assert not hasattr(model, "command_proj")
    embeddings = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Embedding)]
    assert embeddings == ["value", "command"]
    assert model.transition[0].in_features == 4 * model.d_model


def test_slot_permutation_is_externally_equivariant_soft_and_hard():
    torch.manual_seed(123)
    model = LocalEquivariantTransitionModel(d_model=32).eval()
    batch = make_variable_contextual_batch(8, 12, 501, num_registers=6, split="composition")
    canonical = model.binding_matrix(6)
    permuted = _permuted_binding(model, 6)
    soft_a = model.rollout_soft_with_binding(batch, canonical)
    soft_b = model.rollout_soft_with_binding(batch, permuted)
    assert torch.allclose(soft_a, soft_b, atol=1e-6, rtol=1e-6)
    hard_a = model.rollout_hard_with_binding(batch, canonical)
    hard_b = model.rollout_hard_with_binding(batch, permuted)
    assert torch.equal(hard_a, hard_b)


def test_canonical_bindings_are_collision_free_and_probability_valid():
    model = LocalEquivariantTransitionModel(d_model=16)
    for n in range(2, 7):
        binding = model.binding_matrix(n)
        assert binding.shape == (n, NUM_CANDIDATE_SLOTS)
        assert torch.allclose(binding.sum(dim=1), torch.ones(n))
        assert binding.argmax(dim=1).unique().numel() == n
        batch = make_variable_contextual_batch(3, 4, 700 + n, num_registers=n)
        probs = model.initial_internal_probs(batch.initial, binding)
        assert torch.allclose(probs.sum(dim=-1), torch.ones_like(probs[..., 0]), atol=1e-6)
        decoded = model.decode_external_probs(probs, binding)
        assert torch.allclose(decoded.sum(dim=-1), torch.ones_like(decoded[..., 0]), atol=1e-6)


def test_fixed_answer_loss_ignores_hidden_intermediate_and_semantic_targets():
    torch.manual_seed(321)
    model = LocalEquivariantTransitionModel(d_model=32)
    batch = make_variable_contextual_batch(12, 8, 811, num_registers=4)
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
    answer_changed = replace(batch, target_states=answer_changed_targets)
    changed = model.fixed_answer_loss(answer_changed)
    assert not torch.allclose(base, changed, atol=1e-7, rtol=1e-7)


def test_gradients_reach_local_transition_value_and_command_parameters():
    torch.manual_seed(456)
    model = LocalEquivariantTransitionModel(d_model=32)
    batch = make_variable_contextual_batch(16, 8, 912, num_registers=3)
    loss = model.fixed_answer_loss(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.value.weight.grad is not None
    assert model.command.weight.grad is not None
    for parameter in model.transition.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_all_cardinalities_roll_out_through_depth_96():
    model = LocalEquivariantTransitionModel(d_model=16).eval()
    for n in range(2, 7):
        batch = make_variable_contextual_batch(2, 96, 1000 + n, num_registers=n, split="composition")
        soft = model.rollout_soft(batch)
        hard = model.rollout_hard(batch)
        assert soft.shape == (2, 96, n, VALUE_MODULUS + 1)
        assert hard.shape == (2, 96, n)
        assert torch.isfinite(soft).all()


def test_training_cardinality_schedule_is_exactly_234_repeated():
    observed = [training_cardinality_for_step(i) for i in range(1, 16)]
    expected = list(TRAIN_CARDINALITIES) * 5
    assert observed == expected
