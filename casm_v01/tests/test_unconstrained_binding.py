from __future__ import annotations

from copy import deepcopy

import torch

from casm.contextual_data import make_contextual_batch
from casm.unconstrained_binding import (
    BINDING_MODES,
    EMPTY_VALUE,
    NUM_CANDIDATE_SLOTS,
    UnconstrainedBindingTransitionModel,
    cloned_topology_models,
)


def _assert_row_stochastic(matrix: torch.Tensor, atol: float = 1e-6):
    assert torch.isfinite(matrix).all()
    assert torch.all(matrix >= -atol)
    assert torch.all(matrix <= 1.0 + atol)
    assert torch.allclose(matrix.sum(dim=1), torch.ones(4), atol=atol, rtol=0)
    assert abs(float(matrix.sum()) - 4.0) <= atol * 8


def _assert_categorical(probs: torch.Tensor, atol: float = 1e-6):
    assert torch.isfinite(probs).all()
    assert torch.all(probs >= -atol)
    assert torch.allclose(
        probs.sum(dim=-1),
        torch.ones_like(probs[..., 0]),
        atol=atol,
        rtol=0,
    )


def test_regimes_start_from_identical_parameters_and_are_parameter_matched():
    torch.manual_seed(71)
    seed = UnconstrainedBindingTransitionModel(d_model=32, binding_mode="learned_dense")
    models = cloned_topology_models(seed)
    assert set(models) == set(BINDING_MODES)
    counts = {mode: model.parameter_count() for mode, model in models.items()}
    trainable = {mode: model.trainable_parameter_count() for mode, model in models.items()}
    assert len(set(counts.values())) == 1
    assert len(set(trainable.values())) == 1
    reference = models["canonical_sparse"].state_dict()
    for mode in BINDING_MODES[1:]:
        candidate = models[mode].state_dict()
        assert reference.keys() == candidate.keys()
        for key in reference:
            assert torch.equal(reference[key], candidate[key])


def test_learned_dense_starts_near_uniform_and_row_normalized():
    torch.manual_seed(72)
    model = UnconstrainedBindingTransitionModel(d_model=16, binding_mode="learned_dense")
    matrix = model.soft_binding()
    _assert_row_stochastic(matrix)
    assert float(matrix.max(dim=1).values.mean()) < 0.15
    occupancy = matrix.sum(dim=0)
    assert torch.all(torch.abs(occupancy - 0.5) < 0.05), occupancy


def test_dense_binding_allows_adversarial_column_overload():
    model = UnconstrainedBindingTransitionModel(d_model=16, binding_mode="learned_dense")
    logits = torch.full((4, 8), -80.0)
    logits[:, 6] = 80.0
    with torch.no_grad():
        model.binding_logits.copy_(logits)
    matrix = model.soft_binding()
    _assert_row_stochastic(matrix)
    assert float(matrix[:, 6].sum()) > 3.999
    assert float(matrix.sum(dim=0).max()) > 1.0


def test_collision_transport_keeps_internal_and_external_probabilities_normalized():
    torch.manual_seed(73)
    batch = make_contextual_batch(16, 8, 9801, split="train")
    model = UnconstrainedBindingTransitionModel(d_model=16, binding_mode="learned_dense")
    logits = torch.full((4, 8), -80.0)
    logits[:, 6] = 80.0
    with torch.no_grad():
        model.binding_logits.copy_(logits)
    binding = model.soft_binding()
    internal = model.initial_internal_probs(batch.initial, binding)
    _assert_categorical(internal)
    decoded = model.decode_external_probs(internal, binding)
    _assert_categorical(decoded)

    new_world = torch.nn.functional.one_hot(
        torch.arange(16) % 16,
        16,
    ).to(dtype=internal.dtype)
    new_value = model.world_value_to_internal(new_world)
    updated = model.update_internal_state(
        internal,
        torch.zeros(16, dtype=torch.long),
        new_value,
        binding,
    )
    _assert_categorical(updated)
    _assert_categorical(model.decode_external_probs(updated, binding))


def test_dense_hard_projection_preserves_collisions_without_repair():
    model = UnconstrainedBindingTransitionModel(d_model=16, binding_mode="learned_dense")
    logits = torch.full((4, 8), -10.0)
    logits[:, 6] = 10.0
    with torch.no_grad():
        model.binding_logits.copy_(logits)
    hard, assignment = model.independent_argmax_binding()
    assert assignment == [6, 6, 6, 6]
    assert torch.equal(hard.sum(dim=1), torch.ones(4))
    assert int(hard[:, 6].sum().item()) == 4
    decisive = model.binding_matrix(discrete=True)
    assert torch.equal(decisive, hard)
    stats = model.binding_stats()
    assert stats["independent_argmax_unique_slot_count"] == 1
    assert stats["independent_argmax_collision_count"] == 3
    assert len(set(stats["best_injective_assignment"])) == 4


def test_learned_injective_control_retains_x7_contract():
    model = UnconstrainedBindingTransitionModel(d_model=16, binding_mode="learned_injective")
    logits = torch.tensor(
        [
            [80.0, 80.0, -80.0, -80.0, -80.0, -80.0, -80.0, -80.0],
            [80.0, 80.0, -80.0, -80.0, -80.0, -80.0, -80.0, -80.0],
            [80.0, 80.0, -80.0, -80.0, -80.0, -80.0, -80.0, -80.0],
            [80.0, 80.0, -80.0, -80.0, -80.0, -80.0, -80.0, -80.0],
        ]
    )
    with torch.no_grad():
        model.binding_logits.copy_(logits)
    matrix = model.soft_binding()
    _assert_row_stochastic(matrix)
    occupancy = matrix.sum(dim=0)
    assert torch.all(occupancy <= 1.0 + 1e-6)
    hard = model.binding_matrix(discrete=True)
    assert torch.equal(hard.sum(dim=1), torch.ones(4))
    assert torch.all(hard.sum(dim=0) <= 1.0)


def test_canonical_surplus_slots_are_empty_and_diffuse_is_uniform():
    torch.manual_seed(74)
    batch = make_contextual_batch(8, 8, 9802, split="train")
    canonical = UnconstrainedBindingTransitionModel(d_model=16, binding_mode="canonical_sparse")
    c = canonical.soft_binding()
    assert torch.equal(c[:, :4], torch.eye(4))
    assert torch.equal(c[:, 4:], torch.zeros(4, 4))
    internal = canonical.initial_internal_probs(batch.initial, c)
    _assert_categorical(internal)
    assert torch.allclose(
        internal[:, 4:, EMPTY_VALUE],
        torch.ones_like(internal[:, 4:, EMPTY_VALUE]),
    )

    diffuse = UnconstrainedBindingTransitionModel(d_model=16, binding_mode="diffuse_dense")
    d = diffuse.soft_binding()
    assert torch.equal(d, torch.full((4, 8), 0.125))
    _assert_row_stochastic(d)
    _assert_categorical(diffuse.initial_internal_probs(batch.initial, d))


def test_fixed_answer_loss_ignores_intermediate_hidden_and_semantic_targets():
    torch.manual_seed(75)
    batch = make_contextual_batch(16, 8, 9803, split="train")
    changed = deepcopy(batch)
    changed.target_states = batch.target_states.clone()
    changed.target_states[:, :-1] = torch.remainder(changed.target_states[:, :-1] + 1, 16)
    changed.target_states[:, -1, 1:] = torch.remainder(
        changed.target_states[:, -1, 1:] + 1,
        16,
    )
    changed.semantics = torch.randint_like(batch.semantics, 0, 8)
    model = UnconstrainedBindingTransitionModel(d_model=32, binding_mode="learned_dense")
    assert torch.equal(model.fixed_answer_loss(batch), model.fixed_answer_loss(changed))


def test_fixed_answer_loss_changes_when_visible_answer_changes():
    torch.manual_seed(76)
    batch = make_contextual_batch(16, 8, 9804, split="train")
    changed = deepcopy(batch)
    changed.target_states = batch.target_states.clone()
    changed.target_states[:, -1, 0] = torch.remainder(
        changed.target_states[:, -1, 0] + 1,
        16,
    )
    model = UnconstrainedBindingTransitionModel(d_model=32, binding_mode="learned_dense")
    assert not torch.equal(model.fixed_answer_loss(batch), model.fixed_answer_loss(changed))


def test_answer_loss_reaches_dense_binding_and_transition_parameters():
    batch = make_contextual_batch(32, 8, 9805, split="train")
    model = UnconstrainedBindingTransitionModel(d_model=32, binding_mode="learned_dense")
    loss = model.fixed_answer_loss(batch)
    assert torch.isfinite(loss) and float(loss.detach()) >= 0.0
    loss.backward()
    assert model.binding_logits.grad is not None
    assert torch.isfinite(model.binding_logits.grad).all()
    assert model.binding_logits.grad.abs().sum() > 0
    grads = [p.grad for p in model.transition.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_probability_contract_survives_dense_optimizer_updates():
    torch.manual_seed(77)
    batch = make_contextual_batch(32, 8, 9806, split="train")
    model = UnconstrainedBindingTransitionModel(d_model=32, binding_mode="learned_dense")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = model.fixed_answer_loss(batch)
        assert torch.isfinite(loss) and float(loss.detach()) >= 0.0
        loss.backward()
        optimizer.step()

        binding = model.soft_binding()
        _assert_row_stochastic(binding)
        internal = model.initial_internal_probs(batch.initial, binding)
        _assert_categorical(internal)
        decoded = model.rollout_soft(batch)
        _assert_categorical(decoded)
        post_loss = model.fixed_answer_loss(batch)
        assert torch.isfinite(post_loss) and float(post_loss.detach()) >= 0.0


def test_external_register_identity_has_no_direct_embedding_bypass():
    model = UnconstrainedBindingTransitionModel(d_model=16, binding_mode="learned_dense")
    assert hasattr(model, "slot")
    assert not hasattr(model, "register")


def test_all_regimes_extend_to_depth_96():
    batch = make_contextual_batch(2, 96, 9807, split="composition")
    seed = UnconstrainedBindingTransitionModel(d_model=16, binding_mode="learned_dense")
    models = cloned_topology_models(seed)
    assert set(models) == set(BINDING_MODES)
    for model in models.values():
        pred = model.rollout_hard(batch, discrete_binding=True)
        assert pred.shape == batch.target_states.shape
