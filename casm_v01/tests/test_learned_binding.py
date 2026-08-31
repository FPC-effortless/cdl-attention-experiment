from __future__ import annotations

from copy import deepcopy

import torch

from casm.contextual_data import make_contextual_batch
from casm.learned_binding import (
    BINDING_MODES,
    BoundExplicitTransitionModel,
    cloned_binding_models,
)


def _assert_doubly_stochastic(matrix: torch.Tensor, atol: float = 1e-6):
    assert torch.isfinite(matrix).all()
    assert torch.all(matrix >= 0)
    assert torch.allclose(matrix.sum(dim=1), torch.ones(4), atol=atol, rtol=0)
    assert torch.allclose(matrix.sum(dim=0), torch.ones(4), atol=atol, rtol=0)


def _assert_categorical_mass(probs: torch.Tensor, atol: float = 1e-6):
    assert torch.isfinite(probs).all()
    assert torch.all(probs >= 0)
    assert torch.allclose(probs.sum(dim=-1), torch.ones_like(probs[..., 0]), atol=atol, rtol=0)


def test_binding_regimes_start_from_identical_parameter_tensors():
    torch.manual_seed(41)
    seed = BoundExplicitTransitionModel(d_model=32, binding_mode="learned_binding")
    models = cloned_binding_models(seed)
    reference = models["canonical_binding"].state_dict()
    for mode in ("learned_binding", "diffuse_binding"):
        candidate = models[mode].state_dict()
        assert reference.keys() == candidate.keys()
        for key in reference:
            assert torch.equal(reference[key], candidate[key])


def test_learned_binding_starts_near_uninformative_and_is_doubly_stochastic():
    torch.manual_seed(42)
    model = BoundExplicitTransitionModel(d_model=16, binding_mode="learned_binding")
    matrix = model.soft_binding()
    _assert_doubly_stochastic(matrix)
    assert float(matrix.max(dim=1).values.mean()) < 0.30


def test_exact_birkhoff_binding_remains_normalized_for_adversarial_sharp_scores():
    model = BoundExplicitTransitionModel(d_model=16, binding_mode="learned_binding")
    cases = [
        torch.tensor(
            [
                [80.0, -80.0, -80.0, -80.0],
                [80.0, -80.0, -80.0, -80.0],
                [-80.0, 80.0, -80.0, -80.0],
                [-80.0, -80.0, 80.0, -80.0],
            ]
        ),
        torch.tensor(
            [
                [40.0, 0.0, -40.0, 10.0],
                [-40.0, 40.0, 0.0, 10.0],
                [0.0, -40.0, 40.0, 10.0],
                [40.0, 40.0, 40.0, -40.0],
            ]
        ),
    ]
    for logits in cases:
        with torch.no_grad():
            model.binding_logits.copy_(logits)
        _assert_doubly_stochastic(model.soft_binding())


def test_canonical_and_diffuse_bindings_are_fixed_contracts():
    canonical = BoundExplicitTransitionModel(d_model=16, binding_mode="canonical_binding")
    diffuse = BoundExplicitTransitionModel(d_model=16, binding_mode="diffuse_binding")
    assert torch.equal(canonical.soft_binding(), torch.eye(4))
    assert torch.equal(diffuse.soft_binding(), torch.full((4, 4), 0.25))


def test_projected_learned_binding_is_a_one_to_one_permutation():
    torch.manual_seed(43)
    model = BoundExplicitTransitionModel(d_model=16, binding_mode="learned_binding")
    with torch.no_grad():
        model.binding_logits.copy_(
            torch.tensor(
                [
                    [0.0, 0.0, 5.0, 0.0],
                    [5.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 5.0],
                    [0.0, 5.0, 0.0, 0.0],
                ]
            )
        )
    projected, permutation, score = model.best_permutation()
    assert permutation == [2, 0, 3, 1]
    assert torch.equal(projected.sum(dim=1), torch.ones(4))
    assert torch.equal(projected.sum(dim=0), torch.ones(4))
    assert score > 3.8


def test_probability_transport_preserves_mass_in_both_directions():
    torch.manual_seed(45)
    model = BoundExplicitTransitionModel(d_model=16, binding_mode="learned_binding")
    with torch.no_grad():
        model.binding_logits.copy_(torch.randn(4, 4) * 12.0)
    binding = model.soft_binding()
    _assert_doubly_stochastic(binding)

    initial = torch.randint(0, 16, (32, 4))
    internal = model.initial_internal_probs(initial, binding)
    _assert_categorical_mass(internal)
    decoded = model.decode_external_probs(internal, binding)
    _assert_categorical_mass(decoded)


def test_fixed_answer_loss_ignores_intermediate_and_hidden_final_targets():
    batch = make_contextual_batch(16, 8, 9501, split="train")
    corrupted = deepcopy(batch)
    corrupted.target_states = batch.target_states.clone()
    corrupted.target_states[:, :-1] = torch.remainder(corrupted.target_states[:, :-1] + 1, 16)
    corrupted.target_states[:, -1, 1:] = torch.remainder(
        corrupted.target_states[:, -1, 1:] + 1, 16
    )
    model = BoundExplicitTransitionModel(d_model=32, binding_mode="learned_binding")
    a = model.fixed_answer_loss(batch)
    b = model.fixed_answer_loss(corrupted)
    assert torch.equal(a, b)


def test_fixed_answer_loss_changes_when_answer_target_changes():
    torch.manual_seed(44)
    batch = make_contextual_batch(16, 8, 9502, split="train")
    changed = deepcopy(batch)
    changed.target_states = batch.target_states.clone()
    changed.target_states[:, -1, 0] = torch.remainder(changed.target_states[:, -1, 0] + 1, 16)
    model = BoundExplicitTransitionModel(d_model=32, binding_mode="learned_binding")
    assert not torch.equal(model.fixed_answer_loss(batch), model.fixed_answer_loss(changed))


def test_fixed_answer_loss_ignores_private_semantic_operator_labels():
    batch = make_contextual_batch(16, 8, 9503, split="train")
    changed = deepcopy(batch)
    changed.semantics = torch.randint_like(batch.semantics, 0, 8)
    model = BoundExplicitTransitionModel(d_model=32, binding_mode="learned_binding")
    assert torch.equal(model.fixed_answer_loss(batch), model.fixed_answer_loss(changed))


def test_answer_loss_backpropagates_into_binding_and_transition():
    batch = make_contextual_batch(32, 8, 9504, split="train")
    model = BoundExplicitTransitionModel(d_model=32, binding_mode="learned_binding")
    loss = model.fixed_answer_loss(batch)
    assert torch.isfinite(loss) and float(loss.detach()) >= 0.0
    loss.backward()
    assert model.binding_logits.grad is not None
    assert torch.isfinite(model.binding_logits.grad).all()
    assert model.binding_logits.grad.abs().sum() > 0
    transition_grads = [p.grad for p in model.transition.parameters() if p.grad is not None]
    assert transition_grads
    assert all(torch.isfinite(g).all() for g in transition_grads)
    assert any(g.abs().sum() > 0 for g in transition_grads)


def test_probability_contract_survives_optimizer_update():
    torch.manual_seed(46)
    batch = make_contextual_batch(32, 8, 9506, split="train")
    model = BoundExplicitTransitionModel(d_model=32, binding_mode="learned_binding")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = model.fixed_answer_loss(batch)
        assert torch.isfinite(loss) and float(loss.detach()) >= 0.0
        loss.backward()
        optimizer.step()

        binding = model.soft_binding()
        _assert_doubly_stochastic(binding)
        internal = model.initial_internal_probs(batch.initial, binding)
        _assert_categorical_mass(internal)
        decoded = model.rollout_soft(batch)
        _assert_categorical_mass(decoded)
        post_loss = model.fixed_answer_loss(batch)
        assert torch.isfinite(post_loss) and float(post_loss.detach()) >= 0.0


def test_external_register_identity_has_no_direct_register_embedding_bypass():
    model = BoundExplicitTransitionModel(d_model=16, binding_mode="learned_binding")
    assert hasattr(model, "slot")
    assert not hasattr(model, "register")


def test_all_binding_modes_extend_to_depth_96():
    batch = make_contextual_batch(2, 96, 9505, split="composition")
    seed = BoundExplicitTransitionModel(d_model=16, binding_mode="learned_binding")
    models = cloned_binding_models(seed)
    assert set(models) == set(BINDING_MODES)
    for model in models.values():
        pred = model.rollout_hard(batch, discrete_binding=True)
        assert pred.shape == batch.target_states.shape
