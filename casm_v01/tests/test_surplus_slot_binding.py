from __future__ import annotations

from copy import deepcopy

import torch

from casm.contextual_data import make_contextual_batch
from casm.surplus_slot_binding import (
    BINDING_MODES,
    EMPTY_VALUE,
    NUM_CANDIDATE_SLOTS,
    SurplusSlotTransitionModel,
    cloned_surplus_models,
)


def _assert_binding_contract(matrix: torch.Tensor, atol: float = 1e-6):
    assert torch.isfinite(matrix).all()
    assert torch.all(matrix >= -atol)
    assert torch.allclose(matrix.sum(dim=1), torch.ones(4), atol=atol, rtol=0)
    occupancy = matrix.sum(dim=0)
    assert torch.all(occupancy >= -atol)
    assert torch.all(occupancy <= 1.0 + atol)
    assert abs(float(matrix.sum()) - 4.0) <= atol * 8


def _assert_categorical(probs: torch.Tensor, atol: float = 1e-6):
    assert torch.isfinite(probs).all()
    assert torch.all(probs >= -atol)
    assert torch.allclose(probs.sum(dim=-1), torch.ones_like(probs[..., 0]), atol=atol, rtol=0)


def test_regimes_start_from_identical_parameters():
    torch.manual_seed(61)
    seed = SurplusSlotTransitionModel(d_model=32, binding_mode="learned_sparse")
    models = cloned_surplus_models(seed)
    reference = models["canonical_sparse"].state_dict()
    for mode in ("learned_sparse", "diffuse_surplus"):
        candidate = models[mode].state_dict()
        assert reference.keys() == candidate.keys()
        for key in reference:
            assert torch.equal(reference[key], candidate[key])


def test_learned_binding_starts_near_uniform_over_eight_candidates():
    torch.manual_seed(62)
    model = SurplusSlotTransitionModel(d_model=16, binding_mode="learned_sparse")
    matrix = model.soft_binding()
    _assert_binding_contract(matrix)
    assert float(matrix.max(dim=1).values.mean()) < 0.15
    occupancy = matrix.sum(dim=0)
    assert torch.all(torch.abs(occupancy - 0.5) < 0.05), occupancy


def test_injective_mixture_contract_survives_adversarial_scores():
    model = SurplusSlotTransitionModel(d_model=16, binding_mode="learned_sparse")
    cases = [
        torch.randn(4, 8) * 50.0,
        torch.tensor(
            [
                [80.0, 80.0, -80.0, -80.0, -80.0, -80.0, -80.0, -80.0],
                [80.0, 80.0, -80.0, -80.0, -80.0, -80.0, -80.0, -80.0],
                [80.0, 80.0, -80.0, -80.0, -80.0, -80.0, -80.0, -80.0],
                [80.0, 80.0, -80.0, -80.0, -80.0, -80.0, -80.0, -80.0],
            ]
        ),
    ]
    for logits in cases:
        with torch.no_grad():
            model.binding_logits.copy_(logits)
        _assert_binding_contract(model.soft_binding())


def test_projected_assignment_uses_four_distinct_slots():
    model = SurplusSlotTransitionModel(d_model=16, binding_mode="learned_sparse")
    with torch.no_grad():
        model.binding_logits.fill_(-5.0)
        model.binding_logits[0, 7] = 8.0
        model.binding_logits[1, 2] = 8.0
        model.binding_logits[2, 5] = 8.0
        model.binding_logits[3, 0] = 8.0
    projected, assignment, score = model.best_injective_assignment()
    assert assignment == [7, 2, 5, 0]
    assert len(set(assignment)) == 4
    assert torch.equal(projected.sum(dim=1), torch.ones(4))
    assert torch.all(projected.sum(dim=0) <= 1)
    assert score > 3.9


def test_canonical_and_diffuse_contracts():
    canonical = SurplusSlotTransitionModel(d_model=16, binding_mode="canonical_sparse")
    diffuse = SurplusSlotTransitionModel(d_model=16, binding_mode="diffuse_surplus")
    c = canonical.soft_binding()
    d = diffuse.soft_binding()
    _assert_binding_contract(c)
    _assert_binding_contract(d)
    assert torch.equal(c[:, :4], torch.eye(4))
    assert torch.equal(c[:, 4:], torch.zeros(4, 4))
    assert torch.equal(d, torch.full((4, 8), 0.125))


def test_empty_mass_makes_every_internal_slot_categorical():
    torch.manual_seed(63)
    batch = make_contextual_batch(16, 8, 9701, split="train")
    for mode in BINDING_MODES:
        model = SurplusSlotTransitionModel(d_model=16, binding_mode=mode)
        binding = model.soft_binding()
        internal = model.initial_internal_probs(batch.initial, binding)
        assert internal.shape == (16, NUM_CANDIDATE_SLOTS, 17)
        _assert_categorical(internal)
        decoded = model.decode_external_probs(internal, binding)
        _assert_categorical(decoded)

    canonical = SurplusSlotTransitionModel(d_model=16, binding_mode="canonical_sparse")
    internal = canonical.initial_internal_probs(batch.initial, canonical.soft_binding())
    assert torch.allclose(internal[:, 4:, EMPTY_VALUE], torch.ones_like(internal[:, 4:, EMPTY_VALUE]))


def test_fixed_answer_loss_ignores_intermediate_hidden_and_semantic_targets():
    torch.manual_seed(64)
    batch = make_contextual_batch(16, 8, 9702, split="train")
    changed = deepcopy(batch)
    changed.target_states = batch.target_states.clone()
    changed.target_states[:, :-1] = torch.remainder(changed.target_states[:, :-1] + 1, 16)
    changed.target_states[:, -1, 1:] = torch.remainder(changed.target_states[:, -1, 1:] + 1, 16)
    changed.semantics = torch.randint_like(batch.semantics, 0, 8)
    model = SurplusSlotTransitionModel(d_model=32, binding_mode="learned_sparse")
    assert torch.equal(model.fixed_answer_loss(batch), model.fixed_answer_loss(changed))


def test_fixed_answer_loss_changes_when_answer_target_changes():
    torch.manual_seed(65)
    batch = make_contextual_batch(16, 8, 9703, split="train")
    changed = deepcopy(batch)
    changed.target_states = batch.target_states.clone()
    changed.target_states[:, -1, 0] = torch.remainder(changed.target_states[:, -1, 0] + 1, 16)
    model = SurplusSlotTransitionModel(d_model=32, binding_mode="learned_sparse")
    assert not torch.equal(model.fixed_answer_loss(batch), model.fixed_answer_loss(changed))


def test_answer_loss_reaches_assignment_and_transition_parameters():
    batch = make_contextual_batch(32, 8, 9704, split="train")
    model = SurplusSlotTransitionModel(d_model=32, binding_mode="learned_sparse")
    loss = model.fixed_answer_loss(batch)
    assert torch.isfinite(loss) and float(loss.detach()) >= 0.0
    loss.backward()
    assert model.binding_logits.grad is not None
    assert torch.isfinite(model.binding_logits.grad).all()
    assert model.binding_logits.grad.abs().sum() > 0
    grads = [p.grad for p in model.transition.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_probability_contract_survives_optimizer_updates():
    torch.manual_seed(66)
    batch = make_contextual_batch(32, 8, 9705, split="train")
    model = SurplusSlotTransitionModel(d_model=32, binding_mode="learned_sparse")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = model.fixed_answer_loss(batch)
        assert torch.isfinite(loss) and float(loss.detach()) >= 0.0
        loss.backward()
        optimizer.step()

        binding = model.soft_binding()
        _assert_binding_contract(binding)
        internal = model.initial_internal_probs(batch.initial, binding)
        _assert_categorical(internal)
        decoded = model.rollout_soft(batch)
        _assert_categorical(decoded)
        post_loss = model.fixed_answer_loss(batch)
        assert torch.isfinite(post_loss) and float(post_loss.detach()) >= 0.0


def test_external_register_identity_has_no_direct_register_embedding_bypass():
    model = SurplusSlotTransitionModel(d_model=16, binding_mode="learned_sparse")
    assert hasattr(model, "slot")
    assert not hasattr(model, "register")


def test_all_regimes_extend_to_depth_96():
    batch = make_contextual_batch(2, 96, 9706, split="composition")
    seed = SurplusSlotTransitionModel(d_model=16, binding_mode="learned_sparse")
    models = cloned_surplus_models(seed)
    assert set(models) == set(BINDING_MODES)
    for model in models.values():
        pred = model.rollout_hard(batch, discrete_binding=True)
        assert pred.shape == batch.target_states.shape
