from __future__ import annotations

from copy import deepcopy

import torch

from casm.contextual_data import make_contextual_batch
from casm.weak_supervision import SoftExplicitTransitionModel, cloned_regime_models


def test_supervision_indices_match_preregistered_contract():
    assert SoftExplicitTransitionModel.supervision_indices(8, "process") == list(range(8))
    assert SoftExplicitTransitionModel.supervision_indices(8, "quarter") == [3, 7]
    assert SoftExplicitTransitionModel.supervision_indices(8, "final") == [7]


def test_regimes_start_from_identical_parameters():
    torch.manual_seed(123)
    seed = SoftExplicitTransitionModel(d_model=32)
    models = cloned_regime_models(seed)
    reference = models["process"].state_dict()
    for regime in ("quarter", "final"):
        candidate = models[regime].state_dict()
        assert reference.keys() == candidate.keys()
        for key in reference:
            assert torch.equal(reference[key], candidate[key])


def test_rollout_does_not_consume_target_states():
    batch = make_contextual_batch(4, 8, 9001, split="train")
    corrupted = deepcopy(batch)
    corrupted.target_states = torch.randint_like(batch.target_states, 0, 16)
    model = SoftExplicitTransitionModel(d_model=32)
    original = model.rollout_soft(batch)
    changed = model.rollout_soft(corrupted)
    assert torch.equal(original, changed)


def test_final_only_loss_ignores_all_intermediate_targets():
    batch = make_contextual_batch(8, 8, 9002, split="train")
    corrupted = deepcopy(batch)
    corrupted.target_states = batch.target_states.clone()
    corrupted.target_states[:, :-1] = torch.randint_like(corrupted.target_states[:, :-1], 0, 16)
    model = SoftExplicitTransitionModel(d_model=32)
    loss_a = model.training_loss(batch, "final")["loss"]
    loss_b = model.training_loss(corrupted, "final")["loss"]
    assert torch.equal(loss_a, loss_b)


def test_quarter_loss_ignores_unsupervised_targets():
    batch = make_contextual_batch(8, 8, 9003, split="train")
    corrupted = deepcopy(batch)
    corrupted.target_states = batch.target_states.clone()
    for index in (0, 1, 2, 4, 5, 6):
        corrupted.target_states[:, index] = torch.randint_like(corrupted.target_states[:, index], 0, 16)
    model = SoftExplicitTransitionModel(d_model=32)
    loss_a = model.training_loss(batch, "quarter")["loss"]
    loss_b = model.training_loss(corrupted, "quarter")["loss"]
    assert torch.equal(loss_a, loss_b)


def test_final_only_gradient_is_finite():
    batch = make_contextual_batch(8, 8, 9004, split="train")
    model = SoftExplicitTransitionModel(d_model=32)
    loss = model.training_loss(batch, "final")["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_hard_rollout_extends_to_depth_96():
    batch = make_contextual_batch(2, 96, 9005, split="composition")
    model = SoftExplicitTransitionModel(d_model=16)
    pred = model.rollout_hard(batch)
    assert pred.shape == batch.target_states.shape
