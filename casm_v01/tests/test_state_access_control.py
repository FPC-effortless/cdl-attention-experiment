from __future__ import annotations

import torch

from casm.contextual_baselines import MatchedGRUProgramBaseline
from casm.contextual_data import make_contextual_batch
from casm.explicit_compute import SharedTransitionModel
from casm.state_access_controls import StateAccessGRUControl


def test_state_access_gru_is_parameter_matched():
    explicit = SharedTransitionModel(d_model=96)
    control = StateAccessGRUControl(d_model=112)
    ratio = control.parameter_count() / explicit.parameter_count()
    assert 0.95 <= ratio <= 1.05


def test_hidden_only_gru_is_parameter_matched():
    explicit = SharedTransitionModel(d_model=96)
    control = MatchedGRUProgramBaseline(d_model=114)
    ratio = control.parameter_count() / explicit.parameter_count()
    assert 0.95 <= ratio <= 1.05


def test_state_access_training_loss_is_finite_and_backpropagates():
    batch = make_contextual_batch(8, 4, 901, split="train")
    model = StateAccessGRUControl(d_model=32)
    loss = model.training_loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_state_access_rollout_shape_matches_contract_at_depth_96():
    batch = make_contextual_batch(2, 96, 902, split="composition")
    model = StateAccessGRUControl(d_model=16)
    pred = model.rollout(batch)
    assert pred.shape == batch.target_states.shape
