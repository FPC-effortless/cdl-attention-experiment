from __future__ import annotations

import torch

from casm.contextual_data import make_contextual_batch
from casm.explicit_compute import SharedTransitionModel
from casm.state_access_controls import FeatureMatchedStateGRUControl


def test_feature_matched_gru_parameter_match():
    explicit = SharedTransitionModel(d_model=96)
    control = FeatureMatchedStateGRUControl(d_model=88)
    ratio = control.parameter_count() / explicit.parameter_count()
    assert 0.95 <= ratio <= 1.05


def test_feature_matched_gru_loss_and_depth_contract():
    train = make_contextual_batch(8, 4, 1501, split="train")
    model = FeatureMatchedStateGRUControl(d_model=32)
    loss = model.training_loss(train)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

    stress = make_contextual_batch(2, 96, 1502, split="composition")
    small = FeatureMatchedStateGRUControl(d_model=16)
    assert small.rollout(stress).shape == stress.target_states.shape
