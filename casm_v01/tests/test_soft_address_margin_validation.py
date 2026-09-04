from __future__ import annotations

import torch

from casm.noncontractive_role_dynamics import (
    ADDRESS_BETA,
    address_probabilities,
    cloned_x19d_models,
)
from casm.soft_address_margin_validation import (
    COUNTERFACTUAL_BETA,
    TRAIN_BETA,
    X19VAddressView,
    address_probabilities_at_beta,
)


def test_training_beta_is_exact_x19d_beta():
    assert TRAIN_BETA == ADDRESS_BETA == 16.0
    assert COUNTERFACTUAL_BETA == 64.0


def test_beta16_view_is_numerically_identical_to_x19d_addressing():
    torch.manual_seed(1234)
    model = cloned_x19d_models()["orthogonal_recursive"]
    roles = model.roles(4)
    expected = address_probabilities(roles, roles)
    actual = address_probabilities_at_beta(roles, roles, beta=TRAIN_BETA)
    assert torch.equal(expected, actual)
    view = X19VAddressView(model, beta=TRAIN_BETA)
    assert torch.equal(view.address_matrix(4, discrete=False), model.address_matrix(4, discrete=False))


def test_views_share_exact_same_model_roles_and_parameters():
    torch.manual_seed(2345)
    model = cloned_x19d_models()["orthogonal_recursive"]
    v16 = X19VAddressView(model, beta=TRAIN_BETA)
    v64 = X19VAddressView(model, beta=COUNTERFACTUAL_BETA)
    assert v16.model is model and v64.model is model
    assert torch.equal(v16.roles(6), v64.roles(6))
    assert list(v16.model.parameters()) == list(v64.model.parameters())


def test_beta_change_does_not_change_hard_argmax():
    torch.manual_seed(3456)
    for mode, model in cloned_x19d_models().items():
        v16 = X19VAddressView(model, beta=TRAIN_BETA)
        v64 = X19VAddressView(model, beta=COUNTERFACTUAL_BETA)
        for n in (2, 3, 4, 5, 6):
            assert torch.equal(
                v16.address_matrix(n, discrete=True),
                v64.address_matrix(n, discrete=True),
            ), mode


def test_beta64_sharpens_distinct_role_self_addressing_without_repair():
    torch.manual_seed(4567)
    model = cloned_x19d_models()["frozen_random_orthogonal"]
    s16 = X19VAddressView(model, beta=TRAIN_BETA).address_stats(6)
    s64 = X19VAddressView(model, beta=COUNTERFACTUAL_BETA).address_stats(6)
    assert s16["hard_address"] == s64["hard_address"]
    assert s64["mean_soft_self_address_probability"] >= s16["mean_soft_self_address_probability"]
    assert s64["maximum_competing_address_probability"] <= s16["maximum_competing_address_probability"]


def test_identical_keys_remain_hard_ambiguous_at_both_betas():
    role = torch.zeros(32)
    role[0] = 1.0
    roles = torch.stack([role, role, torch.roll(role, 1)], dim=0)
    for beta in (TRAIN_BETA, COUNTERFACTUAL_BETA):
        p = address_probabilities_at_beta(roles, roles, beta=beta)
        assert torch.equal(p[:2, :2], p[:2, :2].T)
        # No matching/repair: argmax chooses the same first duplicate for both duplicate queries.
        hard = p.argmax(dim=-1)
        assert hard[0].item() == 0 and hard[1].item() == 0


def test_beta_view_has_no_trainable_state_of_its_own():
    torch.manual_seed(5678)
    model = cloned_x19d_models()["orthogonal_recursive"]
    view = X19VAddressView(model, beta=COUNTERFACTUAL_BETA)
    assert not hasattr(view, "parameters")
