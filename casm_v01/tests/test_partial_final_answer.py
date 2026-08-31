from __future__ import annotations

from copy import deepcopy

import torch

from casm.contextual_data import make_contextual_batch
from casm.partial_final_answer import (
    cloned_partial_models,
    one_parity_bit_loss,
    one_register_loss,
    sample_query_registers,
)
from casm.weak_supervision import SoftExplicitTransitionModel


def test_partial_regimes_start_from_identical_parameters():
    torch.manual_seed(17)
    seed = SoftExplicitTransitionModel(d_model=32)
    models = cloned_partial_models(seed)
    reference = models["full_final"].state_dict()
    for regime in ("one_register", "one_parity_bit"):
        for key, value in models[regime].state_dict().items():
            assert torch.equal(reference[key], value)


def test_query_registers_are_approximately_uniform():
    query = sample_query_registers(8192, 123456)
    counts = torch.bincount(query, minlength=4).float() / query.numel()
    assert torch.all(torch.abs(counts - 0.25) < 0.025), counts


def test_one_register_loss_ignores_unqueried_and_intermediate_targets():
    batch = make_contextual_batch(16, 8, 9101, split="train")
    query = sample_query_registers(16, 9201)
    corrupted = deepcopy(batch)
    corrupted.target_states = batch.target_states.clone()
    corrupted.target_states[:, :-1] = torch.remainder(corrupted.target_states[:, :-1] + 1, 16)
    rows = torch.arange(16)
    for register in range(4):
        mask = query.ne(register)
        corrupted.target_states[rows[mask], -1, register] = torch.remainder(
            corrupted.target_states[rows[mask], -1, register] + 1, 16
        )
    model = SoftExplicitTransitionModel(d_model=32)
    a = one_register_loss(model, batch, query)
    b = one_register_loss(model, corrupted, query)
    assert torch.equal(a, b)


def test_one_parity_loss_ignores_intermediate_targets_and_same_parity_value_changes():
    batch = make_contextual_batch(16, 8, 9102, split="train")
    query = sample_query_registers(16, 9202)
    corrupted = deepcopy(batch)
    corrupted.target_states = batch.target_states.clone()
    corrupted.target_states[:, :-1] = torch.remainder(corrupted.target_states[:, :-1] + 1, 16)
    rows = torch.arange(16)
    corrupted.target_states[rows, -1, query] = torch.remainder(
        corrupted.target_states[rows, -1, query] + 2, 16
    )
    model = SoftExplicitTransitionModel(d_model=32)
    a = one_parity_bit_loss(model, batch, query)
    b = one_parity_bit_loss(model, corrupted, query)
    assert torch.equal(a, b)


def test_one_parity_loss_changes_when_target_parity_flips():
    torch.manual_seed(19)
    batch = make_contextual_batch(32, 8, 9103, split="train")
    query = sample_query_registers(32, 9203)
    flipped = deepcopy(batch)
    flipped.target_states = batch.target_states.clone()
    rows = torch.arange(32)
    flipped.target_states[rows, -1, query] = torch.remainder(
        flipped.target_states[rows, -1, query] + 1, 16
    )
    model = SoftExplicitTransitionModel(d_model=32)
    a = one_parity_bit_loss(model, batch, query)
    b = one_parity_bit_loss(model, flipped, query)
    assert not torch.equal(a, b)


def test_weak_losses_ignore_semantic_operator_labels():
    batch = make_contextual_batch(16, 8, 9104, split="train")
    query = sample_query_registers(16, 9204)
    corrupted = deepcopy(batch)
    corrupted.semantics = torch.randint_like(batch.semantics, 0, 8)
    model = SoftExplicitTransitionModel(d_model=32)
    assert torch.equal(one_register_loss(model, batch, query), one_register_loss(model, corrupted, query))
    assert torch.equal(one_parity_bit_loss(model, batch, query), one_parity_bit_loss(model, corrupted, query))


def test_partial_terminal_losses_have_finite_transition_gradients():
    batch = make_contextual_batch(16, 8, 9105, split="train")
    query = sample_query_registers(16, 9205)
    for loss_fn in (one_register_loss, one_parity_bit_loss):
        model = SoftExplicitTransitionModel(d_model=32)
        loss = loss_fn(model, batch, query)
        assert torch.isfinite(loss)
        loss.backward()
        grads = [p.grad for p in model.transition.parameters() if p.grad is not None]
        assert grads
        assert all(torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum() > 0 for g in grads)
