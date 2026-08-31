from __future__ import annotations

import torch

from casm.contextual_baselines import MatchedGRUProgramBaseline, MatchedTransformerProgramBaseline
from casm.contextual_data import (
    CONTEXT_ALIASES,
    HELDOUT_FAMILY_BIGRAMS,
    contextual_semantic,
    make_contextual_batch,
)
from casm.explicit_compute import SharedTransitionModel, _apply_python


def _families(commands):
    inverse = {alias: family for family, alias in enumerate(CONTEXT_ALIASES)}
    return [[inverse[value] for value in row] for row in commands.tolist()]


def test_same_command_has_state_dependent_semantics():
    even_state = [0, 0, 0, 0]
    odd_state = [1, 0, 0, 0]
    assert contextual_semantic(even_state, 0, 0, 1, 2) == 0
    assert contextual_semantic(odd_state, 0, 0, 1, 2) == 1
    assert CONTEXT_ALIASES[0] == CONTEXT_ALIASES[0]


def test_train_split_excludes_heldout_family_bigrams():
    batch = make_contextual_batch(128, 4, 42, split="train")
    for row in _families(batch.commands):
        assert not any((a, b) in HELDOUT_FAMILY_BIGRAMS for a, b in zip(row, row[1:]))


def test_composition_split_requires_heldout_family_bigram():
    batch = make_contextual_batch(128, 12, 43, split="composition")
    for row in _families(batch.commands):
        assert any((a, b) in HELDOUT_FAMILY_BIGRAMS for a, b in zip(row, row[1:]))


def test_targets_follow_contextual_semantics_exactly():
    batch = make_contextual_batch(16, 8, 99, split="composition")
    for i in range(batch.initial.shape[0]):
        state = batch.initial[i].tolist()
        families = _families(batch.commands[i : i + 1])[0]
        for t, family in enumerate(families):
            a = int(batch.arg_a[i, t])
            b = int(batch.arg_b[i, t])
            dst = int(batch.dst[i, t])
            semantic = contextual_semantic(state, family, a, b, dst)
            assert semantic == int(batch.semantics[i, t])
            state = _apply_python(state, semantic, a, b, dst)
            assert state == batch.target_states[i, t].tolist()


def test_generic_controls_are_parameter_matched_to_explicit_state():
    explicit = SharedTransitionModel(d_model=96)
    gru = MatchedGRUProgramBaseline(d_model=114)
    transformer = MatchedTransformerProgramBaseline(d_model=92)
    target = explicit.parameter_count()
    assert abs(gru.parameter_count() / target - 1.0) <= 0.05
    assert abs(transformer.parameter_count() / target - 1.0) <= 0.05


def test_all_losses_are_finite_and_backpropagate():
    batch = make_contextual_batch(8, 4, 77, split="train")
    models = [
        SharedTransitionModel(d_model=32),
        MatchedGRUProgramBaseline(d_model=32),
        MatchedTransformerProgramBaseline(d_model=32, nhead=4, max_depth=16),
    ]
    for model in models:
        model.zero_grad(set_to_none=True)
        out = model.training_loss(batch)
        loss = out["loss"] if isinstance(out, dict) else out
        assert torch.isfinite(loss)
        loss.backward()
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_rollout_shape_contract_extends_to_depth_96():
    batch = make_contextual_batch(2, 96, 101, split="composition")
    explicit = SharedTransitionModel(d_model=16)
    gru = MatchedGRUProgramBaseline(d_model=16)
    transformer = MatchedTransformerProgramBaseline(d_model=16, nhead=4, max_depth=128)
    assert explicit.rollout(batch).shape == batch.target_states.shape
    assert gru.rollout(batch).shape == batch.target_states.shape
    assert transformer.rollout(batch).shape == batch.target_states.shape
