from __future__ import annotations

import random

import torch

from casm.explicit_compute import (
    ExplicitOperatorMachine,
    GRUProgramBaseline,
    HELDOUT_BIGRAMS,
    NUM_OPERATORS,
    SharedTransitionModel,
    _apply_python,
    make_program_batch,
    oracle_transition,
)


def test_oracle_transition_matches_python_for_every_operator():
    rng = random.Random(123)
    for op in range(NUM_OPERATORS):
        for _ in range(20):
            state = [rng.randrange(16) for _ in range(4)]
            a, b, dst = [rng.randrange(4) for _ in range(3)]
            expected = _apply_python(state, op, a, b, dst)
            actual = oracle_transition(
                torch.tensor([state]),
                torch.tensor([op]),
                torch.tensor([a]),
                torch.tensor([b]),
                torch.tensor([dst]),
            )[0].tolist()
            assert actual == expected


def test_train_split_excludes_heldout_bigrams_and_composition_contains_them():
    train = make_program_batch(128, 3, 42, split="train")
    for row in train.semantics.tolist():
        assert not any((x, y) in HELDOUT_BIGRAMS for x, y in zip(row, row[1:]))

    composition = make_program_batch(128, 6, 43, split="composition")
    for row in composition.semantics.tolist():
        assert any((x, y) in HELDOUT_BIGRAMS for x, y in zip(row, row[1:]))


def test_oracle_both_reconstructs_all_target_states_exactly():
    batch = make_program_batch(32, 12, 99, split="composition")
    model = ExplicitOperatorMachine(d_model=32)
    states, routes = model.rollout(
        batch,
        use_verifier=False,
        oracle_routing=True,
        oracle_execution=True,
    )
    assert torch.equal(states, batch.target_states)
    assert torch.equal(routes, batch.semantics)


def test_all_training_losses_are_finite_and_backpropagate():
    batch = make_program_batch(16, 3, 77, split="train")
    models = [
        ExplicitOperatorMachine(d_model=32),
        SharedTransitionModel(d_model=32),
        GRUProgramBaseline(d_model=32),
    ]
    for model in models:
        model.zero_grad(set_to_none=True)
        out = model.training_loss(batch)
        loss = out["loss"] if isinstance(out, dict) else out
        assert torch.isfinite(loss)
        loss.backward()
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_rollout_shapes_match_target_state_contract():
    batch = make_program_batch(8, 4, 100, split="composition")
    explicit = ExplicitOperatorMachine(d_model=32)
    shared = SharedTransitionModel(d_model=32)
    gru = GRUProgramBaseline(d_model=32)

    e_states, e_routes = explicit.rollout(batch, use_verifier=True)
    assert e_states.shape == batch.target_states.shape
    assert e_routes.shape == batch.semantics.shape
    assert shared.rollout(batch).shape == batch.target_states.shape
    assert gru.rollout(batch).shape == batch.target_states.shape
