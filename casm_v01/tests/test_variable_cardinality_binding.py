from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn

from casm.variable_cardinality_binding import (
    BINDING_MODES,
    DESCRIPTOR_DIM,
    NUM_CANDIDATE_SLOTS,
    VariableCardinalityTransitionModel,
    cloned_cardinality_models,
    variable_descriptor,
)
from casm.variable_contextual_data import (
    MAX_CARDINALITY,
    MIN_CARDINALITY,
    TRAIN_CARDINALITIES,
    make_variable_contextual_batch,
    training_cardinality_for_step,
)


def _assert_categorical(x: torch.Tensor, atol: float = 1e-6):
    assert torch.isfinite(x).all()
    assert torch.all(x >= -atol)
    assert torch.allclose(x.sum(dim=-1), torch.ones_like(x[..., 0]), atol=atol, rtol=0)


def test_descriptor_is_deterministic_bounded_and_valid_for_unseen_cardinalities():
    for n in range(MIN_CARDINALITY, MAX_CARDINALITY + 1):
        ids = torch.arange(n)
        a = variable_descriptor(ids, n)
        b = variable_descriptor(ids, n)
        assert a.shape == (n, DESCRIPTOR_DIM)
        assert torch.equal(a, b)
        assert torch.isfinite(a).all()
        assert float(a.abs().max()) <= 1.0 + 1e-6
        assert float(a[:, 0].min()) >= 0.0
        assert float(a[:, 0].max()) <= 1.0
        assert torch.allclose(a[:, 1], torch.full((n,), n / 6.0))


def test_descriptor_depends_only_on_external_index_and_cardinality():
    ids = torch.arange(6)
    reference = variable_descriptor(ids, 6)
    # No batch, target, command, state or RNG argument exists in the descriptor API.
    assert torch.equal(reference, variable_descriptor(ids, 6))


def test_shared_model_has_no_external_id_embedding_or_free_binding_row_table():
    model = VariableCardinalityTransitionModel(d_model=32, binding_mode="shared_generator_dense")
    assert not hasattr(model, "register")
    assert not hasattr(model, "external_register")
    assert not hasattr(model, "binding_logits")
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            assert module.num_embeddings not in {2, 3, 4, 5, 6}, (name, module.num_embeddings)
    for name, parameter in model.named_parameters():
        assert tuple(parameter.shape) not in {(6, 8), (5, 8), (4, 8), (3, 8), (2, 8)}, (
            name,
            parameter.shape,
        )


def test_cloned_regimes_are_parameter_matched_and_identically_initialized():
    torch.manual_seed(71)
    seed = VariableCardinalityTransitionModel(d_model=32)
    models = cloned_cardinality_models(seed)
    assert set(models) == set(BINDING_MODES)
    assert len({m.parameter_count() for m in models.values()}) == 1
    assert len({m.trainable_parameter_count() for m in models.values()}) == 1
    reference = models["canonical_functional"].state_dict()
    for mode in ("shared_generator_dense", "diffuse_dense"):
        candidate = models[mode].state_dict()
        assert reference.keys() == candidate.keys()
        for key in reference:
            assert torch.equal(reference[key], candidate[key])


def test_generated_binding_is_near_uniform_row_stochastic_for_all_cardinalities():
    torch.manual_seed(72)
    model = VariableCardinalityTransitionModel(d_model=32, binding_mode="shared_generator_dense")
    for n in range(2, 7):
        binding = model.soft_binding(n)
        assert binding.shape == (n, NUM_CANDIDATE_SLOTS)
        assert torch.isfinite(binding).all()
        assert torch.all(binding >= 0)
        assert torch.allclose(binding.sum(dim=1), torch.ones(n), atol=1e-6, rtol=0)
        assert abs(float(binding.sum().detach()) - n) <= 1e-5
        assert float(binding.max(dim=1).values.mean().detach()) < 0.18


def test_capacity_normalized_transport_handles_adversarial_all_row_collision():
    torch.manual_seed(73)
    model = VariableCardinalityTransitionModel(d_model=32, binding_mode="shared_generator_dense")
    for n in (2, 4, 6):
        batch = make_variable_contextual_batch(8, 8, 10000 + n, num_registers=n, split="train")
        binding = torch.zeros(n, 8)
        binding[:, 3] = 1.0
        internal = model.initial_internal_probs(batch.initial, binding)
        _assert_categorical(internal)
        decoded = model.decode_external_probs(internal, binding)
        _assert_categorical(decoded)


def test_hard_row_argmax_preserves_collisions_without_matching_repair():
    model = VariableCardinalityTransitionModel(d_model=16, binding_mode="shared_generator_dense")
    last = model.binding_generator[-1]
    assert isinstance(last, nn.Linear)
    with torch.no_grad():
        last.weight.zero_()
        last.bias.fill_(-9.0)
        last.bias[6] = 9.0
    _, assignment = model.independent_argmax_binding(6)
    assert assignment == [6, 6, 6, 6, 6, 6]
    assert len(set(assignment)) == 1


def test_fixed_answer_loss_ignores_intermediate_hidden_and_semantic_targets_variable_width():
    torch.manual_seed(74)
    batch = make_variable_contextual_batch(16, 8, 10074, num_registers=6, split="train")
    changed = deepcopy(batch)
    changed.target_states = batch.target_states.clone()
    changed.target_states[:, :-1] = torch.remainder(changed.target_states[:, :-1] + 1, 16)
    changed.target_states[:, -1, 1:] = torch.remainder(changed.target_states[:, -1, 1:] + 1, 16)
    changed.semantics = torch.randint_like(batch.semantics, 0, 8)
    model = VariableCardinalityTransitionModel(d_model=32, binding_mode="shared_generator_dense")
    assert torch.equal(model.fixed_answer_loss(batch), model.fixed_answer_loss(changed))


def test_fixed_answer_loss_changes_when_register_zero_target_changes():
    torch.manual_seed(75)
    batch = make_variable_contextual_batch(16, 8, 10075, num_registers=5, split="train")
    changed = deepcopy(batch)
    changed.target_states = batch.target_states.clone()
    changed.target_states[:, -1, 0] = torch.remainder(changed.target_states[:, -1, 0] + 1, 16)
    model = VariableCardinalityTransitionModel(d_model=32, binding_mode="shared_generator_dense")
    assert not torch.equal(model.fixed_answer_loss(batch), model.fixed_answer_loss(changed))


def test_answer_loss_reaches_shared_generator_and_transition():
    torch.manual_seed(76)
    batch = make_variable_contextual_batch(32, 8, 10076, num_registers=4, split="train")
    model = VariableCardinalityTransitionModel(d_model=32, binding_mode="shared_generator_dense")
    loss = model.fixed_answer_loss(batch)
    assert torch.isfinite(loss) and float(loss.detach()) >= 0.0
    loss.backward()
    generator_grads = [p.grad for p in model.binding_generator.parameters() if p.grad is not None]
    transition_grads = [p.grad for p in model.transition.parameters() if p.grad is not None]
    assert generator_grads and any(g.abs().sum() > 0 for g in generator_grads)
    assert transition_grads and any(g.abs().sum() > 0 for g in transition_grads)
    assert all(torch.isfinite(g).all() for g in generator_grads + transition_grads)


def test_probability_contract_survives_optimizer_updates_across_training_cardinalities():
    torch.manual_seed(77)
    model = VariableCardinalityTransitionModel(d_model=32, binding_mode="shared_generator_dense")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    for step, n in enumerate(TRAIN_CARDINALITIES, start=1):
        batch = make_variable_contextual_batch(24, 8, 10100 + step, num_registers=n, split="train")
        optimizer.zero_grad(set_to_none=True)
        loss = model.fixed_answer_loss(batch)
        assert torch.isfinite(loss) and float(loss.detach()) >= 0.0
        loss.backward()
        optimizer.step()
        binding = model.soft_binding(n)
        assert torch.allclose(binding.sum(dim=1), torch.ones(n), atol=1e-6, rtol=0)
        internal = model.initial_internal_probs(batch.initial, binding)
        _assert_categorical(internal)
        _assert_categorical(model.rollout_soft(batch))


def test_canonical_functional_binding_is_valid_for_every_cardinality():
    model = VariableCardinalityTransitionModel(d_model=16, binding_mode="canonical_functional")
    for n in range(2, 7):
        binding = model.soft_binding(n)
        assert torch.equal(binding[:, :n], torch.eye(n))
        assert torch.equal(binding[:, n:], torch.zeros(n, 8 - n))


def test_training_cardinality_schedule_is_exact_repeating_234():
    observed = [training_cardinality_for_step(step) for step in range(1, 13)]
    assert observed == [2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4]


def test_variable_batches_and_rollouts_extend_to_unseen_n_and_depth_96():
    torch.manual_seed(78)
    seed = VariableCardinalityTransitionModel(d_model=16)
    models = cloned_cardinality_models(seed)
    for n in range(2, 7):
        batch = make_variable_contextual_batch(2, 96, 10200 + n, num_registers=n, split="composition")
        assert batch.initial.shape == (2, n)
        assert batch.target_states.shape == (2, 96, n)
        assert batch.arg_a.max().item() < n
        assert batch.arg_b.max().item() < n
        assert batch.dst.max().item() < n
        for model in models.values():
            pred = model.rollout_hard(batch, discrete_binding=True)
            assert pred.shape == batch.target_states.shape
