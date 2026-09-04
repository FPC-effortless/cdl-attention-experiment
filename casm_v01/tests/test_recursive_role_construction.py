from __future__ import annotations

from dataclasses import replace

import torch

import casm.recursive_role_construction as x19
from casm.coordinated_binding import slot_descriptor
from casm.explicit_compute import VALUE_MODULUS
from casm.global_coordinate_binding import global_variable_descriptor
from casm.recursive_role_construction import (
    LEARNED_X19_MODES,
    ROLE_CONTEXT_DIM,
    ROLE_DIM,
    ROLE_DIAGNOSTIC_COUNT,
    RoleToSlotScorer,
    X19RoleModel,
    cloned_x19_models,
    global_role_context,
)
from casm.variable_cardinality_binding import NUM_CANDIDATE_SLOTS
from casm.variable_contextual_data import make_variable_contextual_batch


def test_global_role_context_matches_x18_on_supported_indices():
    indices = torch.arange(6)
    expected = global_variable_descriptor(indices, 6)
    actual = global_role_context(indices)
    assert torch.equal(actual, expected)
    extended = global_role_context(torch.arange(ROLE_DIAGNOSTIC_COUNT))
    assert extended.shape == (8, 9)
    assert torch.isfinite(extended).all()


def test_learned_regimes_parameter_matched_and_bit_identical_shared_state():
    torch.manual_seed(19001)
    models = cloned_x19_models(d_model=32)
    static = models["static_global_roles"]
    recursive = models["recursive_roles"]
    assert static.parameter_count() == recursive.parameter_count()
    assert static.trainable_parameter_count() == recursive.trainable_parameter_count()
    ss, rs = static.state_dict(), recursive.state_dict()
    assert ss.keys() == rs.keys()
    assert all(torch.equal(ss[k], rs[k]) for k in ss)
    assert static.role_seed is not None and static.role_seed.shape == (ROLE_DIM,)
    assert recursive.role_seed is not None and recursive.role_seed.shape == (ROLE_DIM,)


def test_no_learned_id_tables_or_resource_allocator_modules():
    for mode in LEARNED_X19_MODES:
        model = X19RoleModel(mode=mode, d_model=32)
        assert model.role_cell is not None and model.storage_bridge is not None
        # The validated executor legitimately contains value/command embeddings. The X19
        # restriction applies to role construction and storage identity, not executor values.
        for root in (model.role_cell, model.storage_bridge):
            for name, module in root.named_modules():
                assert not isinstance(module, torch.nn.Embedding), name
        forbidden = ("external_id", "cardinality_id", "role_index", "step_index", "dual", "price", "occupancy", "sinkhorn", "hungarian")
        for name, _ in model.named_parameters():
            lower = name.lower()
            assert not any(token in lower for token in forbidden), name


def test_recursive_roles_ignore_global_context_and_have_prefix_consistency(monkeypatch):
    torch.manual_seed(19002)
    model = X19RoleModel(mode="recursive_roles", d_model=32)
    before = model.roles(4).detach().clone()

    def explode(*args, **kwargs):
        raise AssertionError("recursive path must not request global/index context")

    monkeypatch.setattr(x19, "global_role_context", explode)
    after = model.roles(4).detach().clone()
    assert torch.equal(before, after)

    seed = model.normalized_seed()
    assert model.role_cell is not None
    step = seed.new_zeros(ROLE_CONTEXT_DIM)
    manual = [seed]
    for _ in range(1, 4):
        manual.append(model.role_cell(manual[-1], step))
    assert torch.equal(after, torch.stack(manual))


def test_static_control_uses_global_context_and_not_active_cardinality():
    torch.manual_seed(19003)
    model = X19RoleModel(mode="static_global_roles", d_model=32)
    indices = torch.arange(4)
    expected = global_role_context(indices, dtype=model.role_seed.dtype)
    assert torch.equal(model.static_context(indices), expected)
    roles = model.roles(4)
    assert roles.shape == (4, ROLE_DIM)
    assert torch.isfinite(roles).all()


def test_identical_roles_produce_identical_rows_and_unrepaired_collision():
    torch.manual_seed(19004)
    scorer = RoleToSlotScorer()
    role = torch.randn(ROLE_DIM)
    roles = role[None, :].repeat(4, 1)
    slots = slot_descriptor(torch.arange(NUM_CANDIDATE_SLOTS))
    probs = scorer.probabilities_from_roles(roles, slots)
    assert torch.equal(probs[0], probs[1]) and torch.equal(probs[1], probs[2])
    assignment = probs.argmax(dim=1).tolist()
    assert assignment == [assignment[0]] * 4


def test_storage_bridge_row_and_slot_permutation_equivariance():
    torch.manual_seed(19005)
    scorer = RoleToSlotScorer()
    roles = torch.randn(4, ROLE_DIM)
    slots = slot_descriptor(torch.arange(NUM_CANDIDATE_SLOTS))
    base = scorer.logits_from_roles(roles, slots)
    rp = torch.tensor([2, 0, 3, 1])
    sp = torch.tensor([5, 1, 7, 0, 3, 6, 2, 4])
    assert torch.allclose(scorer.logits_from_roles(roles[rp], slots), base[rp], atol=1e-6, rtol=0.0)
    assert torch.allclose(scorer.logits_from_roles(roles, slots[sp]), base[:, sp], atol=1e-6, rtol=0.0)


def test_seen_bindings_are_valid_categorical_distributions_without_repair():
    torch.manual_seed(19006)
    models = cloned_x19_models(d_model=32)
    for mode in LEARNED_X19_MODES:
        for n in (2, 3, 4):
            binding = models[mode].soft_binding(n)
            assert binding.shape == (n, NUM_CANDIDATE_SLOTS)
            assert torch.isfinite(binding).all() and (binding >= 0).all()
            assert torch.allclose(binding.sum(dim=1), torch.ones(n), atol=1e-5, rtol=0.0)
            hard, assignment = models[mode].independent_argmax_binding(n)
            assert hard.shape == binding.shape
            assert assignment == binding.argmax(dim=1).tolist()


def test_hidden_targets_and_semantics_do_not_change_losses():
    torch.manual_seed(19007)
    models = cloned_x19_models(d_model=32)
    batch = make_variable_contextual_batch(12, 8, 19007, num_registers=4, split="train")
    for mode in LEARNED_X19_MODES:
        model = models[mode]
        base = model.loss_components(batch)
        altered_targets = torch.randint_like(batch.target_states, VALUE_MODULUS)
        altered_targets[:, -1, 0] = batch.target_states[:, -1, 0]
        changed_hidden = replace(batch, target_states=altered_targets, semantics=torch.randint_like(batch.semantics, 8))
        same = model.loss_components(changed_hidden)
        for key in ("answer_loss", "spread_penalty", "barrier_penalty", "total_loss"):
            assert torch.equal(base[key], same[key])

        answer_changed = altered_targets.clone()
        answer_changed[:, -1, 0] = (batch.target_states[:, -1, 0] + 1) % VALUE_MODULUS
        changed = model.loss_components(replace(batch, target_states=answer_changed))
        assert not torch.allclose(base["answer_loss"], changed["answer_loss"], atol=1e-7, rtol=1e-7)
        assert torch.equal(base["spread_penalty"], changed["spread_penalty"])
        assert torch.equal(base["barrier_penalty"], changed["barrier_penalty"])


def test_losses_finite_and_gradients_reach_constructor_bridge_and_executor():
    torch.manual_seed(19008)
    models = cloned_x19_models(d_model=32)
    batch = make_variable_contextual_batch(16, 8, 19008, num_registers=4, split="train")
    for mode, model in models.items():
        model.zero_grad(set_to_none=True)
        parts = model.loss_components(batch)
        for value in parts.values():
            assert torch.isfinite(value)
            assert float(value.detach()) >= 0.0
        parts["total_loss"].backward()
        executor_grad = sum(float(p.grad.abs().sum()) for p in model.executor.parameters() if p.grad is not None)
        assert executor_grad > 0.0
        if mode != "canonical_functional":
            assert model.role_cell is not None and model.storage_bridge is not None
            role_grad = sum(float(p.grad.abs().sum()) for p in model.role_cell.parameters() if p.grad is not None)
            bridge_grad = sum(float(p.grad.abs().sum()) for p in model.storage_bridge.parameters() if p.grad is not None)
            seed_grad = float(model.role_seed.grad.abs().sum()) if model.role_seed.grad is not None else 0.0
            assert role_grad > 0.0 and bridge_grad > 0.0 and seed_grad > 0.0


def test_seen_depth96_shapes_and_finiteness():
    torch.manual_seed(19009)
    models = cloned_x19_models(d_model=32)
    for n in (2, 3, 4):
        batch = make_variable_contextual_batch(2, 96, 19009 + n, num_registers=n, split="composition")
        for model in models.values():
            soft = model.rollout_soft(batch)
            hard = model.rollout_hard(batch, discrete_binding=True)
            assert soft.shape == batch.target_states.shape + (VALUE_MODULUS,)
            assert hard.shape == batch.target_states.shape
            assert torch.isfinite(soft).all() and torch.isfinite(hard).all()
