from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import inspect

import torch
import torch.nn.functional as F

from casm.explicit_compute import VALUE_MODULUS
from casm.noncontractive_role_dynamics import (
    ADDRESS_BETA,
    ALPHA,
    LEARNED_X19D_MODES,
    ORTHOGONAL_X19D_MODES,
    RAW_MATRIX_INIT_STD,
    ROLE_DIM,
    X19DRoleKeyedModel,
    address_probabilities,
    cayley_orthogonal,
    cloned_x19d_models,
    decode_memory,
    hard_address_probabilities,
    read_memory,
    unconstrained_matrix,
    write_memory,
)
from casm.run_stable_cardinality_executor import training_cardinality_for_step
from casm.variable_cardinality_binding import NUM_INTERNAL_VALUES
from casm.variable_contextual_data import make_variable_contextual_batch


def test_constants_and_learned_parameter_match():
    assert ROLE_DIM == 32
    assert ALPHA == 0.1
    assert ADDRESS_BETA == 16.0
    assert RAW_MATRIX_INIT_STD == 0.5
    torch.manual_seed(19101)
    models = cloned_x19d_models(d_model=32)
    left = models["unconstrained_recursive"]
    right = models["orthogonal_recursive"]
    assert left.parameter_count() == right.parameter_count()
    assert left.trainable_parameter_count() == right.trainable_parameter_count()


def test_recurrent_constructor_initialization_is_bit_identical_and_frozen_control_is_nontrainable():
    torch.manual_seed(19102)
    models = cloned_x19d_models(d_model=32)
    frozen = models["frozen_random_orthogonal"]
    unconstrained = models["unconstrained_recursive"]
    orthogonal = models["orthogonal_recursive"]
    for model in (frozen, unconstrained, orthogonal):
        assert model.constructor_seed is not None and model.raw_matrix is not None
    assert torch.equal(frozen.constructor_seed, unconstrained.constructor_seed)
    assert torch.equal(frozen.constructor_seed, orthogonal.constructor_seed)
    assert torch.equal(frozen.raw_matrix, unconstrained.raw_matrix)
    assert torch.equal(frozen.raw_matrix, orthogonal.raw_matrix)
    assert not frozen.constructor_seed.requires_grad
    assert not frozen.raw_matrix.requires_grad
    assert unconstrained.constructor_seed.requires_grad and unconstrained.raw_matrix.requires_grad
    assert orthogonal.constructor_seed.requires_grad and orthogonal.raw_matrix.requires_grad


def test_unconstrained_and_cayley_formulas_exact():
    torch.manual_seed(19103)
    raw = torch.randn(ROLE_DIM, ROLE_DIM) * RAW_MATRIX_INIT_STD
    eye = torch.eye(ROLE_DIM)
    expected_unconstrained = eye + ALPHA * raw
    assert torch.equal(unconstrained_matrix(raw), expected_unconstrained)

    skew = raw - raw.T
    expected_q = torch.linalg.solve(eye - ALPHA * skew, eye + ALPHA * skew)
    actual_q = cayley_orthogonal(raw)
    assert torch.allclose(actual_q, expected_q, atol=1e-7, rtol=0.0)
    assert torch.allclose(actual_q.T @ actual_q, eye, atol=1e-5, rtol=0.0)


def test_orthogonal_map_remains_orthogonal_after_representative_optimizer_step():
    torch.manual_seed(19104)
    model = cloned_x19d_models(d_model=32)["orthogonal_recursive"]
    batch = make_variable_contextual_batch(16, 8, 19104, num_registers=4, split="train")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    optimizer.zero_grad(set_to_none=True)
    loss = model.fixed_answer_loss(batch)
    loss.backward()
    optimizer.step()
    q = model.transition_matrix().detach()
    eye = torch.eye(ROLE_DIM, dtype=q.dtype)
    assert torch.allclose(q.T @ q, eye, atol=1e-5, rtol=0.0)


def test_frozen_random_constructor_does_not_change_after_optimizer_step():
    torch.manual_seed(19105)
    model = cloned_x19d_models(d_model=32)["frozen_random_orthogonal"]
    seed_before = model.constructor_seed.detach().clone()
    matrix_before = model.raw_matrix.detach().clone()
    batch = make_variable_contextual_batch(16, 8, 19105, num_registers=4, split="train")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    optimizer.zero_grad(set_to_none=True)
    model.fixed_answer_loss(batch).backward()
    optimizer.step()
    assert torch.equal(model.constructor_seed, seed_before)
    assert torch.equal(model.raw_matrix, matrix_before)


def test_no_learned_identity_tables_or_constructor_embeddings():
    torch.manual_seed(19106)
    models = cloned_x19d_models(d_model=32)
    forbidden = ("external_id", "cardinality_id", "role_index", "step_index", "address_id", "slot_id")
    for mode in LEARNED_X19D_MODES:
        model = models[mode]
        constructor_names = [name for name, _ in model.named_parameters() if not name.startswith("executor.")]
        assert set(constructor_names) == {"constructor_seed", "raw_matrix"}
        for name in constructor_names:
            assert not any(token in name.lower() for token in forbidden)


def test_roles_use_same_transition_repeatedly_without_index_or_cardinality_features():
    torch.manual_seed(19107)
    models = cloned_x19d_models(d_model=32)
    for mode in LEARNED_X19D_MODES:
        model = models[mode]
        seed = model.normalized_seed()
        transition = model.transition_matrix()
        manual = [seed]
        for _ in range(1, 4):
            nxt = transition @ manual[-1]
            if mode == "unconstrained_recursive":
                nxt = F.normalize(nxt, dim=0, eps=1e-8)
            manual.append(nxt)
        assert torch.allclose(model.roles(4), torch.stack(manual), atol=1e-7, rtol=0.0)


def test_addressing_is_exact_role_cosine_softmax_and_hard_argmax():
    torch.manual_seed(19108)
    roles = F.normalize(torch.randn(4, ROLE_DIM), dim=-1)
    expected = F.softmax(ADDRESS_BETA * (roles @ roles.T), dim=-1)
    actual = address_probabilities(roles, roles)
    assert torch.allclose(actual, expected, atol=1e-7, rtol=0.0)
    hard = hard_address_probabilities(roles, roles)
    assert torch.equal(hard.argmax(dim=-1), expected.argmax(dim=-1))


def test_memory_record_permutation_with_keys_preserves_reads_writes_and_decode():
    torch.manual_seed(19109)
    queries = F.normalize(torch.randn(4, ROLE_DIM), dim=-1)
    keys = queries.clone()
    memory = torch.rand(3, 4, NUM_INTERNAL_VALUES)
    memory = memory / memory.sum(dim=-1, keepdim=True)
    query_index = torch.tensor([0, 2, 3])
    new_value = torch.rand(3, NUM_INTERNAL_VALUES)
    new_value = new_value / new_value.sum(dim=-1, keepdim=True)

    base_address = address_probabilities(queries, keys)
    base_read = read_memory(memory, query_index, base_address)
    base_written = write_memory(memory, query_index, new_value, base_address)
    base_decoded = decode_memory(base_written, base_address)

    perm = torch.tensor([2, 0, 3, 1])
    perm_address = address_probabilities(queries, keys[perm])
    perm_memory = memory[:, perm]
    perm_read = read_memory(perm_memory, query_index, perm_address)
    perm_written = write_memory(perm_memory, query_index, new_value, perm_address)
    perm_decoded = decode_memory(perm_written, perm_address)

    assert torch.allclose(base_read, perm_read, atol=1e-6, rtol=0.0)
    assert torch.allclose(base_decoded, perm_decoded, atol=1e-6, rtol=0.0)


def test_duplicate_role_keys_create_unrepaired_hard_ambiguity():
    torch.manual_seed(19110)
    roles = F.normalize(torch.randn(4, ROLE_DIM), dim=-1)
    roles[1] = roles[0]
    probs = address_probabilities(roles, roles)
    hard = hard_address_probabilities(roles, roles).argmax(dim=-1)
    assert torch.allclose(probs[0], probs[1], atol=0.0, rtol=0.0)
    assert int(hard[0]) == int(hard[1])
    assert hard.tolist() != list(range(4))


def test_read_write_decode_source_has_no_direct_memory_index_bypass():
    for fn in (read_memory, write_memory, decode_memory):
        source = inspect.getsource(fn)
        assert "memory[external_index]" not in source
        assert "memory[:, external_index]" not in source
        assert "memory[dst]" not in source


def test_answer_only_supervision_and_gradients():
    torch.manual_seed(19111)
    models = cloned_x19d_models(d_model=32)
    batch = make_variable_contextual_batch(16, 8, 19111, num_registers=4, split="train")

    for mode, model in models.items():
        base = model.fixed_answer_loss(batch)
        altered = torch.randint_like(batch.target_states, VALUE_MODULUS)
        altered[:, -1, 0] = batch.target_states[:, -1, 0]
        hidden_changed = replace(batch, target_states=altered, semantics=torch.randint_like(batch.semantics, 8))
        same = model.fixed_answer_loss(hidden_changed)
        assert torch.equal(base, same)

        answer_changed = altered.clone()
        answer_changed[:, -1, 0] = (batch.target_states[:, -1, 0] + 1) % VALUE_MODULUS
        changed = model.fixed_answer_loss(replace(batch, target_states=answer_changed))
        assert not torch.allclose(base, changed, atol=1e-7, rtol=1e-7)

        model.zero_grad(set_to_none=True)
        model.fixed_answer_loss(batch).backward()
        executor_grad = sum(float(p.grad.abs().sum()) for p in model.executor.parameters() if p.grad is not None)
        assert executor_grad > 0.0
        if mode in LEARNED_X19D_MODES:
            assert model.constructor_seed.grad is not None and float(model.constructor_seed.grad.abs().sum()) > 0.0
            assert model.raw_matrix.grad is not None and float(model.raw_matrix.grad.abs().sum()) > 0.0
        if mode == "frozen_random_orthogonal":
            assert model.constructor_seed.grad is None and model.raw_matrix.grad is None


def test_seen_addressing_and_depth96_shapes_are_finite():
    torch.manual_seed(19112)
    models = cloned_x19d_models(d_model=32)
    for n in (2, 3, 4):
        batch = make_variable_contextual_batch(2, 96, 19112 + n, num_registers=n, split="composition")
        for model in models.values():
            stats = model.address_stats(n)
            assert stats["hard_all_self"]
            assert stats["hard_tied_query_count"] == 0
            soft = model.rollout_soft(batch)
            hard = model.rollout_hard(batch, discrete_binding=True)
            soft_address_hard_values = model.rollout_hard(batch, discrete_binding=False)
            assert soft.shape == batch.target_states.shape + (NUM_INTERNAL_VALUES,)
            assert hard.shape == soft_address_hard_values.shape == batch.target_states.shape
            assert torch.isfinite(soft).all()


def test_training_schedule_is_exact_234_cycle():
    observed = [training_cardinality_for_step(step) for step in range(1, 13)]
    assert observed == [2, 3, 4] * 4
