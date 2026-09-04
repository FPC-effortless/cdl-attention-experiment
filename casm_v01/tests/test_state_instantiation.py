from __future__ import annotations

from dataclasses import replace

import torch

from casm.explicit_compute import ProgramBatch, VALUE_MODULUS
from casm.state_instantiation_data import (
    NUM_CANDIDATES,
    OUTPUT_CANDIDATE,
    TRAIN_LIVE_CARDINALITIES,
    backward_live_mask,
    make_state_instantiation_batch,
    training_live_cardinality_for_step,
)
from casm.state_instantiation_model import (
    LEARNED_MODES,
    STORAGE_LAMBDA,
    StateInstantiationModel,
    candidate_code,
    cloned_x20_models,
)
from casm.variable_cardinality_binding import EMPTY_VALUE


def _program_with_operands(program: ProgramBatch, *, a=None, b=None, dst=None) -> ProgramBatch:
    return ProgramBatch(
        initial=program.initial,
        commands=program.commands,
        semantics=program.semantics,
        arg_a=program.arg_a if a is None else a,
        arg_b=program.arg_b if b is None else b,
        dst=program.dst if dst is None else dst,
        target_states=program.target_states,
    )


def test_backward_liveness_is_temporal_and_rooted_at_output():
    # t0: 1 -> 2, t1: 2 -> 0. Both 1 and 2 are therefore live for final 0.
    mask = backward_live_mask([1, 2], [1, 2], [2, 0])
    assert mask[0] and mask[1] and mask[2]
    assert sum(mask) == 3


def test_backward_liveness_drops_overwritten_versions():
    # t0: candidate 1 is computed from 7; t1 overwrites candidate 1 from 2;
    # t2 uses the *new* candidate-1 value to produce output 0. Candidate 7's
    # earlier contribution is therefore dead and must not enter the live set.
    mask = backward_live_mask([7, 2, 1], [7, 2, 1], [1, 1, 0])
    assert mask[0] and mask[1] and mask[2]
    assert not mask[7]
    assert sum(mask) == 3


def test_generator_exact_live_cardinality_and_distractor_mentions():
    for n in (2, 3, 4, 5, 6):
        batch = make_state_instantiation_batch(8, 12, 8000 + n, live_cardinality=n, split="iid")
        assert batch.program.initial.shape == (8, NUM_CANDIDATES)
        assert (batch.live_mask.sum(dim=1) == n).all()
        assert batch.live_mask[:, OUTPUT_CANDIDATE].all()
        for row in range(batch.batch_size):
            live = batch.live_mask[row]
            mentioned = torch.zeros(NUM_CANDIDATES, dtype=torch.bool)
            for field in (batch.program.arg_a[row], batch.program.arg_b[row], batch.program.dst[row]):
                mentioned[field] = True
            assert ((~live) & mentioned).any()


def test_training_schedule_is_exact_234_cycle():
    assert [training_live_cardinality_for_step(i) for i in range(1, 10)] == [2, 3, 4] * 3
    assert TRAIN_LIVE_CARDINALITIES == (2, 3, 4)


def test_candidate_codes_are_fixed_nonlearned_and_unique():
    code = candidate_code()
    assert code.shape == (8, 9)
    assert torch.unique(code, dim=0).shape[0] == 8
    assert code.requires_grad is False


def test_learned_pair_parameterization_and_initialization_match():
    torch.manual_seed(7)
    models = cloned_x20_models(d_model=32)
    a = models["learned_instantiation"]
    b = models["structure_blind_gate"]
    assert a.parameter_count() == b.parameter_count()
    assert a.trainable_parameter_count() == b.trainable_parameter_count()
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb
        assert torch.equal(pa, pb)


def test_no_learned_per_candidate_table_exists():
    model = StateInstantiationModel(mode="learned_instantiation", d_model=32)
    assert model.constructor is not None
    # Scope the prohibition to the constructor. The inherited validated executor
    # legitimately has an 8-row operator-command embedding; that is not a
    # candidate-identity table. Constructor command-family embeddings are also
    # allowed because they encode supplied command identity rather than candidates.
    for name, p in model.constructor.named_parameters():
        if name.startswith("command."):
            continue
        assert not (p.ndim >= 2 and p.shape[0] == NUM_CANDIDATES), (name, tuple(p.shape))


def test_hidden_live_mask_cannot_change_learned_gates():
    torch.manual_seed(11)
    batch = make_state_instantiation_batch(4, 12, 991, live_cardinality=3, split="train")
    model = StateInstantiationModel(mode="learned_instantiation", d_model=32)
    g1 = model.gates(batch)
    altered = replace(batch, live_mask=~batch.live_mask)
    g2 = model.gates(altered)
    assert torch.equal(g1, g2)


def test_canonical_mask_uses_live_mask_but_all_records_does_not():
    batch = make_state_instantiation_batch(3, 12, 992, live_cardinality=3, split="train")
    canonical = StateInstantiationModel(mode="canonical_live_mask", d_model=32)
    all_records = StateInstantiationModel(mode="all_records", d_model=32)
    assert torch.equal(canonical.gates(batch).bool(), batch.live_mask)
    assert torch.equal(all_records.gates(batch), torch.ones_like(all_records.gates(batch)))


def test_graph_connectivity_affects_graph_model_but_not_blind_ablation():
    torch.manual_seed(13)
    batch = make_state_instantiation_batch(4, 12, 993, live_cardinality=3, split="train")
    models = cloned_x20_models(d_model=32)
    permutation = torch.tensor([7, 6, 5, 4, 3, 2, 1, 0])
    altered_program = _program_with_operands(
        batch.program,
        a=permutation[batch.program.arg_a],
        b=permutation[batch.program.arg_b],
        dst=permutation[batch.program.dst],
    )
    altered = replace(batch, program=altered_program)
    blind_1 = models["structure_blind_gate"].gates(batch)
    blind_2 = models["structure_blind_gate"].gates(altered)
    graph_1 = models["learned_instantiation"].gates(batch)
    graph_2 = models["learned_instantiation"].gates(altered)
    assert torch.equal(blind_1, blind_2)
    assert not torch.equal(graph_1, graph_2)


def test_gated_initial_state_uses_empty_for_absent_record():
    model = StateInstantiationModel(mode="all_records", d_model=32)
    initial = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]])
    gates = torch.ones(1, 8)
    gates[:, 3] = 0.0
    probs = model.executor.initial_probs(initial, gates)
    assert probs[0, 3, EMPTY_VALUE].item() == 1.0
    assert probs[0, 3, :VALUE_MODULUS].sum().item() == 0.0
    assert probs[0, 2, 5].item() == 1.0


def test_soft_gates_are_normalized_existence_probabilities():
    batch = make_state_instantiation_batch(5, 12, 994, live_cardinality=4, split="train")
    for mode in LEARNED_MODES:
        model = StateInstantiationModel(mode=mode, d_model=32)
        gates = model.gates(batch)
        assert gates.shape == (5, 8)
        assert torch.isfinite(gates).all()
        assert ((gates > 0) & (gates < 1)).all()


def test_hard_instantiation_is_raw_half_threshold():
    model = StateInstantiationModel(mode="all_records", d_model=32)
    gates = torch.tensor([[0.49, 0.50, 0.51, 0.1, 0.9, 0.2, 0.8, 0.3]])
    hard = (gates >= 0.5)
    assert hard.tolist() == [[False, True, True, False, True, False, True, False]]


def test_answer_only_loss_ignores_live_mask_for_learned_model():
    torch.manual_seed(17)
    batch = make_state_instantiation_batch(4, 12, 995, live_cardinality=3, split="train")
    model = StateInstantiationModel(mode="learned_instantiation", d_model=32)
    l1 = model.loss_components(batch)
    l2 = model.loss_components(replace(batch, live_mask=~batch.live_mask))
    for key in l1:
        assert torch.equal(l1[key], l2[key])


def test_storage_cost_has_downward_gate_gradient():
    gates = torch.tensor([[0.2, 0.8]], requires_grad=True)
    storage = STORAGE_LAMBDA * gates.mean()
    storage.backward()
    assert (gates.grad > 0).all()  # gradient descent therefore reduces gate values


def test_answer_loss_reaches_constructor_and_executor():
    torch.manual_seed(19)
    batch = make_state_instantiation_batch(3, 12, 996, live_cardinality=3, split="train")
    model = StateInstantiationModel(mode="learned_instantiation", d_model=32)
    parts = model.loss_components(batch)
    parts["total_loss"].backward()
    constructor_grad = sum(
        float(p.grad.abs().sum()) for p in model.constructor.parameters() if p.grad is not None
    )
    executor_grad = sum(
        float(p.grad.abs().sum()) for p in model.executor.parameters() if p.grad is not None
    )
    assert constructor_grad > 0.0
    assert executor_grad > 0.0
