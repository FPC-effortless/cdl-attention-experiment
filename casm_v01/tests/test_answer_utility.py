import torch

from casm.answer_utility import answer_utility_qk_loss
from casm.data import VOCAB_SIZE, make_batch
from casm.model import CASM, CASMConfig


def tiny_cfg():
    return CASMConfig(
        vocab_size=VOCAB_SIZE,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=1,
        d_ff=128,
        chunk_size=16,
        memory_slots=4,
        state_slots=2,
        memory_dim=32,
        mtp_horizons=2,
        compression_future_tokens=3,
        use_compression_score=False,
        compression_loss_weight=0.0,
        compression_predictor_loss_weight=0.0,
    )


def test_answer_utility_is_finite_and_trains_only_qk_geometry():
    torch.manual_seed(9)
    model = CASM(tiny_cfg())
    x, _, mask = make_batch(4, 129, 991, hard=True, return_answer_mask=True)
    utility = answer_utility_qk_loss(model, x, mask[:, 1:])
    assert torch.isfinite(utility["loss"])
    assert utility["positions"] > 0

    model.zero_grad(set_to_none=True)
    utility["loss"].backward()
    assert model.router.q.weight.grad is not None
    assert model.router.k.weight.grad is not None
    assert model.router.q.weight.grad.abs().sum() > 0
    assert model.router.k.weight.grad.abs().sum() > 0

    # The counterfactual teacher and recurrent replay are detached. The
    # auxiliary objective therefore trains routing geometry rather than
    # directly changing the value path or language-model head.
    assert model.router.value.weight.grad is None
    assert model.memory_gate.weight.grad is None
    assert model.lm_head.weight.grad is None


def test_answer_utility_has_no_first_chunk_pseudo_memory_supervision():
    torch.manual_seed(10)
    model = CASM(tiny_cfg())
    x, _, mask = make_batch(2, 16, 992, return_answer_mask=True)
    utility = answer_utility_qk_loss(model, x, mask[:, 1:])
    assert float(utility["positions"]) == 0.0
    assert float(utility["loss"]) == 0.0
