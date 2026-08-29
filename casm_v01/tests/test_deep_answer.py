from __future__ import annotations

from dataclasses import replace

import torch

from casm.data import VOCAB_SIZE
from casm.deep_answer import deep_answer_loss
from casm.model import CASMConfig
from casm.process_data import make_process_batch
from casm.recurrent_model import CASMRecurrent


def tiny_model() -> CASMRecurrent:
    cfg = replace(
        CASMConfig(vocab_size=VOCAB_SIZE),
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=1,
        d_ff=128,
        memory_dim=32,
        memory_slots=4,
        state_slots=2,
        chunk_size=16,
        use_compression_score=False,
        compression_loss_weight=0.0,
        compression_predictor_loss_weight=0.0,
    )
    return CASMRecurrent(cfg, reasoning_steps=3)


def test_deep_answer_loss_is_finite_and_supervises_two_steps():
    torch.manual_seed(7)
    model = tiny_model()
    toks, _, answer_mask, _ = make_process_batch(
        3, 256, 12345, hard=True, reasoning_steps=3
    )
    loss, metrics = deep_answer_loss(model, toks, answer_mask)
    assert torch.isfinite(loss)
    assert float(loss) > 0.0
    assert int(metrics["supervised_steps"].item()) == 2
    assert "step1_answer_nll" in metrics
    assert "step2_answer_nll" in metrics


def test_deep_answer_loss_reaches_shared_lm_head_and_router():
    torch.manual_seed(11)
    model = tiny_model()
    toks, _, answer_mask, _ = make_process_batch(
        4, 288, 54321, hard=True, reasoning_steps=3
    )
    loss, _ = deep_answer_loss(model, toks, answer_mask)
    model.zero_grad(set_to_none=True)
    loss.backward()

    assert model.lm_head.weight.grad is not None
    assert float(model.lm_head.weight.grad.abs().sum()) > 0.0
    assert model.router.q.weight.grad is not None
    assert float(model.router.q.weight.grad.abs().sum()) > 0.0
    assert model.router.k.weight.grad is not None
    assert float(model.router.k.weight.grad.abs().sum()) > 0.0


def test_deep_supervision_adds_no_parameters():
    model = tiny_model()
    before = sum(p.numel() for p in model.parameters())
    toks, _, answer_mask, _ = make_process_batch(
        2, 224, 10101, hard=False, reasoning_steps=3
    )
    loss, _ = deep_answer_loss(model, toks, answer_mask)
    assert torch.isfinite(loss)
    after = sum(p.numel() for p in model.parameters())
    assert before == after
