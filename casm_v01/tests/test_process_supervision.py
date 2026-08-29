from dataclasses import replace

import torch

from casm.data import VOCAB_SIZE
from casm.model import CASMConfig
from casm.process_data import make_process_batch, state_process
from casm.process_supervision import ProcessHead, fixed_trace_code, process_alignment_loss
from casm.recurrent_model import CASMRecurrent


def tiny_core():
    cfg = replace(
        CASMConfig(vocab_size=VOCAB_SIZE),
        d_model=48,
        n_layers=1,
        n_heads=4,
        n_kv_heads=1,
        d_ff=96,
        memory_dim=24,
        memory_slots=4,
        state_slots=2,
        chunk_size=16,
        mtp_horizons=1,
        use_compression_score=False,
        compression_loss_weight=0.0,
        compression_predictor_loss_weight=0.0,
    )
    return CASMRecurrent(cfg, reasoning_steps=3)


def test_state_prompt_exposes_initial_state():
    import random

    pex = state_process(random.Random(7), hard=False)
    assert "\ninitial " in pex.example.text
    assert len(pex.traces) == 3
    assert all("=" in x for x in pex.traces)


def test_process_batch_anchor_precedes_gold_answer():
    toks, pex, mask, anchors = make_process_batch(4, 320, 99, hard=True)
    assert toks.shape == mask.shape
    for i, ex in enumerate(pex):
        ans_positions = torch.nonzero(mask[i], as_tuple=False).flatten()
        assert len(ans_positions) > 0
        assert int(anchors[i]) == int(ans_positions[0]) - 1
        assert len(ex.traces) == 3


def test_trace_code_is_order_sensitive_and_unit_norm():
    a = fixed_trace_code("a b c", 64)
    b = fixed_trace_code("c b a", 64)
    assert torch.isclose(a.norm(), torch.tensor(1.0), atol=1e-5)
    assert not torch.allclose(a, b)


def test_process_loss_backpropagates_into_recurrent_core_and_head():
    torch.manual_seed(3)
    model = tiny_core()
    head = ProcessHead(model.cfg.d_model, 32)
    toks, pex, _, anchors = make_process_batch(3, 256, 1234, hard=False)
    loss, cosine = process_alignment_loss(
        model, head, toks, anchors, [x.traces for x in pex]
    )
    assert torch.isfinite(loss)
    assert torch.isfinite(cosine)
    loss.backward()
    assert head.net[0].weight.grad is not None
    assert model.router.q.weight.grad is not None
    assert torch.isfinite(model.router.q.weight.grad).all()
