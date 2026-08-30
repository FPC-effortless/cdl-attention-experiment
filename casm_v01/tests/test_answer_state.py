from dataclasses import replace

import torch

from casm.answer_state import CASMAnswerState, answer_state_loss, answer_targets
from casm.data import VOCAB_SIZE
from casm.model import CASMConfig
from casm.process_data import make_process_batch


def cfg():
    return replace(
        CASMConfig(vocab_size=VOCAB_SIZE),
        d_model=48,
        n_layers=1,
        n_heads=4,
        n_kv_heads=1,
        d_ff=96,
        memory_dim=24,
        memory_slots=4,
        state_slots=2,
        chunk_size=24,
        use_compression_score=False,
        compression_loss_weight=0.0,
        compression_predictor_loss_weight=0.0,
        mtp_loss_weight=0.0,
        verifier_loss_weight=0.0,
    )


def test_answer_suffix_is_not_consumed():
    torch.manual_seed(1)
    model = CASMAnswerState(cfg(), reasoning_steps=3, answer_slots=20).eval()
    toks, _, _, anchors = make_process_batch(3, 320, 123, hard=True, reasoning_steps=3)
    corrupted = toks.clone()
    for bi, anchor in enumerate(anchors.tolist()):
        corrupted[bi, anchor + 1 :] = 97 + bi
    with torch.no_grad():
        a = model(toks, anchors).logits_steps[-1]
        b = model(corrupted, anchors).logits_steps[-1]
    assert torch.allclose(a, b, atol=0.0, rtol=0.0)


def test_reasoning_depth_adds_no_parameters():
    torch.manual_seed(2)
    a = CASMAnswerState(cfg(), reasoning_steps=1, answer_slots=20)
    torch.manual_seed(2)
    b = CASMAnswerState(cfg(), reasoning_steps=3, answer_slots=20)
    assert a.parameter_count() == b.parameter_count()
    assert set(a.state_dict()) == set(b.state_dict())


def test_answer_loss_reaches_router_and_answer_state():
    torch.manual_seed(3)
    model = CASMAnswerState(cfg(), reasoning_steps=3, answer_slots=20)
    toks, pex, _, anchors = make_process_batch(4, 320, 321, hard=True, reasoning_steps=3)
    targets = answer_targets([x.example.answer for x in pex], 20)
    out = model(toks, anchors)
    loss, losses = answer_state_loss(out, targets)
    assert len(losses) == 3
    loss.backward()
    answer_grad = sum(
        float(p.grad.abs().sum())
        for p in model.answer_update.parameters()
        if p.grad is not None
    )
    router_grad = sum(
        float(p.grad.abs().sum())
        for p in model.core.router.parameters()
        if p.grad is not None
    )
    lm_grad = float(model.core.embed.weight.grad.abs().sum())
    assert answer_grad > 0
    assert router_grad > 0
    assert lm_grad > 0
