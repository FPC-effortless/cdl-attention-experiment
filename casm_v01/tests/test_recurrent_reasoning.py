import torch
from dataclasses import replace

from casm.data import VOCAB_SIZE, make_batch
from casm.model import CASM, CASMConfig
from casm.recurrent_model import CASMRecurrent
from casm.set_utility import set_conditioned_utility_loss


def cfg():
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


def test_one_step_matches_existing_qk_logits():
    torch.manual_seed(11)
    c = cfg()
    base = CASM(c)
    recurrent = CASMRecurrent(c, reasoning_steps=1)
    recurrent.load_state_dict(base.state_dict())
    x, _ = make_batch(3, 65, 100)
    base.eval(); recurrent.eval()
    with torch.no_grad():
        a = base(x, return_aux=False)["logits"]
        b = recurrent(x, return_aux=False)["logits"]
    assert torch.allclose(a, b, atol=1e-6, rtol=1e-5)


def test_recurrent_depth_does_not_change_parameter_count():
    a = CASMRecurrent(cfg(), reasoning_steps=1)
    b = CASMRecurrent(cfg(), reasoning_steps=3)
    assert a.parameter_count() == b.parameter_count()


def test_three_step_forward_backward_finite():
    torch.manual_seed(12)
    model = CASMRecurrent(cfg(), reasoning_steps=3)
    x, _, mask = make_batch(3, 97, 101, return_answer_mask=True)
    out = model(x, target_weights=1.0 + 7.0 * mask[:, 1:].float())
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_set_utility_updates_only_qk_geometry():
    torch.manual_seed(13)
    model = CASMRecurrent(cfg(), reasoning_steps=3)
    x, _, mask = make_batch(4, 97, 102, return_answer_mask=True)
    utility = set_conditioned_utility_loss(
        model,
        x,
        mask[:, 1:],
        temperature=0.7,
        min_gain_std=0.0,
    )
    assert torch.isfinite(utility["loss"])
    assert utility["positions"] > 0
    utility["loss"].backward()
    assert model.router.q.weight.grad is not None
    assert model.router.q.weight.grad.abs().sum() > 0
    assert model.router.k.weight.grad is not None
    assert model.router.k.weight.grad.abs().sum() > 0
    assert model.router.value.weight.grad is None
    assert model.memory_gate.weight.grad is None
    assert model.lm_head.weight.grad is None


def test_set_utility_reports_informative_fraction():
    torch.manual_seed(14)
    model = CASMRecurrent(cfg(), reasoning_steps=3)
    x, _, mask = make_batch(3, 97, 103, return_answer_mask=True)
    utility = set_conditioned_utility_loss(
        model,
        x,
        mask[:, 1:],
        temperature=0.7,
        min_gain_std=0.01,
    )
    assert 0.0 <= float(utility["informative_fraction"]) <= 1.0
    assert float(utility["informative_positions"]) <= float(utility["positions"])
