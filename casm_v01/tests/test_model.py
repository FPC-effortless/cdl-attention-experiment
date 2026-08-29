import torch

from casm.data import VOCAB_SIZE, make_batch
from casm.model import CASM, CASMConfig


def tiny_cfg():
    return CASMConfig(vocab_size=VOCAB_SIZE, d_model=64, n_layers=2, n_heads=4, n_kv_heads=1, d_ff=128, chunk_size=16, memory_slots=4, state_slots=2, memory_dim=32, mtp_horizons=2, compression_future_tokens=3)


def test_forward_backward_finite():
    torch.manual_seed(1); model = CASM(tiny_cfg()); x, _ = make_batch(3, 65, 10); out = model(x)
    assert out["logits"].shape == (3, 64, VOCAB_SIZE); assert torch.isfinite(out["loss"]); out["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_compression_router_receives_gradient():
    torch.manual_seed(2); model = CASM(tiny_cfg()); x, _ = make_batch(4, 65, 11); out = model(x); out["loss"].backward()
    grad = model.router.score_mlp[0].weight.grad
    assert grad is not None and grad.abs().sum() > 0


def test_parameter_count_small():
    assert CASM(tiny_cfg()).parameter_count() < 2_000_000


def test_answer_mask_marks_only_answer_bytes():
    x, _, mask = make_batch(2, 129, 1234, return_answer_mask=True)
    assert mask.shape == x.shape and mask.any() and not mask[x >= 256].any()


def test_weighted_objective_runs_and_reports_memory_diagnostics():
    torch.manual_seed(3); model = CASM(tiny_cfg()); x, _, mask = make_batch(3, 65, 12, return_answer_mask=True)
    out = model(x, target_weights=1.0 + 7.0 * mask[:, 1:].float())
    assert torch.isfinite(out["loss"]); assert torch.isfinite(out["compression_gain_std"]); assert torch.isfinite(out["memory_gate_mean"])
