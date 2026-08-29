import torch

from casm.data import VOCAB_SIZE, gzip_teacher_distributions, make_batch
from casm.model import CASMConfig
from casm.selective import SelectiveCASM


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
        use_compression_score=False,
    )


def test_selective_forward_backward_and_gate_gradients():
    torch.manual_seed(7)
    cfg = tiny_cfg()
    model = SelectiveCASM(cfg)
    tokens, _, answer_mask = make_batch(3, 65, 123, hard=True, return_answer_mask=True)
    weights = torch.ones_like(tokens[:, 1:], dtype=torch.float32) + 7.0 * answer_mask[:, 1:].float()
    teacher = gzip_teacher_distributions(tokens, cfg.chunk_size, cfg.memory_slots, cfg.state_slots)
    out = model(tokens, external_teacher=teacher, teacher_alpha=0.5, target_weights=weights)
    assert torch.isfinite(out["loss"])
    assert 0.0 < float(out["write_gate_mean"]) < 1.0
    assert 0.0 <= float(out["erase_gate_mean"]) < 1.0
    out["loss"].backward()
    assert model.write_gate_net[-1].weight.grad is not None
    assert model.write_gate_net[-1].weight.grad.abs().sum() > 0
    assert model.erase_gate_net[-1].weight.grad is not None
    assert model.erase_gate_net[-1].weight.grad.abs().sum() > 0
    assert model.router.q.weight.grad is not None
    assert model.router.q.weight.grad.abs().sum() > 0


def test_write_force_recovers_full_write_limit():
    torch.manual_seed(11)
    cfg = tiny_cfg()
    model = SelectiveCASM(cfg)
    b = 2
    ring, strength, state = model._init_selective_memory(b, torch.device("cpu"), torch.float32)
    summary = torch.randn(b, cfg.d_model)
    new_mem = torch.tanh(model.memory_in(summary))
    surprise = torch.ones(b)
    model.write_force = 1.0
    ring2, strength2, write_prob, _, _, _ = model._selective_write(
        ring, strength, state, summary, new_mem, surprise
    )
    assert torch.allclose(write_prob, torch.ones_like(write_prob))
    assert torch.allclose(ring2[:, -1], new_mem, atol=1e-6)
    assert torch.allclose(strength2[:, -1], torch.ones_like(strength2[:, -1]))


def test_observed_surprise_does_not_use_boundary_future_target():
    cfg = tiny_cfg()
    model = SelectiveCASM(cfg)
    logits = torch.randn(2, 5, cfg.vocab_size)
    targets = torch.randint(0, 200, (2, 5))
    a = model._observed_surprise(logits, targets)
    targets2 = targets.clone()
    targets2[:, -1] = (targets2[:, -1] + 37) % 200
    b = model._observed_surprise(logits, targets2)
    assert torch.allclose(a, b)
