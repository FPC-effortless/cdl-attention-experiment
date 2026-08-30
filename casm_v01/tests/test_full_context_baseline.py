import torch

from casm.eval_full_context_baseline import prompt_prefix
from casm.full_context_baseline import FullContextTransformer, baseline_config
from casm.process_data import make_process_batch
from casm.train_full_context_baseline import answer_only_loss


def test_baseline_parameter_budget_is_comparable():
    model = FullContextTransformer(baseline_config())
    count = model.parameter_count()
    assert 1_200_000 <= count <= 1_700_000


def test_generation_prefix_excludes_gold_answer():
    _, pex, _, _ = make_process_batch(1, 320, 123, hard=True, reasoning_steps=3)
    ex = pex[0].example
    prefix = prompt_prefix(ex)
    assert prefix.endswith(b"answer ")
    assert ex.answer.encode("utf-8") not in prefix[-len(ex.answer.encode("utf-8")) :]
    assert len(prefix) < len(ex.text.encode("utf-8"))


def test_answer_only_loss_reaches_attention_and_embedding():
    torch.manual_seed(7)
    cfg = baseline_config()
    cfg.d_model = 40
    cfg.n_layers = 1
    cfg.n_heads = 4
    cfg.n_kv_heads = 1
    cfg.d_ff = 80
    model = FullContextTransformer(cfg)
    tokens, _, answer_mask, _ = make_process_batch(
        3, 320, 321, hard=True, reasoning_steps=3
    )
    loss, acc = answer_only_loss(model, tokens, answer_mask)
    assert torch.isfinite(loss)
    assert 0.0 <= float(acc) <= 1.0
    loss.backward()
    assert float(model.embed.weight.grad.abs().sum()) > 0
    q_grad = model.blocks[0].attn.q_proj.weight.grad
    assert q_grad is not None
    assert float(q_grad.abs().sum()) > 0
