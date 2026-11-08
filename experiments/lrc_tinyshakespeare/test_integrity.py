"""Pre-fit leakage, execution-parity and faithful-training checks.

Run directly to emit auditable JSON, without requiring pytest.  These are
mechanism checks, not model-capability evidence.  Fit-quality evaluation belongs
to the separately frozen benchmark protocol.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time

import torch

from adapters import (ARMS, LEXICAL_ARMS, build_model, build_gated_model,
                      make_stream, train_step)
from auxiliary import teachers
from backbone_models import SuffixCache


@torch.no_grad()
def check_causality_and_parity(model, length=160, tokens=None):
    """Test past logits against changed future, exact streaming and prefill."""
    model.eval()
    gen = torch.Generator().manual_seed(58217)
    # A repeated short alphabet exercises indexed/cache hits and old pages.
    # The optional frozen-data prefix supports independent trained replay.
    if tokens is None:
        x = torch.randint(98, 106, (1, length), generator=gen)
    else:
        x = torch.as_tensor(tokens, dtype=torch.long)
        if x.ndim == 1:
            x = x[None]
        if x.ndim != 2 or x.shape[0] != 1 or x.shape[1] < 18:
            raise ValueError("observed parity tokens must be one prefix of length>=18")
        if (x < 1).any() or (x > 256).any():
            raise ValueError("observed parity tokens must be real raw-byte IDs")
        length = x.shape[1]
    original = model(x).logits.log_softmax(-1)
    changed = x.clone()
    changed[:, length // 2:] = torch.randint(120, 125, (1, length - length // 2), generator=gen)
    future = model(changed).logits.log_softmax(-1)
    prefix_error = float((original[:, :length // 2] - future[:, :length // 2]).abs().max())
    # Two forwards with unrelated document rows must not contaminate a retry.
    model(torch.randint(1, 257, (2, 23), generator=gen))
    retry = model(x).logits.log_softmax(-1)
    retry_error = float((original - retry).abs().max())
    s = make_stream(model)
    stepped = torch.stack([s.step(int(t))[0] for t in x[0]], 0)[None]
    stream_error = float((original - stepped).abs().max())
    stream_argmax = bool(torch.equal(original.argmax(-1), stepped.argmax(-1)))
    prefill_at = length - 17  # >128 for the bounded Transformer KV path.
    p = make_stream(model)
    first = p.prefill(x[0, :prefill_at])
    continuation = torch.stack([first[0]] + [p.step(int(t))[0] for t in x[0, prefill_at:]], 0)[None]
    reference = original[:, prefill_at - 1:]
    prefill_error = float((reference - continuation).abs().max())
    prefill_argmax = bool(torch.equal(reference.argmax(-1), continuation.argmax(-1)))
    # Fresh state from reset must reproduce a second document exactly.
    s.reset()
    restarted = torch.stack([s.step(int(t))[0] for t in x[0, :11]], 0)[None]
    reset_error = float((stepped[:, :11] - restarted).abs().max())
    row = {"future_prefix_max_error": prefix_error, "document_retry_max_error": retry_error,
           "stream_max_error": stream_error, "stream_argmax_equal": stream_argmax,
           "prefill_max_error": prefill_error, "prefill_argmax_equal": prefill_argmax,
           "reset_max_error": reset_error, "retained_state_bytes": p.state_bytes(),
           "length": length, "prefill_at": prefill_at,
           "input": "fixed_repeated_alphabet" if tokens is None else "caller_supplied_prefix"}
    assert prefix_error == 0, row
    assert retry_error == 0, row
    assert torch.allclose(original, stepped, atol=1e-5, rtol=1e-5), row
    assert torch.allclose(reference, continuation, atol=1e-5, rtol=1e-5), row
    assert stream_argmax and prefill_argmax and reset_error == 0, row
    return row


def check_training_recipe(arm):
    torch.manual_seed(60211)
    m = build_model(arm)
    # Repeat structure yields eligible lexical teacher labels and page bits.
    x = (torch.arange(48)[None].expand(4, -1) + torch.arange(4)[:, None]) % 7 + 98
    y = torch.roll(x, shifts=-1, dims=1)
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    opt = torch.optim.AdamW(m.parameters(), lr=0.002, weight_decay=0.0)
    receipt = train_step(m, opt, x, y, seed=300761)
    changed = [n for n, p in m.named_parameters() if not torch.equal(p, before[n])]
    # Hard-route surrogates can be exactly zero for a selected coordinate.  A
    # zero change is recorded rather than falsely requiring every router update
    # to help.  However, skipping the second forward would be a real regression.
    router_names = ("write_relation", "write_address", "route.", "query", "rank", "page_", "continue_ref")
    route_changed = [n for n in changed if any(k in n for k in router_names)]
    route_gradients = {n: float(p.grad.abs().sum()) for n, p in m.named_parameters()
                       if p.grad is not None and any(k in n for k in router_names)}
    expected_forwards = 1 if arm in ("gru", "gru_cache", "transformer") else 2
    assert receipt["forward_passes"] == expected_forwards, receipt
    assert math.isfinite(receipt["loss"]) and changed, receipt
    assert getattr(m, "cache_data", None) is None
    if arm in LEXICAL_ARMS:
        assert receipt["address_aux"] > 0
        assert route_changed, "lexical supervised routing heads must actually update"
    return {"receipt": receipt, "changed_parameter_tensors": len(changed),
            "changed_route_parameter_tensors": route_changed,
            "route_gradient_abs_sums": route_gradients}


def check_lexical_teachers():
    x = torch.tensor([[98, 99, 98, 100, 98, 99, 98, 100]])
    write, query, gold = teachers(x)
    assert int(write[0, 0]) == -100
    for t, j in enumerate(gold[0].tolist()):
        if j >= 0:
            assert 0 < j < t
            assert x[0, j - 1] == x[0, t]
        # Each teacher address for this prefix is independent of future bytes.
        w, q, g = teachers(x[:, :t + 1])
        assert torch.equal(write[:, :t + 1], w)
        assert torch.equal(query[:, :t + 1], q)
        assert torch.equal(gold[:, :t + 1], g)
    cache = SuffixCache()
    for t, token in enumerate(x[0].tolist()):
        ids, _ = cache.observe(token)
        assert all(token == 0 or token in x[0, :t + 1] for token in ids)
    return {"gold_prior_continuations": gold.tolist(), "strictly_past": True,
            "prefix_teacher_parity": True, "cache_only_observed_tokens": True}


def run_checks():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    begin = time.perf_counter()
    result = {"torch_version": torch.__version__, "arms": [], "utility": []}
    for arm in ARMS:
        torch.manual_seed(57411)
        m = build_model(arm)
        row = {"arm": arm, "parameters": sum(p.numel() for p in m.parameters()),
               "parity": check_causality_and_parity(m), "training": check_training_recipe(arm)}
        result["arms"].append(row)
    backbone = build_model("v4").eval()
    for variant in ("utility", "always_on", "last_off", "always_off"):
        model = build_gated_model(backbone, variant=variant)
        result["utility"].append({"variant": variant, "parity": check_causality_and_parity(model)})
    result["lexical_teachers"] = check_lexical_teachers()
    result["seconds"] = time.perf_counter() - begin
    result["passed"] = True
    return result


if __name__ == "__main__":
    result = run_checks()
    output = json.dumps(result, indent=2)
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(output + "\n")
    else:
        print(output)
