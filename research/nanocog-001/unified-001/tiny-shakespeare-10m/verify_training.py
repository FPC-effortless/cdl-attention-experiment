"""Independent bounded falsifiers for the Tiny Shakespeare neural prototype.

No corpus fit or model-quality estimate is performed by this script.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from model import COGTinyLM, LatentRecordBoard, ModelConfig, affine_scan


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from tensors(value[key])
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tensors(item)


def maximum_difference(left, right):
    aa, bb = list(tensors(left)), list(tensors(right))
    if len(aa) != len(bb):
        return float("inf")
    return max((float((a - b).abs().max()) for a, b in zip(aa, bb)), default=0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "review_checks.json")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(77129)
    started = time.perf_counter()
    checks = []

    def record(name, passed, **evidence):
        checks.append({"name": name, "passed": bool(passed), **evidence})

    # A separate sequential calculation checks the optimized scan identity,
    # including nonzero initial state and a non-power-of-two sequence length.
    a = torch.rand(2, 53, 11) * 0.99
    b = torch.randn(2, 53, 11)
    initial = torch.randn(2, 11)
    actual = affine_scan(a, b, initial)
    state = initial
    expected = []
    for t in range(b.shape[1]):
        state = a[:, t] * state + b[:, t]
        expected.append(state)
    reference = torch.stack(expected, 1)
    delta = maximum_difference(actual, reference)
    record("affine_scan_matches_sequential_nonzero_state", delta <= 2e-6, max_abs_error=delta, tolerance=2e-6)

    config = ModelConfig(vocab_size=65)
    model = COGTinyLM(config).eval()
    count = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    record("actual_parameter_budget", 9_000_000 <= count <= 11_000_000 and trainable == count,
           total_parameters=count, trainable_parameters=trainable, config=dataclasses.asdict(config))
    x = torch.randint(0, config.vocab_size, (2, 41))
    state0 = model.init_state(2)
    for t in tensors(state0):
        t.copy_(torch.randn_like(t) * 0.1)
    state_snapshot = [t.clone() for t in tensors(state0)]

    with torch.no_grad():
        full_logits, full_state = model(x, state0)
        step_state = state0
        parts = []
        for t in range(x.shape[1]):
            logits, step_state = model(x[:, t:t+1], step_state)
            parts.append(logits)
        step_logits = torch.cat(parts, 1)
        step_delta = maximum_difference(full_logits, step_logits)
        state_delta = maximum_difference(full_state, step_state)
        record("full_incremental_parity", step_delta <= 2e-5 and state_delta <= 2e-5,
               logits_max_abs_error=step_delta, state_max_abs_error=state_delta, tolerance=2e-5)

        chunk_state = state0
        chunks = []
        for left, right in [(0, 7), (7, 8), (8, 23), (23, 41)]:
            logits, chunk_state = model(x[:, left:right], chunk_state)
            chunks.append(logits)
        chunk_delta = maximum_difference(full_logits, torch.cat(chunks, 1))
        chunk_state_delta = maximum_difference(full_state, chunk_state)
        record("irregular_chunk_parity", chunk_delta <= 2e-5 and chunk_state_delta <= 2e-5,
               logits_max_abs_error=chunk_delta, state_max_abs_error=chunk_state_delta, tolerance=2e-5)

        changed = x.clone()
        changed[:, 19:] = (changed[:, 19:] + 17) % config.vocab_size
        changed_logits, _ = model(changed, state0)
        causal_delta = maximum_difference(full_logits[:, :19], changed_logits[:, :19])
        record("unseen_suffix_does_not_change_prefix", causal_delta == 0.0, max_abs_error=causal_delta)

        fresh_a, _ = model(x)
        fresh_b, _ = model(x)
        reset_delta = maximum_difference(fresh_a, fresh_b)
        record("fresh_state_determinism", reset_delta == 0.0, max_abs_error=reset_delta)
        single_a, _ = model(x[:1])
        isolated_delta = maximum_difference(fresh_a[:1], single_a)
        record("batch_stream_isolation", isolated_delta <= 2e-5, max_abs_error=isolated_delta, tolerance=2e-5)
        unchanged = all(torch.equal(a, b) for a, b in zip(state_snapshot, tensors(state0)))
        record("provided_state_is_not_mutated", unchanged)

    # Actual next-token objective, rather than an auxiliary route-only loss.
    # One backward pass tests gradient connectivity, not learning quality.
    model.zero_grad(set_to_none=True)
    logits, _ = model(x)
    target = torch.randint(0, config.vocab_size, x.shape)
    loss = F.cross_entropy(logits.flatten(0, 1), target.flatten())
    loss.backward()
    gradients = {}
    for name, p in model.named_parameters():
        gradients[name] = {
            "present": p.grad is not None,
            "finite": bool(torch.isfinite(p.grad).all()) if p.grad is not None else False,
            "l1": float(p.grad.abs().sum()) if p.grad is not None else 0.0,
        }
    disconnected = [name for name, info in gradients.items() if not info["present"] or not info["finite"] or info["l1"] == 0.0]
    record("lm_loss_reaches_all_enabled_parameter_tensors", not disconnected,
           cross_entropy=float(loss.detach()), disconnected_or_zero=disconnected, gradients=gradients)

    # The first read must not depend on a current/future write, while later
    # reads must provide a learning path into the write-value head.
    model.zero_grad(set_to_none=True)
    logits, _ = model(x)
    F.cross_entropy(logits[:, 0], target[:, 0]).backward()
    write_grad = model.record_board.write_value.weight.grad
    first_write_grad = float(write_grad.abs().sum()) if write_grad is not None else 0.0
    record("first_output_has_no_current_write_gradient", first_write_grad == 0.0, write_value_gradient_l1=first_write_grad)

    # Controlled exact-forward board fixture. Force both categorical factors
    # to slot zero, force SET, make projected values and reads directly visible.
    tiny = ModelConfig(vocab_size=5, d_model=8, n_layers=1, memory_dim=2, dropout=0.0)
    board = LatentRecordBoard(tiny).eval()
    with torch.no_grad():
        for p in board.parameters():
            p.zero_()
        board.norm.weight.fill_(1.0)
        for layer in (board.query_entity, board.query_relation, board.write_entity, board.write_relation):
            layer.bias[0] = 20.0
        board.write_commit.bias.fill_(100.0)
        board.read_gate.bias.fill_(100.0)
        board.write_value.weight[0, 0] = 1.0
        board.write_value.weight[1, 1] = 1.0
        board.read_value.weight[0, 0] = 1.0
        board.read_value.weight[1, 1] = 1.0
        features = torch.randn(1, 7, 8)
        memory = torch.randn(1, tiny.n_entities * tiny.n_relations, tiny.memory_dim)
        written = torch.tanh(board.norm(features)[..., :2])
        output, final_memory = board(features, memory)
        expected_read = torch.cat((memory[:, None, 0], written[:, :-1]), 1)
        expected_out = features.clone()
        expected_out[..., :2] += expected_read
        board_delta = maximum_difference(output, expected_out)
        last_delta = maximum_difference(final_memory[:, 0], written[:, -1])
        untouched = torch.equal(final_memory[:, 1:], memory[:, 1:])
        record("controlled_board_reads_before_write", board_delta <= 1e-6 and last_delta <= 1e-6,
               output_max_abs_error=board_delta, final_write_max_abs_error=last_delta)
        record("controlled_board_preserves_unaddressed_slots", untouched)
        board.write_commit.bias.fill_(-100.0)
        held_output, held_memory = board(features, memory)
        held_expected = features.clone()
        held_expected[..., :2] += memory[:, None, 0]
        hold_delta = maximum_difference(held_output, held_expected)
        record("controlled_hold_preserves_memory", torch.equal(held_memory, memory) and hold_delta <= 1e-6,
               output_max_abs_error=hold_delta)

    result = {
        "scope": "independent constructed neural-interface falsifiers; no model fit or quality conclusion",
        "source_sha256": hashlib.sha256(Path(__file__).with_name("model.py").read_bytes()).hexdigest(),
        "review_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch_version": torch.__version__, "device": "cpu", "threads": args.threads,
        "seed": 77129, "elapsed_seconds": time.perf_counter() - started,
        "passed": sum(c["passed"] for c in checks), "total": len(checks), "checks": checks,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("source_sha256", "passed", "total", "elapsed_seconds")}, indent=2))
    for c in checks:
        if not c["passed"]:
            print(json.dumps(c, indent=2))
    if result["passed"] != result["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
