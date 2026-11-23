"""Deterministic inert-tensor engineering checks; no fit or held evaluation.

Importing this file imports Torch and the pinned models. Execution must occur
only after review of the exact source. Run as a script or call run_checks(device).
No optimizer is constructed and no optimizer update occurs.
"""
from __future__ import annotations

import argparse
import json
import traceback
from typing import Any, Callable
from unittest.mock import patch

import torch
from torch import Tensor
from torch.nn import functional as F

try:
    from .adapters import build_model_triplet, copy_common_shell, shared_tensor_mapping
except ImportError:
    from adapters import build_model_triplet, copy_common_shell, shared_tensor_mapping


def _models(device: str, layers: int = 2):
    return build_model_triplet(vocab_size=97, width=32, layers=layers, n_head=2,
                               n_kv_head=1, max_seq_len=32, seed=2841, device=device)


def _tokens(device: str, batch: int = 2, length: int = 9) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(5521)
    return torch.randint(0, 97, (batch, length), generator=generator).to(device)


def _close(a: Tensor, b: Tensor) -> float:
    torch.testing.assert_close(a, b, rtol=2e-4, atol=3e-5)
    return float((a - b).abs().max())


def _raises(call: Callable[[], Any]) -> str:
    try:
        call()
    except (ValueError, RuntimeError) as error:
        return str(error)
    raise AssertionError("invalid operation unexpectedly succeeded")


@torch.no_grad()
def _nontrivial_connectivity_fixture(models) -> None:
    """A labeled intervention, NOT the native initializer or a training update.

    Nonzero smear and output projections prevent default zero initializations
    from hiding broken causal/cache/gradient paths. Fixed read/write topic zero
    plus positive publishing deliberately fills a FIFO with varying payloads.
    """
    baseline = models["gqa"].model
    generator = torch.Generator(device="cpu").manual_seed(1807)
    baseline.smear_lambda.fill_(0.3)
    for block in baseline.transformer.h:
        for p in (block.attn.c_proj.weight, block.mlp.c_proj.weight):
            values = torch.rand(p.shape, generator=generator, dtype=torch.float32) * 0.08 - 0.04
            p.copy_(values.to(p.device))
    for key in ("packet", "local"):
        copy_common_shell(baseline, models[key].model)
    for mixer in models["packet"].model.mixers:
        packet = mixer.packets
        packet.publish.weight.zero_()
        packet.publish.bias.fill_(1.0)
        packet.read_topic.weight.zero_()
        packet.write_topic.weight.zero_()
        packet.packet_age.copy_(torch.linspace(-0.2, 0.3, packet.capacity,
                                               device=packet.packet_age.device).expand(packet.groups, -1))


def check_common_initialization_and_loss(device: str) -> dict:
    models, provenance = _models(device)
    base = dict(models["gqa"].model.named_parameters())
    checked = 0
    for key in ("packet", "local"):
        other = dict(models[key].model.named_parameters())
        for candidate_name, upstream_name in shared_tensor_mapping(2).items():
            expected = base[upstream_name].reshape(other[candidate_name].shape)
            assert torch.equal(other[candidate_name], expected), candidate_name
            assert other[candidate_name].data_ptr() != base[upstream_name].data_ptr(), candidate_name
            checked += 1
    assert models["gqa"].model.config.n_head > models["gqa"].model.config.n_kv_head
    assert models["gqa"].model.config.window_pattern == "L"
    x = _tokens(device)
    y = torch.roll(x, shifts=-1, dims=1)
    y[:, -1] = -1
    comparisons = {}
    sliced_comparisons = {}
    with torch.inference_mode():
        for key, model in models.items():
            logits = model(x)
            expected = F.cross_entropy(logits.reshape(-1, 97), y.reshape(-1),
                                       ignore_index=-1, reduction="none")
            actual = model(x, y, loss_reduction="none")
            comparisons[key] = _close(actual.reshape(-1), expected)
            _close(model(x, y), expected[y.reshape(-1) != -1].mean())
            report = model.parameter_report()
            assert sum(e["numel"] for e in report["entries"]) == report["total_unique_parameters"]
            assert len({e["name"] for e in report["entries"]}) == len(report["entries"])
            raw = _tokens(device, batch=2, length=10)
            sliced_x, sliced_y = raw[:, :-1], raw[:, 1:]
            assert not sliced_y.is_contiguous()
            expected_slice = F.cross_entropy(model(sliced_x).reshape(-1, 97),
                                            sliced_y.reshape(-1), reduction="none")
            sliced_comparisons[key] = _close(model(sliced_x, sliced_y, loss_reduction="none"), expected_slice)
    return {"exact_independent_tensor_copies_checked": checked, "loss_max_abs_errors": comparisons,
            "noncontiguous_next_token_target_max_abs_errors": sliced_comparisons,
            "parameter_counts": {k: m.parameter_report()["total_unique_parameters"] for k, m in models.items()},
            "initialization_status": provenance["status"], "optimizer_updates": 0}


def _parity(models, x: Tensor) -> dict:
    results = {}
    with torch.inference_mode():
        for key, model in models.items():
            model.eval()
            full = model(x)
            state = model.new_session(batch_size=x.size(0), max_tokens=20, session_id="parity")
            prefix = model.prefill(x[:, :3], state, session_id="parity")
            owned_storages = {name: t.untyped_storage().data_ptr()
                              for name, t in model.state_tensors(state, session_id="parity")
                              if "previous_embedding" not in name}
            outputs = [prefix]
            for i in range(3, x.size(1)):
                outputs.append(model.step(x[:, i:i+1], state, session_id="parity"))
                if key in ("packet", "gqa"):
                    now = {name: t.untyped_storage().data_ptr()
                           for name, t in model.state_tensors(state, session_id="parity")
                           if "previous_embedding" not in name}
                    assert now == owned_storages, "owned decode replaced persistent cache storage"
            result = {"full_vs_prefill_step_max_abs_error": _close(full, torch.cat(outputs, dim=1)),
                      "final_position": state.position,
                      "persistent_cache_storage_preserved": True if key in ("packet", "gqa") else "functional local state"}
            assert state.position == x.size(1)
            mutated = x.clone()
            mutated[:, 4:] = (mutated[:, 4:] + 19) % 97
            result["suffix_causality_max_abs_error"] = _close(full[:, :4], model(mutated)[:, :4])
            if key != "gqa":
                first, functional = model.model(x[:, :3], state=None)
                second, _ = model.model(x[:, 3:], state=functional)
                result["native_candidate_chunk_max_abs_error"] = _close(full, torch.cat((first, second), dim=1))
            else:
                result["cached_multitoken_rejection"] = _raises(
                    lambda: model.step(x[:, :2], state, session_id="parity"))
            results[key] = result
    return results


def check_causality_and_cache_parity(device: str) -> dict:
    models, _ = _models(device)
    x = _tokens(device)
    native = _parity(models, x)
    _nontrivial_connectivity_fixture(models)
    exercised = _parity(models, x)
    with torch.inference_mode():
        modified_history = x.clone()
        modified_history[:, :2] = (modified_history[:, :2] + 23) % 97
        before = models["gqa"](x)[:, -1]
        after = models["gqa"](modified_history)[:, -1]
        sensitivity = float((before - after).abs().max())
        assert sensitivity > 1e-7, "activated attention fixture did not respond to distant history"
    return {"native_initialization": native,
            "nonzero_smear_and_projection_fixture": exercised,
            "activated_gqa_distant_history_max_logit_change": sensitivity,
            "fixture_is_training": False}


def check_fork_reset_ownership(device: str) -> dict:
    models, _ = _models(device)
    _nontrivial_connectivity_fixture(models)
    x = _tokens(device)
    results = {}
    with torch.inference_mode():
        for key, model in models.items():
            parent = model.new_session(batch_size=1, max_tokens=16, session_id="parent")
            model.prefill(x[:1, :4], parent, session_id="parent")
            snapshots = [(name, t.clone()) for name, t in model.state_tensors(parent, session_id="parent")]
            fork = model.fork(parent, torch.tensor([0, 0], device=device),
                              session_id="parent", new_session_id="fork")
            pstorage = {t.untyped_storage().data_ptr() for _, t in model.state_tensors(parent, session_id="parent")}
            fstorage = {t.untyped_storage().data_ptr() for _, t in model.state_tensors(fork, session_id="fork")}
            assert pstorage.isdisjoint(fstorage), "fork shares retained state storage"
            next_tokens = x[:, 4:5]
            actual = model.step(next_tokens, fork, session_id="fork")
            reference = model(torch.cat((x[:1, :4].expand(2, -1), next_tokens), dim=1))[:, -1:]
            err = _close(actual, reference)
            assert parent.position == 4 and fork.position == 5
            for (name, old), (now_name, now) in zip(snapshots, model.state_tensors(parent, session_id="parent")):
                assert name == now_name and torch.equal(old, now), "fork mutated its parent"
            _close(model.step(x[:1, 4:5], parent, session_id="parent"), model(x[:1, :5])[:, -1:])
            bad_session = _raises(lambda: model.step(next_tokens, fork, session_id="another-request"))
            bad_mask = torch.ones_like(next_tokens, dtype=torch.bool)
            bad_mask[0, 0] = False
            padding = _raises(lambda: model.step(next_tokens, fork, session_id="fork", valid_mask=bad_mask))
            model.reset(fork, session_id="fork")
            assert fork.position == 0 and model.state_report(fork, session_id="fork")["tensor_storage_bytes"] == 0
            fresh = model.prefill(x[:, :4], fork, session_id="fork")
            _close(fresh, model(x[:, :4]))
            results[key] = {"fork_reference_max_abs_error": err, "parent_unchanged": True,
                            "whole_batch_reset": True, "session_rejection": bad_session,
                            "padding_rejection": padding}
        foreign = models["packet"].new_session(batch_size=1, max_tokens=16, session_id="foreign")
        results["cross_model_rejection"] = _raises(
            lambda: models["gqa"].prefill(x[:1, :4], foreign, session_id="foreign"))
    # Versions must invalidate state after a model edit/reload/optimizer update.
    model = models["local"]
    stale = model.new_session(batch_size=1, max_tokens=16, session_id="stale")
    with torch.no_grad():
        next(model.parameters()).add_(0.001)
    results["changed_weights_rejection"] = _raises(
        lambda: model.prefill(x[:1, :4], stale, session_id="stale"))
    baseline = models["gqa"]
    buffer_stale = baseline.new_session(batch_size=1, max_tokens=16, session_id="buffer-stale")
    with torch.no_grad():
        baseline.model.cos.add_(0.001)
    results["changed_buffer_rejection"] = _raises(
        lambda: baseline.prefill(x[:1, :4], buffer_stale, session_id="buffer-stale"))
    # Simulate an in-place failure after a cache write. A rejected input does not
    # poison a session, but a partially executed model update must do so.
    failing = baseline.new_session(batch_size=1, max_tokens=16, session_id="failed-step")
    baseline.prefill(x[:1, :4], failing, session_id="failed-step")
    def injected_failure(*args, **kwargs):
        kwargs["kv_cache"].k_cache[0, 0, 0, 0, 0].add_(0.5)
        raise RuntimeError("injected failure after partial cache mutation")
    with patch.object(baseline.model, "forward", side_effect=injected_failure):
        results["owned_update_failure"] = _raises(
            lambda: baseline.step(x[:1, 4:5], failing, session_id="failed-step"))
    results["poisoned_state_rejection"] = _raises(
        lambda: baseline.step(x[:1, 4:5], failing, session_id="failed-step"))
    assert "invalidated" in results["poisoned_state_rejection"]
    return results


def check_state_accounting(device: str) -> dict:
    models, _ = _models(device)
    x = _tokens(device, batch=2)
    reports = {}
    with torch.inference_mode():
        for key, model in models.items():
            state = model.new_session(batch_size=2, max_tokens=16, session_id="bytes")
            model.prefill(x[:, :3], state, session_id="bytes")
            report = model.state_report(state, session_id="bytes")
            assert report["tensor_logical_bytes"] == sum(t.numel() * t.element_size()
                    for _, t in model.state_tensors(state, session_id="bytes"))
            assert report["retained_state_bytes_estimate"] > report["tensor_storage_bytes"]
            if key == "packet":
                # B*s*(G*A*K*(D+1)+(W-1)*C)+8*B, each layer; plus preceding embedding.
                expected = 2 * (2 * 4 * (4 * 8 * 4 * 17 + 3 * 32) + 8 * 2) + 2 * 32 * 4
                assert report["tensor_logical_bytes"] == expected
                assert report["tensor_storage_bytes"] == expected
            elif key == "local":
                expected = 2 * (2 * 2 * 32 * 4) + 2 * 32 * 4
                assert report["tensor_storage_bytes"] == expected
            else:
                expected = 2 * 2 * 2 * 16 * 1 * 16 * 4 + 2 * 4 + 2 * 32 * 4
                assert report["tensor_logical_bytes"] == expected
                # The unchanged upstream preceding-embedding slice may retain
                # its whole prefill activation storage. Account rather than hide.
                assert report["tensor_storage_bytes"] >= expected
            large = model.new_session(batch_size=2, max_tokens=24, session_id="larger")
            model.prefill(x[:, :3], large, session_id="larger")
            larger = model.state_report(large, session_id="larger")
            difference = larger["tensor_storage_bytes"] - report["tensor_storage_bytes"]
            expected_delta = 2 * 2 * 2 * (24 - 16) * 1 * 16 * 4 if key == "gqa" else 0
            assert difference == expected_delta
            reports[key] = {"state": report, "capacity_storage_delta_bytes": difference}
    return reports


def _gradient_audit(models, x: Tensor, *, require_all_nonzero: bool) -> dict:
    results = {}
    targets = torch.roll(x, -1, 1)
    for key, model in models.items():
        model.train()
        model.zero_grad(set_to_none=True)
        loss = model(x, targets)
        assert bool(torch.isfinite(loss)), "non-finite synthetic loss"
        loss.backward()
        entries = []
        for name, p in model.model.named_parameters():
            if not p.requires_grad:
                continue
            present = p.grad is not None
            finite = present and bool(torch.isfinite(p.grad).all())
            nonzero = finite and bool(p.grad.ne(0).any())
            entries.append({"name": name, "present": present, "finite": finite, "nonzero": nonzero})
        assert all(e["present"] and e["finite"] for e in entries), entries
        if require_all_nonzero:
            assert all(e["nonzero"] for e in entries), [e for e in entries if not e["nonzero"]]
        results[key] = {"loss_nats": float(loss.detach()), "parameters": entries,
                        "zero_gradient_names": [e["name"] for e in entries if not e["nonzero"]]}
        model.zero_grad(set_to_none=True)
        model.eval()
    return results


def check_gradient_connectivity(device: str) -> dict:
    # With two layers the backout tap is at the final layer; its scale is nearly
    # cancelled by final RMSNorm. Three layers put an actual block after the tap.
    models, _ = _models(device, layers=3)
    x = _tokens(device, length=12)
    native = _gradient_audit(models, x, require_all_nonzero=False)
    # Zero smear and zero projection matrices intentionally block some native
    # initialization gradients. Test connectivity on a separate labeled fixture.
    _nontrivial_connectivity_fixture(models)
    fixture = _gradient_audit(models, x, require_all_nonzero=True)
    return {"native_initialization": native, "intervened_connectivity_fixture": fixture,
            "gradient_fixture_layers": 3,
            "optimizer_updates": 0, "trained_checkpoint_created": False}


def run_checks(device: str = "cpu") -> dict[str, Any]:
    checks = []
    functions = (check_common_initialization_and_loss, check_causality_and_cache_parity,
                 check_fork_reset_ownership, check_state_accounting, check_gradient_connectivity)
    for function in functions:
        try:
            details = function(device)
            checks.append({"name": function.__name__, "status": "PASS", "details": details})
        except Exception as error:
            checks.append({"name": function.__name__, "status": "FAIL",
                           "error_type": type(error).__name__, "error": str(error),
                           "traceback": traceback.format_exc()})
    return {"status": "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL",
            "scope": "INERT_TENSOR_ENGINEERING_ONLY", "device": str(device),
            "torch_version": torch.__version__, "training_updates": 0,
            "scientific_comparison_admitted": False, "checks": checks}


def test_cpu_adapter_contracts() -> None:
    """Optional pytest entry point; execution is controlled by the launcher."""
    report = run_checks("cpu")
    assert report["status"] == "PASS", json.dumps(report, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run_checks(args.device)
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
