"""Independent pre-fit checks for split accounting and utility supervision.

The scoring-accounting fixture is synthetic and uses a constant, untrained
distribution.  It does not inspect a fitted model's validation or test quality.
Run this file directly; its optional argument is the evidence JSON path.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

import run as benchmark
from adapters import build_model, build_gated_model


class ConstantDistribution(nn.Module):
    """A data-independent sentinel used only to audit target enumeration."""

    def forward(self, x):
        assert x.ndim == 2
        assert not bool(((x[:, :-1] == 0) & (x[:, 1:] != 0)).any())
        return SimpleNamespace(logits=torch.zeros((*x.shape, 257)), metrics={})


def check_data_and_budget():
    raw = (benchmark.P / "data/input.txt").read_bytes()
    ds = benchmark.data()
    assert hashlib.sha256(raw).hexdigest() == benchmark.CORPUS_SHA
    assert len(raw) == 1115394
    assert {k: len(v) for k, v in ds.items()} == {
        "train": 1003854, "dev": 55770, "test": 55770,
    }
    recovered = torch.cat([ds[k] for k in ("train", "dev", "test")]) - 1
    assert recovered.tolist() == list(raw)
    assert benchmark.STEPS * benchmark.BATCH * benchmark.CONTEXT == 1048576
    assert benchmark.GATE_STEPS * benchmark.BATCH * benchmark.CONTEXT == 307200
    assert 2 * benchmark.GATE_STEPS == 1200
    for seed in benchmark.SEEDS:
        left = random.Random(1100000 + seed)
        right = random.Random(1100000 + seed)
        for _ in range(20):
            x, y, starts = benchmark.new_batch(ds["train"], left)
            xx, yy, other = benchmark.new_batch(ds["train"], right)
            assert starts == other and torch.equal(x, xx) and torch.equal(y, yy)
            assert torch.equal(x[:, 1:], y[:, :-1])
            for row, start in enumerate(starts):
                assert 0 <= start < len(ds["train"]) - benchmark.CONTEXT
                assert torch.equal(x[row], ds["train"][start:start + benchmark.CONTEXT])
                assert torch.equal(y[row], ds["train"][start + 1:start + benchmark.CONTEXT + 1])
    return {
        "corpus_sha256": benchmark.CORPUS_SHA,
        "split_bytes": {k: len(v) for k, v in ds.items()},
        "strict_train_only_shifted_windows": True,
        "paired_sampling_reproduced_for_all_seeds": True,
        "backbone_target_exposures": 1048576,
        "utility_extra_target_exposures": 307200,
        "utility_teacher_forwards": 1200,
    }


def check_all_target_accounting():
    # This has the TEST length but is not any corpus split's contents.
    tokens = (torch.arange(55770, dtype=torch.long) % 64) + 65
    result = {}
    expected_positions = list(range(1, len(tokens)))
    for context, expected_tail in ((128, 89), (512, 473)):
        scored = benchmark.score(ConstantDistribution(), tokens, context)
        covered = [i for row in scored["blocks"]
                   for i in range(row["start"] + 1, row["start"] + row["targets"] + 1)]
        assert covered == expected_positions
        assert scored["targets"] == 55769
        assert scored["blocks"][-1]["targets"] == expected_tail
        assert math.isclose(scored["nll_nats_per_byte"], math.log(257), abs_tol=1e-6)
        assert scored["correct"] == 0  # constant argmax is PAD; no target is PAD
        result[str(context)] = {
            "targets": scored["targets"],
            "blocks": len(scored["blocks"]),
            "last_block_targets": scored["blocks"][-1]["targets"],
            "every_target_exactly_once_in_order": True,
        }
    return result


def check_utility_semantics():
    torch.manual_seed(49386)
    backbone = build_model("v4").eval()
    gated = build_gated_model(backbone)
    x = torch.randint(98, 106, (4, 128))
    y = torch.randint(98, 106, x.shape)
    original = {name: value.clone() for name, value in backbone.state_dict().items()}
    with torch.no_grad():
        on_reference = backbone(x).logits
        off_reference = backbone(x, memory="disabled").logits
        on_gated = build_gated_model(backbone, variant="always_on")(x).logits
        off_gated = build_gated_model(backbone, variant="always_off")(x).logits
        assert torch.equal(on_reference, on_gated)
        assert torch.equal(off_reference, off_gated)

        captured = []
        hook = gated.layers[1].write_relation.register_forward_pre_hook(
            lambda _module, args: captured.append(args[0].clone()))
        try:
            on = gated(x, overrides={(1, "read_gate"): torch.ones_like(x)}).logits
            off = gated(x, overrides={(1, "read_gate"): torch.zeros_like(x)}).logits
            mask = torch.randint(0, 2, x.shape)
            mixed = gated(x, overrides={(1, "read_gate"): mask}).logits
        finally:
            hook.remove()
        assert len(captured) == 3 and all(torch.equal(v, captured[0]) for v in captured)
        selected = torch.where(mask.bool()[..., None], on, off)
        assert torch.equal(mixed, selected)

    features, loss_on, loss_off = benchmark.utility_teacher(gated, x, y)
    changed_y = (y % 256) + 1
    changed_features, _, _ = benchmark.utility_teacher(gated, x, changed_y)
    assert torch.equal(features, changed_features)
    assert not features.requires_grad
    assert all(not p.requires_grad for p in gated.parameters())
    assert torch.allclose(loss_on, F.cross_entropy(on.flatten(0, 1), y.flatten(),
                                                  reduction="none").reshape(y.shape))
    assert torch.allclose(loss_off, F.cross_entropy(off.flatten(0, 1), y.flatten(),
                                                   reduction="none").reshape(y.shape))

    # Demonstrate a real standalone gate optimizer update without changing the
    # frozen teacher or inheriting a training graph through its features.
    head = nn.Linear(24, 2)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    with torch.no_grad():
        head.bias[1] = -0.02
    previous_head = {name: value.clone() for name, value in head.state_dict().items()}
    optimizer = benchmark.opt_for(head)
    optimizer.zero_grad(set_to_none=True)
    logits = head(features)
    gain = (loss_off - loss_on).clamp(-1, 1)
    loss = F.mse_loss(logits[..., 1] - logits[..., 0], gain)
    loss.backward()
    norm = nn.utils.clip_grad_norm_(head.parameters(), 1.0)
    assert torch.isfinite(norm)
    optimizer.step()
    assert any(not torch.equal(value, previous_head[name]) for name, value in head.state_dict().items())
    assert sum(p.numel() for p in head.parameters()) == 50
    for name, value in original.items():
        assert torch.equal(value, backbone.state_dict()[name])
        assert torch.equal(value, gated.state_dict()[name])
    return {
        "full_on_matches_original_v4_exactly": True,
        "full_off_matches_original_memory_disabled_exactly": True,
        "on_off_mixed_share_identical_pre_last_layer_features": True,
        "mixed_gate_selects_exact_per_position_branch_logits": True,
        "teacher_features_independent_of_next_byte_targets": True,
        "frozen_backbone_unchanged_after_head_update": True,
        "head_fitted_parameters": 50,
    }


def run_checks():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return {
        "reviewer": "independent_protocol_review",
        "torch_version": str(torch.__version__),
        "data_and_budget": check_data_and_budget(),
        "synthetic_target_accounting": check_all_target_accounting(),
        "utility_semantics": check_utility_semantics(),
        "passed": True,
    }


if __name__ == "__main__":
    evidence = json.dumps(run_checks(), indent=2) + "\n"
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(evidence)
    else:
        print(evidence)
