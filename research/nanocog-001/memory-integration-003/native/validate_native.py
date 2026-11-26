#!/usr/bin/env python3
"""Independently reproduce admitted native validation and adapter non-regression.

Private input directories are explicit arguments. No corpus/checkpoint is copied
into this public evidence directory. This uses the already exposed validation
split, not a new test-set result or a factual-memory accuracy score.
"""
from __future__ import annotations

import argparse
from array import array
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import torch
from torch.nn import functional as F


CHECKPOINT_SHA256 = "d8bab3d1d99a43dfb6cca2257f2caae4d3a7282d62b079c92874ce736a44ec3c"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weight_digest(model) -> str:
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def tensor_digest(tree) -> str:
    h = hashlib.sha256()

    def visit(value):
        if isinstance(value, torch.Tensor):
            h.update(str((tuple(value.shape), value.dtype)).encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value):
                h.update(key.encode())
                visit(value[key])
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        else:
            raise TypeError(type(value))

    visit(tree)
    return h.hexdigest()


@torch.inference_mode()
def evaluate_pair(original, attached, tokens):
    """All targets once, reset128, padding masked, two independent model objects."""
    total_loss, correct, targets = 0.0, 0, 0
    digest = hashlib.sha256()
    records = []
    starts = list(range(0, len(tokens) - 1, 128))
    for p in range(0, len(starts), 8):
        group = starts[p:p + 8]
        lengths = [min(128, len(tokens) - 1 - s) for s in group]
        width = max(lengths)
        x = torch.zeros((len(group), width), dtype=torch.long)
        y = torch.zeros_like(x)
        for j, (s, n) in enumerate(zip(group, lengths)):
            x[j, :n] = tokens[s:s + n]
            y[j, :n] = tokens[s + 1:s + n + 1]
        logits, state = original(x, feedback_enabled=True, memory_enabled=True)
        other, other_state = attached.native(x, feedback_enabled=True, memory_enabled=True)
        if not torch.equal(logits, other) or tensor_digest(state) != tensor_digest(other_state):
            raise AssertionError("adapter attachment changed native logits or full recurrent state")
        digest.update(logits.numpy().tobytes())
        losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                                 reduction="none").reshape_as(y)
        for j, (s, n) in enumerate(zip(group, lengths)):
            loss_sum = losses[j, :n].double().sum().item()
            hit = int((logits[j, :n].argmax(-1) == y[j, :n]).sum())
            records.append({"start": s, "targets": n, "loss_sum": loss_sum, "correct": hit})
            total_loss += loss_sum
            correct += hit
            targets += n
    if targets != len(tokens) - 1:
        raise AssertionError("validation coverage mismatch")
    return {"mean_nll": total_loss / targets, "next_character_accuracy": correct / targets,
            "targets": targets, "windows": records, "logits_sha256": digest.hexdigest(),
            "native_logits_and_complete_state_bit_equal_after_attachment": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    kit, data = args.kit.resolve(), args.data.resolve()
    manifest = json.loads((kit / "MANIFEST.json").read_text())
    for name, expected in manifest.items():
        path = kit / name
        if path.stat().st_size != expected["bytes"] or sha(path) != expected["sha256"]:
            raise ValueError("test kit manifest mismatch: " + name)
    checkpoint = kit / "COG_CONNECTED_1M.pt"
    if sha(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("unexpected native checkpoint")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for name in ("model.py", "base_model.py", "scripted_core.py"):
        embedded = [v for k, v in payload["source_sha256"].items() if Path(k).name == name]
        if embedded != [sha(kit / name)]:
            raise ValueError("native source does not match the pinned checkpoint: " + name)
    metadata = json.loads((data / "dataset.json").read_text())
    if sha(data / "val.bin") != metadata["splits"]["val"]["binary_sha256"]:
        raise ValueError("validation split checksum mismatch")
    ids = array("H")
    ids.frombytes((data / "val.bin").read_bytes())
    if sys.byteorder != "little":
        ids.byteswap()
    tokens = torch.tensor(ids, dtype=torch.long)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(39001)
    sys.path.insert(0, str(kit))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from model import ConnectedConfig, ConnectedCOGTinyLM
    from nanocog.memory.learned_archive import FactualAdapter

    original = ConnectedCOGTinyLM(ConnectedConfig(**payload["model_config"])).eval()
    original.load_state_dict(payload["model"], strict=True)
    clone = ConnectedCOGTinyLM(ConnectedConfig(**payload["model_config"])).eval()
    clone.load_state_dict(payload["model"], strict=True)
    chars = payload["tokenizer"]["chars"]
    attached = FactualAdapter(clone, chars).eval()
    count = sum(p.numel() for p in original.parameters())
    if count != 993574:
        raise AssertionError("unexpected native parameter count")
    original_hash = weight_digest(original)
    if weight_digest(attached.native) != original_hash:
        raise AssertionError("attachment altered native tensors")
    # Exercise the actual feature route, including its connected board, before
    # the complete native regression comparison. No synthetic labels are inputs.
    attached.native_features(["Amara holds the chest.", "Who holds the coin?"])
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    measured = evaluate_pair(original, attached, tokens)
    measured.update(cpu_seconds=time.process_time() - started_cpu,
                    wall_seconds=time.perf_counter() - started_wall)
    expected_nll = float(payload["val"]["mean_nll"])
    if abs(measured["mean_nll"] - expected_nll) > 5e-6:
        raise AssertionError("native validation no longer reproduces saved result")
    if weight_digest(attached.native) != original_hash or sha(checkpoint) != CHECKPOINT_SHA256:
        raise AssertionError("native model or checkpoint changed")
    result = {
        "status": "PASS",
        "claim_scope": "exposed native validation regression and adapter-attachment isolation only",
        "native_parameters": count,
        "full_adapter_parameters": sum(p.numel() for p in attached.parameters()),
        "trainable_adapter_parameters": sum(p.numel() for p in attached.parameters() if p.requires_grad),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "native_tensor_digest": original_hash,
        "kit_manifest_entries_verified": len(manifest),
        "native_source_sha256": {name: sha(kit / name) for name in ("model.py", "base_model.py", "scripted_core.py")},
        "adapter_source_sha256": sha(Path(__file__).resolve().parents[2] / "nanocog/memory/learned_archive.py"),
        "validation_sha256": sha(data / "val.bin"),
        "saved_validation_mean_nll": expected_nll,
        "reproduction_absolute_nll_difference": abs(measured["mean_nll"] - expected_nll),
        "python": platform.python_version(), "torch": torch.__version__,
        "threads": 1, "dtype": "float32", "device": "cpu",
        "context_policy": "reset128, nonoverlapping; all55769 targets once",
        "measurement": measured,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "measurement"}, sort_keys=True))
    print(json.dumps({k: v for k, v in measured.items() if k != "windows"}, sort_keys=True))


if __name__ == "__main__":
    main()
