"""Fixed-window language-preservation gate for the completed fusion campaign.

All models receive the exact same 64 nonoverlapping 128-target validation
windows, with recurrence reset independently for every window. The local-only
admitted parent defines the <=10% relative NLL degradation gate. The original
Connected COG, including its native record board, is a separate reference.

This program refuses an incomplete campaign and does not inspect generated
factual answers. Its timing fields document execution, not serving benchmarks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from .model import FusionModel, NATIVE_SHA256


VALIDATION_SHA256 = "42586f62517e6504c04c45990fe22bf8ee50f49f45872a953cbbb645d3b9feae"
WINDOW_SEED = 20261123
WINDOW_COUNT = 64
TARGETS_PER_WINDOW = 128
SEEDS = (17, 29, 41)
ARMS = ("memory", "no_memory")
NATIVE_DEFAULT = Path("/workspace/scratch/2a5ba884844e/deliverables/COG_CONNECTED_1M.pt")
VALIDATION_DEFAULT = Path("/workspace/scratch/2a5ba884844e/cog_connected_1m_003/data/val.bin")
HERE = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixed_windows(tokens: np.ndarray, *, count=WINDOW_COUNT, targets=TARGETS_PER_WINDOW,
                  seed=WINDOW_SEED) -> tuple[list[int], torch.Tensor]:
    """Select unique target blocks, with one preceding token per block.

    Adjacent blocks may share the boundary input token, but their scored target
    positions never overlap. Sampling is deterministic and without replacement.
    """
    if tokens.ndim != 1 or count < 1 or targets < 1:
        raise ValueError("expected a token vector and positive sampling geometry")
    starts_available = list(range(0, len(tokens) - targets, targets))
    if len(starts_available) < count:
        raise ValueError("not enough disjoint validation target blocks")
    starts = sorted(random.Random(seed).sample(starts_available, count))
    windows = torch.tensor(np.stack([np.asarray(tokens[start:start + targets + 1], dtype=np.int64)
                                    for start in starts]), dtype=torch.long)
    return starts, windows


def preservation_gate(mean_nll: float, parent_mean_nll: float) -> dict:
    if not math.isfinite(mean_nll) or not math.isfinite(parent_mean_nll) or parent_mean_nll <= 0:
        raise ValueError("NLL gate requires finite losses and a positive parent loss")
    relative = mean_nll / parent_mean_nll - 1.0
    return {"reference": "parent_local_only", "metric": "relative_mean_nll_change",
            "relative_mean_nll_change": relative, "allowed_relative_increase": 0.10,
            "passed": mean_nll <= parent_mean_nll * 1.10,
            "character_perplexity_ratio": math.exp(mean_nll - parent_mean_nll)}


def completed_runs(root: Path) -> list[dict]:
    """Resolve and verify every completed fit before any evaluation starts."""
    found = []
    locked_dependencies = None
    lock_path = root.parent / "TRAINING_LOCK.json"
    for seed in SEEDS:
        for arm in ARMS:
            directory = root / f"{arm}_seed{seed}"
            result_path, checkpoint_path = directory / "RESULT.json", directory / "final.pt"
            identity_path, log_path = directory / "IDENTITY.json", directory / "train.jsonl"
            required = (result_path, checkpoint_path, identity_path, log_path)
            if not all(path.is_file() for path in required):
                raise RuntimeError(f"campaign incomplete: {directory.name}")
            if locked_dependencies is None:
                if not lock_path.is_file():
                    raise RuntimeError("prospective training dependency lock is missing")
                locked_dependencies = json.loads(lock_path.read_text())["source_sha256"]
                for relative, expected in locked_dependencies.items():
                    source = HERE / relative
                    if not source.is_file() or sha256(source) != expected:
                        raise ValueError(f"locked training dependency changed: {relative}")
            result, identity = json.loads(result_path.read_text()), json.loads(identity_path.read_text())
            if (result.get("status") != "completed" or result.get("arm") != arm
                    or result.get("seed") != seed or identity.get("arm") != arm
                    or identity.get("seed") != seed or result.get("steps") != 1000):
                raise ValueError(f"fit identity/status violates the locked protocol: {directory.name}")
            checkpoint_sha = sha256(checkpoint_path)
            if checkpoint_sha != result["checkpoint_sha256"]:
                raise ValueError(f"checkpoint digest changed: {directory.name}")
            rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
            if [row["step"] for row in rows] != list(range(1, result["steps"] + 1)):
                raise ValueError(f"training trace has missing, duplicate or out-of-order updates: {directory.name}")
            unrelated_changes = []
            for relative, expected in identity["source_sha256"].items():
                source = HERE / relative
                observed = sha256(source) if source.is_file() else None
                if relative in locked_dependencies:
                    if expected != locked_dependencies[relative]:
                        raise ValueError(f"fit did not bind its locked training dependency: {relative}")
                elif observed != expected:
                    # The broad provenance glob also captured independent runtime
                    # modules not imported by training. Their later implementation
                    # changes do not alter an already loaded training dependency.
                    unrelated_changes.append({"path": relative, "captured_sha256": expected,
                                              "current_sha256": observed})
            summary = {key: result[key] for key in (
                "arm", "seed", "steps", "teacher_steps", "batch_worlds", "wall_seconds", "cpu_seconds",
                "input_chars", "output_targets", "decode_input_chars", "physical_token_positions",
                "rehearsal_targets", "batch_trace_sha256", "initial_tensor_sha256")}
            summary.update({"first_training_loss": rows[0]["loss"],
                            "final_training_loss": rows[-1]["loss"],
                            "last_fifty_mean_training_loss": sum(row["loss"] for row in rows[-50:]) / 50,
                            "final_semantic_loss": rows[-1]["semantic_loss"],
                            "final_answer_loss": rows[-1]["answer_loss"],
                            "final_route_loss": rows[-1]["route_loss"],
                            "train_log_sha256": sha256(log_path),
                            "result_sha256": sha256(result_path),
                            "identity_sha256": sha256(identity_path),
                            "training_dependency_lock_verified": True,
                            "training_dependency_lock_sha256": sha256(lock_path),
                            "unrelated_recorded_sources_changed": unrelated_changes,
                            "training_source_sha256": identity["source_sha256"]})
            found.append({"name": directory.name, "checkpoint_path": checkpoint_path,
                          "checkpoint_sha256": checkpoint_sha, "training": summary,
                          "recorded_model_identity": result["final_model_identity"]})
    return found


def evaluate_windows(model: FusionModel, windows: torch.Tensor, *, mode: str, batch_size=8,
                     native_feedback_enabled=True) -> dict:
    if mode not in ("full_original_native", "local_only"):
        raise ValueError("unknown preservation model path")
    started, cpu_started = time.perf_counter(), time.process_time()
    model.eval()
    per_window = []
    with torch.inference_mode():
        for start in range(0, len(windows), batch_size):
            batch = windows[start:start + batch_size]
            inputs, targets = batch[:, :-1], batch[:, 1:]
            # state=None resets every independently sampled window.
            if mode == "full_original_native":
                logits, _ = model.native(inputs, state=None,
                    feedback_enabled=native_feedback_enabled, memory_enabled=True)
            else:
                logits, _ = model.forward_lm(inputs, state=None)
            losses = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
            if not torch.isfinite(losses).all():
                raise FloatingPointError("nonfinite validation loss")
            sums = losses.double().sum(1)
            correct = (logits.argmax(-1) == targets).sum(1)
            for offset, (loss_sum, successes) in enumerate(zip(sums.tolist(), correct.tolist())):
                mean = loss_sum / targets.size(1)
                per_window.append({"window_index": start + offset, "targets": targets.size(1),
                                   "nll_sum": loss_sum, "mean_nll": mean,
                                   "character_perplexity": math.exp(mean),
                                   "next_character_correct": successes})
    total_targets = sum(row["targets"] for row in per_window)
    mean_nll = sum(row["nll_sum"] for row in per_window) / total_targets
    return {"mode": mode, "window_count": len(per_window), "targets": total_targets,
            "mean_nll": mean_nll, "character_perplexity": math.exp(mean_nll),
            "bits_per_character": mean_nll / math.log(2),
            "next_character_accuracy": sum(row["next_character_correct"] for row in per_window) / total_targets,
            "wall_seconds": time.perf_counter() - started, "cpu_seconds": time.process_time() - cpu_started,
            "per_window": per_window}


def run(args):
    # The completion check precedes baseline computation, so this entry point
    # cannot accidentally overlap the still-running fixed training campaign.
    fits = completed_runs(args.runs)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if sha256(args.native) != NATIVE_SHA256:
        raise ValueError("historical native checkpoint is not the admitted parent")
    if sha256(args.validation) != VALIDATION_SHA256:
        raise ValueError("validation token bytes do not match the admitted split")
    validation = np.memmap(args.validation, dtype="<u2", mode="r")
    parent = FusionModel.from_native(args.native, seed=17).eval()
    if len(parent.chars) != 65 or len(validation) != 55770 or int(validation.max()) >= len(parent.chars):
        raise ValueError("validation length or inherited tokenizer compatibility mismatch")
    starts, windows = fixed_windows(validation)
    native_payload = torch.load(args.native, map_location="cpu", weights_only=True)
    native_feedback = native_payload["feedback_enabled"]
    references = {
        "original_connected_cog": evaluate_windows(parent, windows, mode="full_original_native",
                                                   native_feedback_enabled=native_feedback),
        "parent_local_only": evaluate_windows(parent, windows, mode="local_only"),
    }
    parent_loss = references["parent_local_only"]["mean_nll"]
    rows = []
    for fit in fits:
        model = FusionModel.from_checkpoint(fit["checkpoint_path"]).eval()
        if (model.chars != parent.chars or model.native.config != parent.native.config
                or model.admission.get("native_checkpoint_sha256") != NATIVE_SHA256
                or model.model_identity() != fit["recorded_model_identity"]):
            raise ValueError(f"checkpoint/tokenizer/model identity mismatch: {fit['name']}")
        metrics = evaluate_windows(model, windows, mode="local_only")
        rows.append({"name": fit["name"], "checkpoint_sha256": fit["checkpoint_sha256"],
                     "model_identity": model.model_identity(), "training": fit["training"],
                     "metrics": metrics, "preservation_gate": preservation_gate(metrics["mean_nll"], parent_loss)})
        print(json.dumps({"fit": fit["name"], "mean_nll": metrics["mean_nll"],
                          "character_perplexity": metrics["character_perplexity"],
                          "preservation_gate": rows[-1]["preservation_gate"]}), flush=True)
    source_paths = [Path(__file__), Path(__file__).with_name("model.py"),
                    *sorted((Path(__file__).parent / "native").glob("*.py"))]
    output = {
        "schema": "cog-fusion-language-preservation-v1",
        "purpose": "fixed final-model ordinary-language preservation; not model selection or a latency benchmark",
        "validation": {"sha256": VALIDATION_SHA256, "encoding": "little-endian uint16 inherited character IDs",
                       "tokens": len(validation), "vocab_size": len(parent.chars), "chars": list(parent.chars),
                       "window_seed": WINDOW_SEED, "windows": WINDOW_COUNT, "targets_per_window": TARGETS_PER_WINDOW,
                       "total_targets": WINDOW_COUNT * TARGETS_PER_WINDOW, "unique_scored_target_positions": 8192,
                       "start_positions": starts, "sampling": "without replacement from disjoint 128-target blocks",
                       "state_policy": "reset independently for each window in every model",
                       "window_tokens_sha256": hashlib.sha256(windows.numpy().astype("<u2").tobytes()).hexdigest()},
        "native_checkpoint_sha256": NATIVE_SHA256,
        "native_feedback_enabled": native_feedback,
        "runtime": {"torch_version": torch.__version__, "torch_threads": 1, "torch_interop_threads": 1,
                    "evaluation_batch_size": 8, "device": "cpu",
                    "source_sha256": {str(path.relative_to(HERE)): sha256(path) for path in source_paths}},
        "references": references, "runs": rows,
        "all_three_memory_fits_pass_preservation": all(row["preservation_gate"]["passed"]
                                                       for row in rows if row["training"]["arm"] == "memory"),
        "limits": ["sampled TinyShakespeare validation, not a general language benchmark",
                   "original full native and parent local-only references use different computation paths",
                   "perplexity is exp(mean NLL), not the mean of window perplexities",
                   "timings document this check and are not isolated serving-cost estimates"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "parent_local_nll": parent_loss,
                      "original_connected_nll": references["original_connected_cog"]["mean_nll"],
                      "all_three_memory_fits_pass_preservation": output["all_three_memory_fits_pass_preservation"]}), flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=HERE / "evidence" / "runs")
    parser.add_argument("--native", type=Path, default=NATIVE_DEFAULT)
    parser.add_argument("--validation", type=Path, default=VALIDATION_DEFAULT)
    parser.add_argument("--output", type=Path, default=HERE / "evidence" / "LANGUAGE_PRESERVATION.json")
    run(parser.parse_args())
