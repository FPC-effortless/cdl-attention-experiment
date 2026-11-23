"""Reproducible exploratory Tiny Shakespeare training; no architecture-ranking claim."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import signal
import time

import torch
import torch.nn.functional as F

from data import CharTokenizer, load_split, prepare, sha256
from model import COGTinyLM, ModelConfig

HERE = Path(__file__).resolve().parent
DEFAULT_CORPUS_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"


def atomic_json(path: Path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_checkpoint(path: Path, payload):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def append_json(path: Path, value):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False) + "\n")
        f.flush()


def source_hashes():
    return {name: sha256(HERE / name) for name in ("model.py", "train.py", "data.py")}


def cpu_weights(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def amp_context(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def get_batch(tokens, batch_size, context_length, generator, device):
    # randint upper bound is exclusive: last possible x ends at len-2 and y at len-1.
    high = len(tokens) - context_length
    if high <= 0:
        raise ValueError("Split shorter than context plus one target")
    starts = torch.randint(high, (batch_size,), generator=generator)
    offsets = torch.arange(context_length)
    ix = starts[:, None] + offsets[None, :]
    return tokens[ix].to(device), tokens[ix + 1].to(device)


def unigram_result(train_tokens, evaluation_tokens, vocab_size):
    counts = torch.bincount(train_tokens, minlength=vocab_size).double()
    if (counts == 0).any():
        raise ValueError("Train-only vocabulary unexpectedly has an unobserved character")
    log_prob = (counts / counts.sum()).log()
    targets = evaluation_tokens[1:]
    nll = -log_prob[targets].mean().item()
    return {
        "method": "maximum-likelihood character frequencies from training text only",
        "mean_nll": nll, "character_perplexity": math.exp(nll),
        "bits_per_character": nll / math.log(2),
        "next_character_accuracy": (targets == counts.argmax()).double().mean().item(),
        "targets": len(targets),
    }


@torch.inference_mode()
def evaluate(model, tokens, context_length, batch_size, device, precision="fp32"):
    """Score each next-character target once, with reset context at each window."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_targets = 0
    start_time = time.perf_counter()
    starts = list(range(0, len(tokens) - 1, context_length))
    for p in range(0, len(starts), batch_size):
        group = starts[p:p + batch_size]
        lengths = [min(context_length, len(tokens) - 1 - s) for s in group]
        width = max(lengths)
        x = torch.zeros((len(group), width), dtype=torch.long)
        y = torch.zeros_like(x)
        mask = torch.zeros_like(x, dtype=torch.bool)
        for j, (s, length) in enumerate(zip(group, lengths)):
            x[j, :length] = tokens[s:s + length]
            y[j, :length] = tokens[s + 1:s + length + 1]
            mask[j, :length] = True
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        with amp_context(device, precision):
            logits, _ = model(x)
        losses = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none").view_as(y)
        total_loss += losses.masked_select(mask).double().sum().item()
        total_correct += ((logits.argmax(-1) == y) & mask).sum().item()
        total_targets += int(mask.sum().item())
    model.train(was_training)
    nll = total_loss / total_targets
    return {
        "mean_nll": nll,
        "character_perplexity": math.exp(nll),
        "bits_per_character": nll / math.log(2),
        "next_character_accuracy": total_correct / total_targets,
        "targets": total_targets,
        "seconds": time.perf_counter() - start_time,
        "context_policy": "reset each nonoverlapping context-length window; all targets once",
    }


def learning_rate(step, cfg):
    """Step is zero based; scheduling has no mutable hidden state."""
    if step < cfg["warmup_steps"]:
        return cfg["learning_rate"] * (step + 1) / max(1, cfg["warmup_steps"])
    progress = (step - cfg["warmup_steps"]) / max(1, cfg["max_steps"] - cfg["warmup_steps"] - 1)
    progress = min(1.0, max(0.0, progress))
    ratio = cfg["min_lr_ratio"] + (1 - cfg["min_lr_ratio"]) * 0.5 * (1 + math.cos(math.pi * progress))
    return cfg["learning_rate"] * ratio


def capture_rng(generator):
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "batch_sampler": generator.get_state(),
    }


def restore_rng(payload, generator):
    random.setstate(payload["python"])
    torch.set_rng_state(payload["torch"].cpu())
    if payload["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["cuda"])
    generator.set_state(payload["batch_sampler"].cpu())


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=HERE / "data")
    p.add_argument("--out-dir", type=Path, default=HERE / "runs" / "tiny10m_seed20261122")
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    p.add_argument("--seed", type=int, default=20261122)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--gradient-accumulation", type=int, default=2)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--min-lr-ratio", type=float, default=0.03)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--minimum-steps", type=int, default=1000)
    p.add_argument("--minimum-improvement", type=float, default=0.001)
    p.add_argument("--max-wall-seconds", type=float, default=0)
    p.add_argument("--resume", type=Path)
    p.add_argument("--model-config", type=Path)
    p.add_argument("--evaluate-only", type=Path, help="Evaluate one chosen checkpoint on val and final test without fitting.")
    p.add_argument("--allow-parameter-count", action="store_true", help="Development-only: allow outside9M–11M, recorded in manifest.")
    return p.parse_args()


def main():
    args = parse_args()
    for name in ("batch_size", "gradient_accumulation", "context_length", "max_steps", "eval_every", "eval_batch_size", "log_every", "threads"):
        if getattr(args, name) <= 0:
            raise ValueError(name + " must be positive")
    if args.max_wall_seconds < 0:
        raise ValueError("max-wall-seconds must be nonnegative")
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = choose_device(args.device)
    if args.precision != "fp32" and device.type != "cuda":
        raise ValueError("Use fp32 on CPU; mixed precision here is qualified only for CUDA.")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("CUDA device does not support bfloat16")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if not (args.data_dir / "dataset.json").exists():
        prepare(args.data_dir, DEFAULT_CORPUS_SHA256)
    meta = json.loads((args.data_dir / "dataset.json").read_text(encoding="utf-8"))
    if meta["input_sha256"] != DEFAULT_CORPUS_SHA256:
        raise ValueError("This run requires the pinned Tiny Shakespeare corpus")
    tokenizer = CharTokenizer.load(args.data_dir / "tokenizer.json")
    if sha256(args.data_dir / "tokenizer.json") != meta["tokenizer_sha256"]:
        raise ValueError("Tokenizer hash mismatch")
    val_tokens = load_split(args.data_dir, "val")

    if args.evaluate_only:
        ckpt = torch.load(args.evaluate_only, map_location="cpu", weights_only=True)
        if ckpt["dataset_sha256"] != meta["input_sha256"]:
            raise ValueError("Checkpoint data mismatch")
        model = COGTinyLM(ModelConfig(**ckpt["model_config"])).to(device)
        model.load_state_dict(ckpt["model"])
        results = {}
        for split in ("val", "test"):
            tokens = val_tokens if split == "val" else load_split(args.data_dir, "test")
            results[split] = evaluate(model, tokens, args.context_length, args.eval_batch_size, device, args.precision)
        print(json.dumps({"checkpoint": str(args.evaluate_only), "results": results}, indent=2))
        return

    train_tokens = load_split(args.data_dir, "train")
    model_values = json.loads(args.model_config.read_text()) if args.model_config else {}
    model_values["vocab_size"] = len(tokenizer.chars)
    config = ModelConfig(**model_values)
    model = COGTinyLM(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if not args.allow_parameter_count and not 9_000_000 <= n_params <= 11_000_000:
        raise ValueError(f"Expected approximately10M active parameters, got {n_params:,}")
    cfg = {
        name: getattr(args, name) for name in (
            "seed", "batch_size", "gradient_accumulation", "context_length", "max_steps",
            "learning_rate", "min_lr_ratio", "warmup_steps", "weight_decay", "grad_clip",
            "eval_every", "patience", "minimum_steps", "minimum_improvement", "precision",
        )
    }
    cfg.update(betas=[0.9, 0.95], optimizer="AdamW", train_state_policy="reset for each sampled window",
               effective_targets_per_step=args.batch_size * args.gradient_accumulation * args.context_length)
    groups = [
        {"params": [p for p in model.parameters() if p.requires_grad and p.ndim >= 2], "weight_decay": args.weight_decay},
        {"params": [p for p in model.parameters() if p.requires_grad and p.ndim < 2], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=args.learning_rate, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16", init_scale=4096.0)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 100)
    step = 0
    attempted_updates = 0
    skipped_updates = 0
    total_targets = 0
    total_train_seconds = 0.0
    best_val = float("inf")
    best_step = 0
    bad_evals = 0
    sources = source_hashes()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "run_config.json"
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        if ckpt.get("checkpoint_status") != "resumable":
            raise ValueError("Checkpoint is not marked as a completed-step resumable state")
        if ckpt["training_config"] != cfg or ckpt["model_config"] != asdict(config):
            raise ValueError("Resume configuration differs; use identical training and model settings")
        if ckpt["dataset_sha256"] != meta["input_sha256"] or ckpt["source_sha256"] != sources:
            raise ValueError("Resume data/source differs; refusing an unrecorded change")
        if ckpt["tokenizer_sha256"] != meta["tokenizer_sha256"]:
            raise ValueError("Resume tokenizer differs")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        # torch optimizer normally remaps state on load; make ownership explicit.
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and key != "step":
                    state[key] = value.to(device)
        scaler.load_state_dict(ckpt["scaler"])
        step, total_targets = ckpt["step"], ckpt["training_targets_seen"]
        attempted_updates, skipped_updates = ckpt["attempted_updates"], ckpt["skipped_updates"]
        total_train_seconds = ckpt["training_seconds"]
        best_val, best_step, bad_evals = ckpt["best_val_nll"], ckpt["best_step"], ckpt["bad_evals"]
        restore_rng(ckpt["rng"], generator)
        best_path = args.out_dir / "best.pt"
        if not best_path.exists():
            raise ValueError("Resume directory must preserve best.pt along with the resume checkpoint")
        if sha256(best_path) != ckpt["best_checkpoint_sha256"]:
            raise ValueError("best.pt and resume checkpoint are not a matching snapshot pair; restore matching files")
        best_payload = torch.load(best_path, map_location="cpu", weights_only=True)
        expected_best_metadata = {
            "dataset_sha256": meta["input_sha256"],
            "tokenizer_sha256": meta["tokenizer_sha256"],
            "source_sha256": sources, "model_config": asdict(config), "step": best_step,
        }
        if any(best_payload.get(k) != v for k, v in expected_best_metadata.items()):
            raise ValueError("best.pt metadata is inconsistent with this resumable run")
        if best_step > step or best_payload["val"]["mean_nll"] != best_val:
            raise ValueError("best.pt selection step/loss is inconsistent with the resume checkpoint")
        del best_payload
    elif manifest_path.exists():
        raise ValueError("Output directory already contains a run; choose a new directory or --resume")

    manifest = {
        "study": "COG-UNIFIED-TINY-10M-001",
        "scope": "single-seed exploratory character language model for user testing; no attention superiority or typed-semantic qualification",
        "model_config": asdict(config),
        "parameters": n_params,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "training_config": cfg,
        "dataset": meta,
        "source_sha256": sources,
        "training_authorized_by": "user request to train roughly10M model on Tiny Shakespeare",
        "parameter_range_override": args.allow_parameter_count,
        "early_stopping": "validation only; patience evaluations with improvement threshold, gated by minimum steps",
        "final_test_policy": "evaluate best validation checkpoint only after completed budget or declared early stopping",
        "unigram_validation": unigram_result(train_tokens, val_tokens, len(tokenizer.chars)),
    }
    if not args.resume:
        atomic_json(manifest_path, manifest)
        atomic_json(args.out_dir / "tokenizer.json", json.loads((args.data_dir / "tokenizer.json").read_text()))
    environment = {
        "python": platform.python_version(), "torch": str(torch.__version__),
        "device": str(device), "threads": torch.get_num_threads(),
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "resume_step": step, "max_wall_seconds": args.max_wall_seconds,
        "wall_start_unix": time.time(),
    }
    append_json(args.out_dir / "sessions.jsonl", environment)
    print(json.dumps({"event": "start", "parameters": n_params, "config": cfg, "environment": environment}), flush=True)
    interrupted = {"signal": None}
    def handle_signal(signum, frame):
        interrupted["signal"] = signum
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    wall_start = time.perf_counter()

    def save_state(filename="resume.pt", checkpoint_status="resumable"):
        payload = {
            "format_version": 1, "model": cpu_weights(model), "model_config": asdict(config),
            "checkpoint_status": checkpoint_status,
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "training_config": cfg, "step": step, "training_targets_seen": total_targets,
            "attempted_updates": attempted_updates, "skipped_updates": skipped_updates,
            "training_seconds": total_train_seconds, "best_val_nll": best_val,
            "best_step": best_step, "bad_evals": bad_evals, "rng": capture_rng(generator),
            "source_sha256": sources, "dataset_sha256": meta["input_sha256"],
            "tokenizer_sha256": meta["tokenizer_sha256"],
            "best_checkpoint_sha256": sha256(args.out_dir / "best.pt"),
        }
        atomic_checkpoint(args.out_dir / filename, payload)

    def run_validation():
        nonlocal best_val, best_step, bad_evals
        result = evaluate(model, val_tokens, args.context_length, args.eval_batch_size, device, args.precision)
        improved = result["mean_nll"] < best_val - args.minimum_improvement
        # Persist the strict numerical best even if improvement does not reset patience.
        if result["mean_nll"] < best_val:
            best_val, best_step = result["mean_nll"], step
            atomic_checkpoint(args.out_dir / "best.pt", {
                "format_version": 1, "model": cpu_weights(model), "model_config": asdict(config),
                "step": step, "val": result, "training_targets_seen": total_targets,
                "source_sha256": sources, "dataset_sha256": meta["input_sha256"],
                "tokenizer_sha256": meta["tokenizer_sha256"],
                "tokenizer": json.loads((args.data_dir / "tokenizer.json").read_text()),
            })
        bad_evals = 0 if improved else bad_evals + 1
        event = {"event": "validation", "step": step, "targets_seen": total_targets,
                 "best_step": best_step, "bad_evals": bad_evals, **result}
        append_json(args.out_dir / "metrics.jsonl", event)
        print(json.dumps(event), flush=True)
        save_state()
        return result

    if step == 0 and not args.resume:
        initial_val = run_validation()
        atomic_json(args.out_dir / "initial_evaluation.json", {"step": 0, "val": initial_val})
    model.train()
    stop_reason = "max_steps"
    failed = False
    try:
        while step < args.max_steps:
            if interrupted["signal"] is not None:
                stop_reason = "signal_" + str(interrupted["signal"])
                break
            if args.max_wall_seconds and time.perf_counter() - wall_start >= args.max_wall_seconds:
                stop_reason = "wall_time_limit"
                break
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            tick = time.perf_counter()
            lr = learning_rate(step, cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            attempted_updates += 1
            for _ in range(args.gradient_accumulation):
                x, y = get_batch(train_tokens, args.batch_size, args.context_length, generator, device)
                with amp_context(device, args.precision):
                    logits, _ = model(x)
                    loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1))
                total_targets += args.batch_size * args.context_length
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                loss_sum += loss.detach().item()
                scaler.scale(loss / args.gradient_accumulation).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if not torch.isfinite(grad_norm):
                if args.precision != "fp16":
                    raise FloatingPointError("Non-finite gradient norm")
                # GradScaler observes overflow, skips this optimizer update, and lowers scale.
                scaler.step(optimizer)
                scaler.update()
                skipped_updates += 1
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                total_train_seconds += time.perf_counter() - tick
                event = {"event": "fp16_overflow_skipped", "step": step,
                         "attempted_updates": attempted_updates, "skipped_updates": skipped_updates,
                         "targets_seen": total_targets, "new_scale": scaler.get_scale()}
                append_json(args.out_dir / "metrics.jsonl", event)
                print(json.dumps(event), flush=True)
                if skipped_updates >= 20:
                    raise FloatingPointError("Twenty fp16 overflow skips; stop for numerical diagnosis")
                continue
            scaler.step(optimizer)
            scaler.update()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step += 1
            duration = time.perf_counter() - tick
            total_train_seconds += duration
            if step == 1 or step % args.log_every == 0:
                event = {
                    "event": "train", "step": step,
                    "train_loss": loss_sum / args.gradient_accumulation,
                    "learning_rate": lr, "gradient_norm_before_clip": float(grad_norm),
                    "targets_seen": total_targets,
                    "attempted_updates": attempted_updates, "skipped_updates": skipped_updates,
                    "train_target_equivalent_passes": total_targets / (len(train_tokens) - 1),
                    "step_seconds": duration,
                    "targets_per_second": cfg["effective_targets_per_step"] / duration,
                    "training_seconds": total_train_seconds,
                }
                append_json(args.out_dir / "metrics.jsonl", event)
                print(json.dumps(event), flush=True)
            if step % args.eval_every == 0 or step == args.max_steps:
                run_validation()
                if args.patience and step >= args.minimum_steps and bad_evals >= args.patience:
                    stop_reason = "validation_early_stopping"
                    break
    except BaseException as exc:
        failed = True
        # A failure is recorded distinctly; the current checkpoint is for diagnosis.
        atomic_json(args.out_dir / "failure.json", {
            "step": step, "exception": type(exc).__name__, "message": str(exc),
            "training_targets_seen": total_targets, "attempted_updates": attempted_updates,
            "skipped_updates": skipped_updates,
        })
        raise
    finally:
        if failed:
            save_state("failure_state.pt", checkpoint_status="failed_diagnostic_only")
        else:
            save_state()

    completed = stop_reason in ("max_steps", "validation_early_stopping")
    if completed and step % args.eval_every != 0 and step != args.max_steps:
        run_validation()
    outcome = {
        "status": "completed" if completed else "paused",
        "stop_reason": stop_reason, "step": step, "planned_max_steps": args.max_steps,
        "parameters": n_params, "training_targets_seen": total_targets,
        "attempted_updates": attempted_updates, "skipped_updates": skipped_updates,
        "train_target_equivalent_passes": total_targets / (len(train_tokens)-1),
        "training_seconds": total_train_seconds,
        "session_wall_seconds": time.perf_counter() - wall_start,
        "best_step": best_step, "best_val_nll": best_val,
        "test_evaluated": False,
        "scope": manifest["scope"],
        "unigram_validation": manifest["unigram_validation"],
        "initial_evaluation": json.loads((args.out_dir / "initial_evaluation.json").read_text()),
    }
    if completed:
        best = torch.load(args.out_dir / "best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(best["model"])
        test_tokens = load_split(args.data_dir, "test")
        outcome["test"] = evaluate(model, test_tokens, args.context_length, args.eval_batch_size, device, args.precision)
        outcome["unigram_test"] = unigram_result(train_tokens, test_tokens, len(tokenizer.chars))
        outcome["test_evaluated"] = True
        outcome["best_checkpoint_sha256"] = sha256(args.out_dir / "best.pt")
        outcome["best_checkpoint_training_targets"] = best["training_targets_seen"]
    atomic_json(args.out_dir / "result.json", outcome)
    print(json.dumps({"event": "finish", **outcome}), flush=True)


if __name__ == "__main__":
    main()

