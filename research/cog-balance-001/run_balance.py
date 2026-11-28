"""Train one architecture with fixed data and fixed CPU-time readouts."""
import argparse
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from corpus import load_corpus
from panels import evaluate, train_batch, train_prose_batch
from profiler import profile_model
from scale import build_model


def write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


@torch.no_grad()
def assess(model, data, worlds, dev_batches):
    model.eval()
    rng = np.random.default_rng(781299)
    losses = []
    digest = hashlib.sha256()
    for _ in range(dev_batches):
        xy = torch.from_numpy(data.batch("dev", rng, 8, 128))
        digest.update(xy.numpy().tobytes())
        logits = model(xy[:, :-1])
        losses.append(F.cross_entropy(logits.flatten(0, 1), xy[:, 1:].flatten()).item())
    result = {
        "dev_bpb": float(np.mean(losses) / math.log(2)),
        "dev_byte_targets": dev_batches * 1024, "dev_sha256": digest.hexdigest(),
        "panels": evaluate(model, 610003, worlds=worlds),
        "profile": profile_model(model, xy),
    }
    if hasattr(model, "measure_routes"):
        model.measure_routes(True)
        model(xy[:, :-1])
        result["route_counts_by_layer_on_last_text_batch"] = [
            {key: int(value) for key, value in row.items()} for row in model.routing_metrics()
        ]
        model.measure_routes(False)
    model.train()
    return result


def run(args):
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    data = load_corpus(args.corpus)
    model = build_model(args.arm, args.seed, args.scale)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.95), weight_decay=0.01)
    args.out.mkdir(exist_ok=True, parents=True)
    rng = np.random.default_rng(args.seed + 83021)
    result = {
        "arm": args.arm, "seed": args.seed, "scale": args.scale, "parameters": params,
        "cpu": platform.processor(), "torch": str(torch.__version__), "threads": args.threads,
        "budgets": {k: getattr(args, k) for k in ("steps", "cpu_seconds", "batch", "price", "worlds", "dev_batches")},
        "corpus": data.manifest, "source_commit": os.getenv("SOURCE_SHA"), "readouts": {},
    }
    cpu = wall = 0.0
    data_hash = hashlib.sha256()
    step = 0
    while step < args.steps or "matched_cpu" not in result["readouts"]:
        start_cpu, start_wall = time.process_time(), time.perf_counter()
        xy = torch.from_numpy(data.batch("train", rng, args.batch, 128))
        draw = train_batch if step % 2 == 0 else train_prose_batch
        ids, labels = draw(args.seed, step, args.batch)
        opt.zero_grad(set_to_none=True)
        # Normalize each objective independently across targets.
        for inp, target, weight in ((xy[:, :-1], xy[:, 1:], 1.0), (ids, labels, 0.5)):
            output = model(inp)
            loss = weight * F.cross_entropy(output.flatten(0, 1), target.flatten(), ignore_index=-100)
            if hasattr(model, "routing_cost"):
                loss = loss + 0.5 * args.price * model.routing_cost()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite loss")
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        step += 1
        cpu += time.process_time() - start_cpu
        wall += time.perf_counter() - start_wall
        for tensor in (xy, ids, labels):
            data_hash.update(tensor.numpy().tobytes())
        readouts = []
        if cpu >= args.cpu_seconds and "matched_cpu" not in result["readouts"]:
            readouts.append("matched_cpu")
        if step == args.steps:
            readouts.append("matched_data")
        if readouts:
            record = assess(model, data, args.worlds, args.dev_batches)
            record.update(step=step, train_cpu_seconds=cpu, train_wall_seconds=wall,
                          training_text_targets=step * args.batch * 128,
                          training_questions=step * args.batch, training_sha256=data_hash.hexdigest(),
                          train_language_bytes_per_wall_second=step * args.batch * 128 / wall)
            if record["profile"]["batch_vs_stream_max_abs_logit_error"] > 0.002:
                raise ValueError("post-training streaming parity failed")
            for key in readouts:
                result["readouts"][key] = record
                print(json.dumps({"readout": key, "step": step, "arm": args.arm,
                                  "seed": args.seed, "scale": args.scale, "dev_bpb": record["dev_bpb"],
                                  "retrieval_pairs": record["panels"]["retrieval"]["both_correct"]}), flush=True)
            write_json(args.out / "metrics.json", result)
        if step == args.steps:
            torch.save(model.state_dict(), args.out / "matched_data_weights.pt")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--scale", choices=["1m", "10m"], default="1m")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=384)
    parser.add_argument("--cpu-seconds", type=float, default=60)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--price", type=float, default=0.01)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--worlds", type=int, default=128)
    parser.add_argument("--dev-batches", type=int, default=32)
    args = parser.parse_args()
    if min(args.steps, args.cpu_seconds, args.threads, args.batch, args.worlds, args.dev_batches) <= 0:
        parser.error("budgets and batch sizes must be positive")
    run(args)
