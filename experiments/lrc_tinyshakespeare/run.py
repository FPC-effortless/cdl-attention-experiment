"""LRC-TS-001: frozen, paired, complete Tiny Shakespeare experiment.

Commands: prepare; train --arm NAME --seed N; gates --seed N; close;
evaluate --arm NAME --seed N; profile; summarize.  No test scores are computed
until ``close`` verifies every backbone and post-training checkpoint.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
import statistics
import subprocess
import time

import torch
from torch import nn
from torch.nn import functional as F

from adapters import (ARMS, BACKBONE_ARMS, build_model, build_gated_model,
                      make_stream, train_step, cache_batch)
from test_integrity import check_causality_and_parity

P = Path(__file__).resolve().parent
OUT = P / "artifacts"
SEEDS = (2201, 2202, 2203)
STEPS = 2048
BATCH = 4
CONTEXT = 128
GATE_STEPS = 600
GATES = ("utility", "utility_shuffled", "utility_random")
EVAL_ARMS = ARMS + GATES
CORPUS_SHA = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def setup():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    OUT.mkdir(parents=True, exist_ok=True)


def data():
    raw = (P / "data/input.txt").read_bytes()
    if hashlib.sha256(raw).hexdigest() != CORPUS_SHA:
        raise RuntimeError("corpus identity mismatch")
    n = len(raw)
    a, b = int(.90 * n), int(.95 * n)
    t = torch.tensor(list(raw), dtype=torch.long) + 1
    return {"train": t[:a], "dev": t[a:b], "test": t[b:]}


def new_batch(tokens, rng):
    starts = [rng.randrange(len(tokens) - CONTEXT) for _ in range(BATCH)]
    z = torch.stack([tokens[i:i + CONTEXT + 1] for i in starts])
    return z[:, :-1], z[:, 1:], starts


def opt_for(model):
    return torch.optim.AdamW(model.parameters(), lr=.002, betas=(.9, .999),
                             eps=1e-8, weight_decay=.01)


def source_files():
    return sorted(p for p in P.rglob("*") if p.is_file()
                  and "artifacts" not in p.relative_to(P).parts
                  and "__pycache__" not in p.relative_to(P).parts
                  and p.suffix in (".py", ".md", ".txt", ".json")
                  and p.name not in ("prefit_manifest.json", "summary.json"))


def verify_sources():
    manifest = json.loads((P / "prefit_manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        if sha(P / name) != expected:
            raise RuntimeError(f"frozen source changed: {name}")
    return sha(P / "prefit_manifest.json")


def checkpoint_path(arm, seed):
    return OUT / "checkpoints" / f"{arm}_{seed}.pt"


def receipt_path(arm, seed):
    return OUT / "fits" / f"{arm}_{seed}.json"


def prepare():
    if list((OUT / "fits").glob("*.json")):
        raise RuntimeError("refusing to replace protocol after fitting")
    ds = data()
    for k, t in ds.items():
        assert len(t) > CONTEXT and bool((t > 0).all())
    manifest = {
        "source_url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "sha256": CORPUS_SHA, "bytes": sum(map(len, ds.values())),
        "train": {"start": 0, "end": 1003854, "bytes": 1003854},
        "dev": {"start": 1003854, "end": 1059624, "bytes": 55770},
        "test": {"start": 1059624, "end": 1115394, "bytes": 55770},
        "test_targets": len(ds["test"]) - 1, "encoding": "byte + 1; zero padding; 257 logits",
        "training_target_exposures_per_backbone": STEPS * BATCH * CONTEXT,
        "sampled_exposure_ratio_to_train_positions": STEPS * BATCH * CONTEXT / (len(ds["train"]) - 1),
        "seeds": SEEDS, "updates": STEPS, "batch": BATCH, "context": CONTEXT,
        "gate_updates": GATE_STEPS, "gate_teacher_target_exposures": GATE_STEPS * BATCH * CONTEXT,
    }
    write_json(P / "data/manifest.json", manifest)
    params = {a: sum(p.numel() for p in build_model(a).parameters()) for a in ARMS}
    assert max(params.values()) / min(params.values()) < 1.05
    env = {"python": platform.python_version(), "torch": str(torch.__version__),
           "platform": platform.platform(), "cuda": torch.cuda.is_available(),
           "cpu_threads_per_worker": 1, "available_cpus": os.cpu_count(),
           "cpu_model": next((l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines()
                              if l.startswith("model name")), "unknown"),
           "cpu_quota": Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
           "memory_limit": Path("/sys/fs/cgroup/memory.max").read_text().strip()}
    write_json(P / "environment.json", env)
    write_json(P / "parameter_counts.json", params)
    hashes = {str(p.relative_to(P)): sha(p) for p in source_files()}
    write_json(P / "prefit_manifest.json", {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "sha256": hashes, "parameters": params, "arms": ARMS, "seeds": SEEDS,
               "updates": STEPS, "gate_updates": GATE_STEPS})
    print(json.dumps({"prepared": True, "arms": len(ARMS), "backbone_fits": len(ARMS) * len(SEEDS),
                      "test_targets": len(ds["test"]) - 1, "parameters": params}), flush=True)


def fit(arm, seed):
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError("arm/seed outside protocol")
    protocol_sha = verify_sources()
    done = receipt_path(arm, seed)
    if done.exists():
        r = json.loads(done.read_text())
        assert r["protocol_sha256"] == protocol_sha and sha(checkpoint_path(arm, seed)) == r["checkpoint_sha256"]
        print("ALREADY_FROZEN", arm, seed, flush=True)
        return
    torch.manual_seed(seed)
    m = build_model(arm)
    optimizer = opt_for(m)
    rng = random.Random(1100000 + seed)
    tokens = data()["train"]
    logs_path = OUT / "logs" / f"{arm}_{seed}.jsonl.gz"
    logs_path.parent.mkdir(parents=True, exist_ok=True)
    batch_hash = hashlib.sha256()
    forwards = 0
    losses = []
    begin, cpu_begin = time.perf_counter(), time.process_time()
    with gzip.open(logs_path, "wt") as log:
        for step in range(STEPS):
            x, y, starts = new_batch(tokens, rng)
            batch_hash.update(json.dumps(starts, separators=(",", ":")).encode())
            receipt = train_step(m, optimizer, x, y, seed=1300000 + seed * 10000 + step)
            receipt.update(step=step + 1, starts=starts)
            assert math.isfinite(receipt["loss"])
            log.write(json.dumps(receipt, allow_nan=False) + "\n")
            forwards += receipt["forward_passes"]
            losses.append(receipt["loss"])
            if (step + 1) % 256 == 0:
                log.flush()
                print(json.dumps({"arm": arm, "seed": seed, "update": step + 1,
                                  "loss_last64": statistics.mean(losses[-64:]),
                                  "wall_seconds": time.perf_counter() - begin}), flush=True)
    wall, cpu = time.perf_counter() - begin, time.process_time() - cpu_begin
    path = checkpoint_path(arm, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": m.state_dict(), "optimizer": optimizer.state_dict(),
                "torch_rng": torch.get_rng_state(), "batch_rng": rng.getstate(),
                "arm": arm, "seed": seed, "updates": STEPS,
                "protocol_sha256": protocol_sha}, path)
    write_json(done, {"arm": arm, "seed": seed, "updates": STEPS,
               "target_bytes": STEPS * BATCH * CONTEXT,
               "forward_passes": forwards, "parameters": sum(p.numel() for p in m.parameters()),
               "wall_seconds_parallel_worker": wall, "process_cpu_seconds": cpu,
               "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               "batch_starts_sha256": batch_hash.hexdigest(), "checkpoint_sha256": sha(path),
               "checkpoint_bytes": path.stat().st_size, "log_sha256": sha(logs_path),
               "train_loss_last64": statistics.mean(losses[-64:]),
               "protocol_sha256": protocol_sha,
               "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print("FIT_CLOSED", arm, seed, wall, flush=True)


def load(arm, seed):
    path = checkpoint_path(arm, seed)
    record = json.loads(receipt_path(arm, seed).read_text())
    if sha(path) != record["checkpoint_sha256"]:
        raise RuntimeError("checkpoint changed")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if arm in GATES:
        base = load("v4", seed)
        m = build_gated_model(base, variant=arm, gate_state=ckpt["gate"])
    else:
        m = build_model(arm)
        m.load_state_dict(ckpt["model"], strict=True)
    return m.eval()


@torch.no_grad()
def utility_teacher(m, x, y):
    captured = []
    hook = m.layers[1].write_relation.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach()))
    try:
        on = m(x, overrides={(1, "read_gate"): torch.ones_like(x)})
        off = m(x, overrides={(1, "read_gate"): torch.zeros_like(x)})
    finally:
        hook.remove()
    assert len(captured) == 2 and torch.equal(captured[0], captured[1])
    a = F.cross_entropy(on.logits.flatten(0, 1), y.flatten(), reduction="none").reshape(y.shape)
    b = F.cross_entropy(off.logits.flatten(0, 1), y.flatten(), reduction="none").reshape(y.shape)
    return captured[0], a, b


def fit_gates(seed):
    protocol_sha = verify_sources()
    if all(receipt_path(a, seed).exists() for a in GATES):
        for a in GATES:
            load(a, seed)
        print("GATES_ALREADY_FROZEN", seed, flush=True)
        return
    if any(receipt_path(a, seed).exists() for a in GATES):
        raise RuntimeError("partial gate cohort; refusing implicit overwrite")
    torch.manual_seed(210000 + seed)
    base = load("v4", seed)
    model = build_gated_model(base)
    heads = {a: nn.Linear(24, 2) for a in GATES}
    for h in heads.values():
        nn.init.zeros_(h.weight)
        nn.init.zeros_(h.bias)
        with torch.no_grad():
            h.bias[1] = -.02
    opts = {a: opt_for(h) for a, h in heads.items()}
    rng = random.Random(7100000 + seed)
    tokens = data()["train"]
    teacher_seconds, teacher_cpu = 0., 0.
    head_seconds = {a: 0. for a in GATES}
    feature_hash = hashlib.sha256()
    logs = []
    begin = time.perf_counter()
    for step in range(GATE_STEPS):
        x, y, starts = new_batch(tokens, rng)
        t, c = time.perf_counter(), time.process_time()
        features, on, off = utility_teacher(model, x, y)
        teacher_seconds += time.perf_counter() - t
        teacher_cpu += time.process_time() - c
        feature_hash.update(features.numpy().tobytes())
        gain = (off - on).clamp(-1, 1).flatten()
        rr = random.Random(7200000 + seed * 10000 + step)
        perm = list(range(gain.numel()))
        rr.shuffle(perm)
        targets = {"utility": gain, "utility_shuffled": gain[torch.tensor(perm)],
                   "utility_random": gain[torch.tensor([rr.randrange(gain.numel()) for _ in perm])]}
        rec = {"step": step + 1, "starts": starts, "mean_signed_gain": float((off - on).mean()),
               "positive_fraction": float(((off - on) > 0).float().mean()), "heads": {}}
        for a, h in heads.items():
            t = time.perf_counter()
            opts[a].zero_grad(set_to_none=True)
            z = h(features).flatten(0, 1)
            loss = F.mse_loss(z[:, 1] - z[:, 0], targets[a])
            loss.backward()
            norm = nn.utils.clip_grad_norm_(h.parameters(), 1.)
            assert torch.isfinite(norm)
            opts[a].step()
            head_seconds[a] += time.perf_counter() - t
            rec["heads"][a] = {"mse": float(loss.detach()), "gradient_norm": float(norm)}
        logs.append(rec)
        if (step + 1) % 100 == 0:
            print("GATE_PROGRESS", seed, step + 1, time.perf_counter() - begin, flush=True)
    # No backbone tensor can be optimized through the detached teacher.
    for name, value in base.state_dict().items():
        assert torch.equal(value, model.state_dict()[name])
    log_path = OUT / "logs" / f"utility_{seed}.jsonl.gz"
    with gzip.open(log_path, "wt") as f:
        for row in logs:
            f.write(json.dumps(row, allow_nan=False) + "\n")
    parent = json.loads(receipt_path("v4", seed).read_text())
    for a, h in heads.items():
        path = checkpoint_path(a, seed)
        torch.save({"gate": h.state_dict(), "optimizer": opts[a].state_dict(),
                    "arm": a, "seed": seed, "updates": GATE_STEPS,
                    "backbone_arm": "v4", "backbone_sha256": parent["checkpoint_sha256"],
                    "protocol_sha256": protocol_sha}, path)
        write_json(receipt_path(a, seed), {
            "arm": a, "seed": seed, "updates": GATE_STEPS,
            "parameters": sum(p.numel() for p in model.parameters()),
            "gate_fitted_parameters": sum(p.numel() for p in h.parameters()),
            "backbone_target_bytes": STEPS * BATCH * CONTEXT,
            "additional_teacher_target_bytes": GATE_STEPS * BATCH * CONTEXT,
            "teacher_forward_passes_shared": 2 * GATE_STEPS,
            "teacher_wall_seconds_shared": teacher_seconds,
            "teacher_process_cpu_seconds_shared": teacher_cpu,
            "head_seconds": head_seconds[a], "feature_sha256": feature_hash.hexdigest(),
            "checkpoint_sha256": sha(path), "checkpoint_bytes": path.stat().st_size,
            "backbone_sha256": parent["checkpoint_sha256"], "protocol_sha256": protocol_sha,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print("GATES_CLOSED", seed, time.perf_counter() - begin, flush=True)


def close():
    protocol = verify_sources()
    records = []
    for seed in SEEDS:
        for arm in EVAL_ARMS:
            rec = json.loads(receipt_path(arm, seed).read_text())
            assert rec["protocol_sha256"] == protocol
            assert sha(checkpoint_path(arm, seed)) == rec["checkpoint_sha256"]
            records.append(rec)
    for seed in SEEDS:
        cohort = [r for r in records if r["seed"] == seed and r["arm"] in ARMS]
        assert len({r["batch_starts_sha256"] for r in cohort}) == 1
    write_json(OUT / "fits_closed.json", {"protocol_sha256": protocol, "count": len(records),
               "closed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "checkpoints": {f'{r["arm"]}_{r["seed"]}': r["checkpoint_sha256"] for r in records}})
    print("ALL_FITS_FROZEN", len(records), flush=True)


@torch.no_grad()
def score(m, tokens, length=128, condition="normal"):
    m.eval()
    total_loss, correct, count = 0., 0, 0
    metrics, rows = {}, []
    begins = list(range(0, len(tokens) - 1, length))
    for offset in range(0, len(begins), 8):
        starts = begins[offset:offset + 8]
        sizes = [min(length, len(tokens) - 1 - s) for s in starts]
        max_n = max(sizes)
        x = torch.zeros((len(starts), max_n), dtype=torch.long)
        y = torch.zeros_like(x)
        for j, (s, n) in enumerate(zip(starts, sizes)):
            x[j, :n] = tokens[s:s + n]
            y[j, :n] = tokens[s + 1:s + n + 1]
        kw = {}
        is_cache = getattr(m, "cached", False)
        if is_cache:
            m.cache_data = cache_batch(x, corrupt=condition == "cache_corrupt")
            if condition == "memory_off":
                kw["force"] = "disabled"
        elif condition == "memory_off":
            kw["memory"] = "disabled"
        if condition == "last_off":
            kw["overrides"] = {(1, "read_gate"): torch.zeros_like(x)}
        out = m(x, **kw)
        if is_cache:
            m.cache_data = None
        losses = F.cross_entropy(out.logits.flatten(0, 1), y.flatten(),
                                 ignore_index=0, reduction="none").reshape(y.shape)
        assert torch.isfinite(losses).all()
        for j, (start, n) in enumerate(zip(starts, sizes)):
            value = float(losses[j, :n].double().sum())
            hit = int((out.logits[j, :n].argmax(-1) == y[j, :n]).sum())
            total_loss += value
            correct += hit
            count += n
            rows.append({"start": start, "targets": n, "sum_nll": value, "correct": hit})
        for k, v in out.metrics.items():
            metrics[k] = metrics.get(k, 0) + v
    assert count == len(tokens) - 1 == 55769
    loss = total_loss / count
    return {"context": length, "condition": condition, "targets": count,
            "nll_nats_per_byte": loss, "bits_per_byte": loss / math.log(2),
            "perplexity": math.exp(loss), "accuracy": correct / count,
            "sum_nll": total_loss, "correct": correct, "metrics": metrics, "blocks": rows}


@torch.no_grad()
def parity_on_validation(m, tokens):
    # Match the timed long-history workload, including Transformer KV truncation.
    x = tokens[:640][None]
    bulk = m(x).logits.log_softmax(-1)
    s = make_stream(m)
    stream = torch.stack([s.step(int(t))[0] for t in x[0]])[None]
    p = make_stream(m)
    first = p.prefill(x[0, :512])
    cont = torch.stack([first[0]] + [p.step(int(t))[0] for t in x[0, 512:]])[None]
    error = float((bulk - stream).abs().max())
    error_p = float((bulk[:, 511:] - cont).abs().max())
    argmax_equal = bool(torch.equal(bulk.argmax(-1), stream.argmax(-1)))
    prefill_argmax_equal = bool(torch.equal(bulk[:, 511:].argmax(-1), cont.argmax(-1)))
    ok = (torch.allclose(bulk, stream, atol=1e-5, rtol=1e-5)
          and torch.allclose(bulk[:, 511:], cont, atol=1e-5, rtol=1e-5)
          and argmax_equal and prefill_argmax_equal)
    assert ok, {"stream_error": error, "prefill_error": error_p}
    return {"stream_max_error": error, "prefill_max_error": error_p,
            "stream_argmax_equal": argmax_equal,
            "prefill_argmax_equal": prefill_argmax_equal,
            "prefix_bytes": 512, "continued_bytes": 128}


@torch.no_grad()
def generation(m, seed):
    prompt = "ROMEO:\n"
    s = make_stream(m)
    z = s.prefill([b + 1 for b in prompt.encode()])
    generator = torch.Generator().manual_seed(300000 + seed)
    output = []
    for _ in range(192):
        z[:, 0] = -float("inf")
        token = int(torch.multinomial((z / .8).softmax(-1), 1, generator=generator))
        output.append(token - 1)
        z = s.step(token)
    return {"prompt": prompt, "temperature": .8,
            "continuation": bytes(output).decode("utf-8", "replace"), "hex": bytes(output).hex()}


def evaluate(arm, seed):
    assert (OUT / "fits_closed.json").exists(), "all fits must close before test"
    verify_sources()
    m = load(arm, seed)
    ds = data()
    checks = check_causality_and_parity(m)
    actual_parity = parity_on_validation(m, ds["dev"])
    results = [score(m, ds["test"], context) for context in (128, 512)]
    if arm not in ("gru", "transformer", "lrc_local"):
        results.append(score(m, ds["test"], 128, "memory_off"))
    if arm in ("lrc_cache", "gru_cache"):
        results.append(score(m, ds["test"], 128, "cache_corrupt"))
    if arm in GATES:
        results.append(score(m, ds["test"], 128, "last_off"))
    output = {"arm": arm, "seed": seed, "checkpoint_sha256": sha(checkpoint_path(arm, seed)),
              "integrity": checks, "validation_parity": actual_parity,
              "scores": results, "sample": generation(m, seed)}
    write_json(OUT / "eval" / f"{arm}_{seed}.json", output)
    print("EVAL_CLOSED", arm, seed, results[0]["bits_per_byte"], flush=True)


@torch.no_grad()
def time_stream(m, stream):
    st = make_stream(m)
    start = time.perf_counter_ns()
    z = st.prefill(stream[:512])
    mid = time.perf_counter_ns()
    for t in stream[512:640]:
        z = st.step(t)
    end = time.perf_counter_ns()
    assert torch.isfinite(z).all()
    return {"prefill_seconds": (mid - start) / 1e9, "decode_seconds": (end - mid) / 1e9,
            "total_seconds": (end - start) / 1e9, "prefill_bytes": 512, "decode_bytes": 128,
            "retained_state_bytes": st.state_bytes()}


def profile():
    assert (OUT / "fits_closed.json").exists()
    verify_sources()
    ds = data()
    stream = ds["dev"][:640].tolist()
    pairs = [(a, s) for s in SEEDS for a in EVAL_ARMS]
    models = {(a, s): load(a, s) for a, s in pairs}
    for m in models.values():
        time_stream(m, stream)
    timing_rows = []
    rng = random.Random(529782)
    for repeat in range(5):
        rng.shuffle(pairs)
        for arm, seed in pairs:
            r = time_stream(models[arm, seed], stream)
            timing_rows.append(dict(arm=arm, seed=seed, repeat=repeat, **r))
        write_json(OUT / "timings.json", timing_rows)
        print("INFERENCE_TIMING_REPEAT", repeat + 1, flush=True)
    # Disposable checkpoint copies; these never replace scored weights.
    training = []
    for arm, seed in pairs:
        if arm not in ARMS:
            continue
        m = copy.deepcopy(models[arm, seed]).requires_grad_(True)
        optimizer = opt_for(m)
        rr = random.Random(182000 + seed)
        for i in range(3):
            x, y, _ = new_batch(ds["train"], rr)
            train_step(m, optimizer, x, y, seed=192000 + i)
        elapsed = []
        for i in range(16):
            x, y, _ = new_batch(ds["train"], rr)
            start = time.perf_counter_ns()
            train_step(m, optimizer, x, y, seed=193000 + i)
            elapsed.append((time.perf_counter_ns() - start) / 1e9)
        training.append({"arm": arm, "seed": seed, "step_seconds": elapsed,
                         "median_seconds": statistics.median(elapsed),
                         "target_bytes_per_step": BATCH * CONTEXT,
                         "targets_per_second": BATCH * CONTEXT / statistics.median(elapsed)})
        write_json(OUT / "training_speed.json", training)
    # PyTorch reports only recognized FLOP-bearing operators. Never label these totals complete.
    flops = []
    from torch.profiler import profile as torch_profile, ProfilerActivity
    x = ds["train"][:128][None]
    for arm in EVAL_ARMS:
        m = models[arm, SEEDS[0]]
        with torch.no_grad(), torch_profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
            m(x)
        events = [{"operator": e.key, "flops": e.flops} for e in prof.key_averages() if e.flops]
        flops.append({"arm": arm, "input_batch": 1, "input_bytes": 128,
                      "counted_operator_flops": sum(e["flops"] for e in events),
                      "coverage": "partial; excludes unsupported operators, Python/indexing and memory traffic",
                      "operators": events})
    write_json(OUT / "partial_flops.json", flops)
    print("PROFILING_CLOSED", flush=True)


def summary():
    rows, all_scores = [], []
    for a in EVAL_ARMS:
        records = [json.loads((OUT / "eval" / f"{a}_{s}.json").read_text()) for s in SEEDS]
        primary = [r["scores"][0] for r in records]
        row = {"arm": a, "parameters": json.loads(receipt_path(a, SEEDS[0]).read_text())["parameters"],
               "seeds": list(SEEDS)}
        for key in ("nll_nats_per_byte", "bits_per_byte", "perplexity", "accuracy"):
            values = [p[key] for p in primary]
            row[key] = {"mean": statistics.mean(values), "std": statistics.stdev(values), "by_seed": values}
        long = [next(p for p in r["scores"] if p["context"] == 512) for r in records]
        row["context512_bpb"] = {"mean": statistics.mean(p["bits_per_byte"] for p in long),
                                 "by_seed": [p["bits_per_byte"] for p in long]}
        for condition in ("memory_off", "cache_corrupt", "last_off"):
            controls = [next((p for p in r["scores"] if p["condition"] == condition), None) for r in records]
            if all(v is not None for v in controls):
                delta = [c["bits_per_byte"] - p["bits_per_byte"] for c, p in zip(controls, primary)]
                row[condition + "_delta_bpb"] = {"mean": statistics.mean(delta), "std": statistics.stdev(delta), "by_seed": delta}
        for r in records:
            all_scores += [{"arm": a, "seed": r["seed"], **{k: v for k, v in p.items() if k != "blocks"}} for p in r["scores"]]
        rows.append(row)
    timings = json.loads((OUT / "timings.json").read_text()) if (OUT / "timings.json").exists() else []
    speeds = json.loads((OUT / "training_speed.json").read_text()) if (OUT / "training_speed.json").exists() else []
    counts = json.loads((OUT / "partial_flops.json").read_text()) if (OUT / "partial_flops.json").exists() else []
    for row in rows:
        arm = row["arm"]
        times = [t for t in timings if t["arm"] == arm]
        if times:
            for key in ("prefill_seconds", "decode_seconds", "total_seconds", "retained_state_bytes"):
                row[key + "_median"] = statistics.median(t[key] for t in times)
            row["decode_bytes_per_second"] = 128 / row["decode_seconds_median"]
        ts = [t for t in speeds if t["arm"] == arm]
        if ts:
            row["isolated_train_bytes_per_second"] = statistics.median(t["targets_per_second"] for t in ts)
            row["isolated_train_step_ms"] = 1000 * statistics.median(t["median_seconds"] for t in ts)
        c = next((r for r in counts if r["arm"] == arm), None)
        if c:
            row["partial_flops_per_byte"] = c["counted_operator_flops"] / 128
    rows.sort(key=lambda r: r["bits_per_byte"]["mean"])
    write_json(OUT / "summary.json", {"experiment": "LRC-TS-001", "results": rows,
               "all_scores": all_scores, "test_targets": 55769,
               "protocol_sha256": sha(P / "prefit_manifest.json")})
    print(json.dumps([{k: r[k] for k in ("arm", "parameters", "bits_per_byte", "accuracy")} for r in rows], indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "train", "gates", "close", "evaluate", "profile", "summarize"))
    parser.add_argument("--arm")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    setup()
    if args.command == "prepare": prepare()
    elif args.command == "train": fit(args.arm, args.seed)
    elif args.command == "gates": fit_gates(args.seed)
    elif args.command == "close": close()
    elif args.command == "evaluate": evaluate(args.arm, args.seed)
    elif args.command == "profile": profile()
    elif args.command == "summarize": summary()


if __name__ == "__main__":
    main()
