#!/usr/bin/env python3
"""One-host, untrained reference-decoder cost; no training or quality claims."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT))
import torch

from nanocog.model import Config, Decoder, LocalOnly
from nanocog.interaction import PacketMixer, PacketAndTypedProbe
from nanocog.memory.codec import encode, decode
from nanocog.memory.decoder import PersistentDecoder, _model_identity
from nanocog.memory.store import MemoryStore

SEED = 91023
KINDS = {"local": LocalOnly, "packet": PacketMixer, "typed_packet": PacketAndTypedProbe}
TOKENIZER = "synthetic-byte-256-v1"
SESSION = "cost-stream"


def model_for(kind):
    torch.manual_seed(SEED)
    return Decoder(Config(256, 160, 4), KINDS[kind]).eval()


def inputs():
    g = torch.Generator().manual_seed(SEED + 1)
    return torch.randint(0, 256, (1, 1040), generator=g)


def timed(call):
    begin = time.perf_counter_ns()
    result = call()
    return (time.perf_counter_ns() - begin) / 1e6, result


def describe(samples):
    ordered = sorted(samples)
    return {"median_ms": statistics.median(samples),
            "p95_ms": ordered[math.ceil(0.95 * len(samples)) - 1],
            "samples_ms": samples}


def wire_sizes(value):
    wire = encode(value)
    raw = 0
    def count(node):
        nonlocal raw
        if isinstance(node, dict):
            if node.get("type") == "tensor":
                raw += len(base64.b64decode(node["data"]))
            for v in node.values():
                count(v)
        elif isinstance(node, list):
            for v in node:
                count(v)
    count(wire)
    return {"raw_tensor_bytes": raw,
            "encoded_json_bytes": len(json.dumps(wire, separators=(",", ":")).encode())}


def check(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    return float((actual - expected).abs().max())


def child(args):
    model = model_for(args.kind)
    tail = inputs()[:, args.history:args.history + args.chunk]
    with MemoryStore(args.db, principal_id="cost", namespace="validation") as store:
        runner = PersistentDecoder(model, store, session_id=SESSION, tokenizer_id=TOKENIZER)
        duration, result = timed(lambda: runner.run(tail, request_id="cold-tail"))
        print(json.dumps({"inner_run_ms": duration, "revision": result.revision,
                          "tokens_seen": result.tokens_seen,
                          "logits": result.logits.tolist()}))


def run(args):
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    harness_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    files = ("model.py", "interaction.py", "comm/packet.py", "state/records.py",
             "memory/store.py", "memory/decoder.py", "memory/codec.py")
    source_hashes = {f: hashlib.sha256((PACKAGE_ROOT / "nanocog" / f).read_bytes()).hexdigest()
                     for f in files}
    output = {"study": "MEMORY-VALIDATION-002", "kind": "untrained_reference_cost",
              "seed": SEED, "warmup_trials": 1, "measured_trials": 5,
              "tolerance": {"rtol": 1e-5, "atol": 1e-6},
              "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                              "sqlite": sqlite3.sqlite_version, "platform": platform.platform(),
                              "machine": platform.machine(), "cpu_threads": torch.get_num_threads(),
                              "visible_cpus": os.cpu_count()},
              "source_sha256": source_hashes, "harness_sha256": harness_hash,
              "models": [], "cases": [], "cold_processes": []}
    tokens = inputs()
    with tempfile.TemporaryDirectory(prefix="nanocog-cost-", dir=args.work_dir) as scratch:
        folder = Path(scratch)
        output["filesystem"] = subprocess.run(["df", "-PT", str(folder)], text=True,
                                                 capture_output=True, check=True).stdout.strip()
        for kind in KINDS:
            model = model_for(kind)
            identity_samples = [timed(lambda: _model_identity(model, TOKENIZER, SESSION))[0]
                                for _ in range(6)][1:]
            model_row = {"kind": kind, "parameters": sum(p.numel() for p in model.parameters()),
                         "standalone_identity": describe(identity_samples)}
            output["models"].append(model_row)
            for history in (128, 1024):
                prefix = tokens[:, :history]
                with torch.inference_mode():
                    _, prefix_state = model(prefix)
                base_path = folder / f"{kind}-{history}-base.sqlite3"
                with MemoryStore(base_path, principal_id="cost", namespace="validation") as store:
                    runner = PersistentDecoder(model, store, session_id=SESSION, tokenizer_id=TOKENIZER)
                    prefix_receipt = runner.run(prefix, request_id="prefix")
                    assert prefix_receipt.tokens_seen == history
                # No connection remains open. SQLite has checkpointed its WAL;
                # fixture copies are outside timings and are not backup API tests.
                assert not Path(str(base_path) + "-wal").exists()
                codec_samples = [timed(lambda: decode(encode(prefix_state)))[0]
                                 for _ in range(6)][1:]
                for chunk in (1, 16):
                    tail = tokens[:, history:history + chunk]
                    full = tokens[:, :history + chunk]
                    with torch.inference_mode():
                        expected = model(full)[0][:, history:]
                    samples = {name: [] for name in ("recompute_prefix", "ram_incremental",
                                                       "persistent_commit", "exact_retry")}
                    errors = {name: 0.0 for name in samples}
                    for trial in range(6):
                        trial_path = folder / f"{kind}-{history}-{chunk}-{trial}.sqlite3"
                        shutil.copyfile(base_path, trial_path)
                        with MemoryStore(trial_path, principal_id="cost", namespace="validation") as store:
                            runner = PersistentDecoder(model, store, session_id=SESSION, tokenizer_id=TOKENIZER)
                            with torch.inference_mode():
                                # Alternate direct-path order to reduce systematic warm-cache bias.
                                paths = ("ram_incremental", "recompute_prefix") if trial % 2 else ("recompute_prefix", "ram_incremental")
                                for name in paths:
                                    call = (lambda: model(full)[0][:, history:]) if name == "recompute_prefix" else (lambda: model(tail, prefix_state)[0])
                                    duration, logits = timed(call)
                                    errors[name] = max(errors[name], check(logits, expected))
                                    if trial:
                                        samples[name].append(duration)
                            duration, receipt = timed(lambda: runner.run(tail, request_id="tail"))
                            errors["persistent_commit"] = max(errors["persistent_commit"], check(receipt.logits, expected))
                            assert receipt.tokens_seen == history + chunk and receipt.revision == 2 and not receipt.replayed
                            if trial:
                                samples["persistent_commit"].append(duration)
                            duration, replay = timed(lambda: runner.run(tail, request_id="tail"))
                            errors["exact_retry"] = max(errors["exact_retry"], check(replay.logits, expected))
                            assert replay.replayed and replay.revision == 2
                            if trial:
                                samples["exact_retry"].append(duration)
                    output["cases"].append({"kind": kind, "history_tokens": history,
                                            "new_tokens": chunk, "timings": {k: describe(v) for k, v in samples.items()},
                                            "max_abs_error": errors, "prefix_state": wire_sizes(prefix_state),
                                            "tail_output": wire_sizes(expected),
                                            "standalone_state_codec_round_trip": describe(codec_samples)})
                # The fresh subprocess computes only the new 16-token chunk.
                cold_path = folder / f"{kind}-{history}-cold.sqlite3"
                shutil.copyfile(base_path, cold_path)
                env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
                cmd = [sys.executable, str(Path(__file__).resolve()), "--child", "--db", str(cold_path),
                       "--kind", kind, "--history", str(history), "--chunk", "16"]
                begin = time.perf_counter()
                completed = subprocess.run(cmd, text=True, capture_output=True, check=True, timeout=90, env=env)
                process_wall_ms = (time.perf_counter() - begin) * 1000
                result = json.loads(completed.stdout)
                assert result["revision"] == 2 and result["tokens_seen"] == history + 16
                with torch.inference_mode():
                    expected = model(tokens[:, :history + 16])[0][:, history:]
                error = check(torch.tensor(result.pop("logits")), expected)
                output["cold_processes"].append({"kind": kind, "history_tokens": history,
                                                  "new_tokens": 16, "process_wall_ms": process_wall_ms,
                                                  "max_abs_error": error, **result})
                print(json.dumps({"progress": kind, "history_tokens": history}), flush=True)
    final_hashes = {f: hashlib.sha256((PACKAGE_ROOT / "nanocog" / f).read_bytes()).hexdigest()
                    for f in files}
    if final_hashes != source_hashes or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != harness_hash:
        raise RuntimeError("source files changed during timing; discard and rerun")
    output["wall_seconds"] = time.perf_counter() - start_wall
    output["parent_cpu_seconds"] = time.process_time() - start_cpu
    output["qualification"] = "numerical continuation and descriptive one-host cost only; no trained quality or lifetime claim"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "cases": len(output["cases"]),
                      "wall_seconds": output["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results.json"))
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--db")
    parser.add_argument("--kind", choices=tuple(KINDS))
    parser.add_argument("--history", type=int)
    parser.add_argument("--chunk", type=int)
    parser.add_argument("--work-dir", type=Path, default=PACKAGE_ROOT.parents[2])
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    child(args) if args.child else run(args)
