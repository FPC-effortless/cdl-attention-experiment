#!/usr/bin/env python3
"""Paired frozen-package persistence costs; seeded fixtures, no model fit."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
WORKSPACE = REPO.parent
BASELINE = WORKSPACE / "memory003_serving_baseline/research/nanocog-001"
CANDIDATE = WORKSPACE / "memory003_serving_candidate/research/nanocog-001"
SEED = 91023
FILES = ("model.py", "interaction.py", "comm/packet.py", "state/records.py",
         "memory/store.py", "memory/decoder.py", "memory/codec.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def measure(call):
    start = time.perf_counter_ns()
    result = call()
    return (time.perf_counter_ns() - start) / 1e6, result


def stats(values):
    return dict(median_ms=statistics.median(values), p95_ms=sorted(values)[math.ceil(.95 * len(values)) - 1],
                samples_ms=values)


def worker(args):
    sys.path.insert(0, str(args.package))
    import sqlite3
    import torch
    from nanocog.model import Config, Decoder, LocalOnly
    from nanocog.interaction import PacketMixer, PacketAndTypedProbe
    from nanocog.memory.decoder import PersistentDecoder, _model_identity
    from nanocog.memory.store import MemoryStore
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cls = {"local": LocalOnly, "packet": PacketMixer, "typed_packet": PacketAndTypedProbe}[args.kind]
    torch.manual_seed(SEED)
    model = Decoder(Config(256, 160, 4), cls).eval()
    generator = torch.Generator().manual_seed(SEED + 1)
    tokens = torch.randint(0, 256, (1, 1040), generator=generator)
    tokenizer, session = "synthetic-byte-256-v1", "cost-stream"
    sources = {name: sha(args.package / "nanocog" / name) for name in FILES}
    evidence = dict(variant=args.variant, kind=args.kind, seed=SEED,
                    parameters=sum(p.numel() for p in model.parameters()), trained=False,
                    versions=dict(python=platform.python_version(), torch=str(torch.__version__), sqlite=sqlite3.sqlite_version),
                    source_sha256=sources, harness_sha256=sha(__file__), cases=[])
    args.fixture_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "cold":
        target = args.fixture_dir / f"{args.kind}-{args.variant}-cold.sqlite3"
        with MemoryStore(target, principal_id="cost", namespace="validation") as store:
            adapter = PersistentDecoder(model, store, session_id=session, tokenizer_id=tokenizer)
            evidence["manifest"] = adapter._manifest
            duration, result = measure(lambda: adapter.run(tokens[:, 1024:1040], request_id="tail"))
            assert result.revision == 2 and result.tokens_seen == 1040 and not result.replayed
            checkpoint = store.load_checkpoint(session)
            evidence["compatibility"] = dict(
                output_sha256=hashlib.sha256(result.logits.contiguous().numpy().tobytes()).hexdigest(),
                checkpoint_sha256=hashlib.sha256(canonical(checkpoint)).hexdigest(),
                revision=result.revision, tokens_seen=result.tokens_seen,
                manifest_equal=checkpoint["manifest"] == evidence["manifest"], inner_run_ms=duration)
    else:
        evidence["manifest"] = _model_identity(model, tokenizer, session)
        identity_samples = [measure(lambda: _model_identity(model, tokenizer, session))[0] for _ in range(6)][1:]
        evidence["attachment_identity"] = stats(identity_samples)
        for history in (128, 1024):
            prefix = tokens[:, :history]
            with torch.inference_mode():
                _, prefix_state = model(prefix)
            base_path = args.fixture_dir / f"{args.kind}-{args.variant}-{history}-base.sqlite3"
            with MemoryStore(base_path, principal_id="cost", namespace="validation") as store:
                adapter = PersistentDecoder(model, store, session_id=session, tokenizer_id=tokenizer)
                adapter.run(prefix, request_id="prefix")
                hot_identity = [measure(adapter._check_model)[0] for _ in range(6)][1:]
            assert not Path(str(base_path) + "-wal").exists()
            for chunk in (1, 16):
                tail = tokens[:, history:history + chunk]
                with torch.inference_mode():
                    expected = model(tokens[:, :history + chunk])[0][:, history:]
                times = {key: [] for key in ("ram", "persistent", "retry")}
                checkpoint_hashes, output_hashes = set(), set()
                maximum_error = 0.0
                for trial in range(6):
                    path = args.fixture_dir / f"{args.kind}-{args.variant}-{history}-{chunk}-{trial}.sqlite3"
                    shutil.copyfile(base_path, path)
                    with MemoryStore(path, principal_id="cost", namespace="validation") as store:
                        adapter = PersistentDecoder(model, store, session_id=session, tokenizer_id=tokenizer)
                        with torch.inference_mode():
                            duration, direct = measure(lambda: model(tail, prefix_state)[0])
                        torch.testing.assert_close(direct, expected, rtol=1e-5, atol=1e-6)
                        if trial:
                            times["ram"].append(duration)
                        duration, result = measure(lambda: adapter.run(tail, request_id="tail"))
                        torch.testing.assert_close(result.logits, expected, rtol=1e-5, atol=1e-6)
                        assert result.revision == 2 and result.tokens_seen == history + chunk and not result.replayed
                        maximum_error = max(maximum_error, float((result.logits - expected).abs().max()))
                        checkpoint = store.load_checkpoint(session)
                        checkpoint_hash = hashlib.sha256(canonical(checkpoint)).hexdigest()
                        checkpoint_hashes.add(checkpoint_hash)
                        output_hashes.add(hashlib.sha256(result.logits.contiguous().numpy().tobytes()).hexdigest())
                        if trial:
                            times["persistent"].append(duration)
                        duration, replay = measure(lambda: adapter.run(tail, request_id="tail"))
                        assert replay.replayed and replay.revision == 2 and replay.tokens_seen == history + chunk
                        assert torch.equal(replay.logits, result.logits)
                        assert hashlib.sha256(canonical(store.load_checkpoint(session))).hexdigest() == checkpoint_hash
                        if trial:
                            times["retry"].append(duration)
                assert len(checkpoint_hashes) == len(output_hashes) == 1
                evidence["cases"].append(dict(history_tokens=history, new_tokens=chunk,
                    timings={key: stats(values) for key, values in times.items()},
                    hot_identity=stats(hot_identity), maximum_abs_error=maximum_error,
                    checkpoint_sha256=checkpoint_hashes.pop(), output_sha256=output_hashes.pop()))
    evidence["source_sha256_after"] = {name: sha(args.package / "nanocog" / name) for name in FILES}
    assert evidence["source_sha256_after"] == sources
    assert evidence["harness_sha256"] == sha(__file__)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(output=str(args.output), variant=args.variant, kind=args.kind, mode=args.mode)))


def orchestrate(args):
    if not (BASELINE / "nanocog").is_dir():
        raise RuntimeError("first extract the published ffb842a baseline into memory003_serving_baseline")
    CANDIDATE.mkdir(parents=True, exist_ok=True)
    shutil.copytree(BASELINE / "nanocog", CANDIDATE / "nanocog", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("decoder.py", "codec.py"):
        shutil.copyfile(REPO / "research/nanocog-001/nanocog/memory" / name,
                        CANDIDATE / "nanocog/memory" / name)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    harness_hash = sha(__file__)
    summary = dict(baseline_commit="ffb842adc176496d8838b774d0fe412aec5d1d20", harness_sha256=harness_hash,
                   trained=False, warmups=1, measured_trials=5, round="one quiet reverse-order confirmation",
                   fixture_store="frozen baseline v2 in both variants", models=[],
                   cold_timing_boundary="subprocess wall time: Python/PyTorch import, deterministic model construction, store opening, finite-checked attachment, continuation, checkpoint/output hash verification and evidence writing; database copy excluded")
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="memory003-serving-", dir=WORKSPACE) as td:
        fixture_dir = Path(td)
        summary["filesystem"] = subprocess.check_output(["df", "-PT", str(fixture_dir)], text=True).strip()
        for index, kind in enumerate(("local", "packet", "typed_packet")):
            order = ("candidate", "baseline") if index % 2 == 0 else ("baseline", "candidate")
            pair = {}
            for variant in order:
                package = BASELINE if variant == "baseline" else CANDIDATE
                output = args.output_dir / f"{kind}-{variant}.json"
                cmd = [sys.executable, str(Path(__file__).resolve()), "--mode", "worker", "--kind", kind,
                       "--variant", variant, "--package", str(package), "--fixture-dir", str(fixture_dir), "--output", str(output)]
                completed = subprocess.run(cmd, env=env, text=True, capture_output=True, check=True, timeout=180)
                print(completed.stdout.strip(), flush=True)
                pair[variant] = json.loads(output.read_text())
            assert pair["baseline"]["manifest"] == pair["candidate"]["manifest"]
            for old, new in zip(pair["baseline"]["cases"], pair["candidate"]["cases"]):
                assert (old["history_tokens"], old["new_tokens"]) == (new["history_tokens"], new["new_tokens"])
                assert old["checkpoint_sha256"] == new["checkpoint_sha256"]
                assert old["output_sha256"] == new["output_sha256"]
            expected = next(c for c in pair["baseline"]["cases"] if c["history_tokens"] == 1024 and c["new_tokens"] == 16)
            cold = {}
            for variant in order:
                package = BASELINE if variant == "baseline" else CANDIDATE
                origin = fixture_dir / f"{kind}-baseline-1024-base.sqlite3"
                target = fixture_dir / f"{kind}-{variant}-cold.sqlite3"
                shutil.copyfile(origin, target)
                cold_path = args.output_dir / f"{kind}-{variant}-cold.json"
                cmd = [sys.executable, str(Path(__file__).resolve()), "--mode", "cold", "--kind", kind,
                       "--variant", variant, "--package", str(package), "--fixture-dir", str(fixture_dir), "--output", str(cold_path)]
                cold_start = time.perf_counter_ns()
                subprocess.run(cmd, env=env, text=True, capture_output=True, check=True, timeout=90)
                process_wall_ms = (time.perf_counter_ns() - cold_start) / 1e6
                compat = json.loads(cold_path.read_text())
                assert compat["compatibility"]["checkpoint_sha256"] == expected["checkpoint_sha256"]
                assert compat["compatibility"]["output_sha256"] == expected["output_sha256"]
                assert compat["compatibility"]["manifest_equal"]
                cold[variant] = dict(process_wall_ms=process_wall_ms,
                                     inner_run_ms=compat["compatibility"]["inner_run_ms"])
            summary["models"].append(dict(kind=kind, order=order, manifest_equal=True,
                complete_checkpoint_cases_equal=4, output_cases_equal=4, baseline_native_session_continues=True,
                cold=cold))
    assert sha(__file__) == harness_hash
    summary["wall_seconds"] = time.perf_counter() - start
    summary["passed"] = True
    (args.output_dir / "comparison.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("orchestrate", "worker", "cold"), default="orchestrate")
    parser.add_argument("--variant", choices=("baseline", "candidate"))
    parser.add_argument("--kind", choices=("local", "packet", "typed_packet"))
    parser.add_argument("--package", type=Path)
    parser.add_argument("--fixture-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path, default=HERE / "confirmation")
    args = parser.parse_args()
    orchestrate(args) if args.mode == "orchestrate" else worker(args)
