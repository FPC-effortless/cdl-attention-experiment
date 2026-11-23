"""Actual trained inference through SQLite; labels remain evaluator-only.

The prespecified candidate17 is used; all sampled histories run, including
failures. This bounded integration/serving sample is not the main TEST panel.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time

import torch

from runner import (clock, file_sha, load_adapter, load_corpus, save_json, timing,
                    HERE, predictions, evaluate_pairs)
from nanocog.memory.learned_archive import FactualMemory, UNKNOWN, execute_operation, payload_value, read_ledger, fact_key
from nanocog.memory.store import MemoryStore

SELECTION = {"arm": "candidate", "seed": 17,
             "panels": ["familiar_test", "fresh_test"], "lengths": [8, 64],
             "pairs_per_panel_length": 4,
             "selection": "first four family indices in each panel/length; fixed before TEST evaluation",
             "raw_control": "same actual native/text heads with exact local typed ledger; no feature cache",
             "timing_claim": "paired output-matched infrastructure overhead, not a model speedup"}


def stats(values):
    ordered = sorted(values)
    return {"calls": len(values), "mean_ms": 1000 * sum(values) / len(values),
            "p50_ms": 1000 * ordered[len(values) // 2],
            "p95_ms": 1000 * ordered[min(len(values) - 1, int(.95 * len(values)))],
            "total_seconds": sum(values)}


def run(kit: Path, private: Path):
    started = clock()
    saved_selection = json.loads((HERE / "DURABLE_PROTOCOL.json").read_text())
    if saved_selection != SELECTION:
        raise ValueError("durable sample protocol changed")
    load_start = clock()
    adapter = load_adapter(kit, private / "checkpoints/candidate-17.pt")
    load_time = timing(load_start)
    data = load_corpus(private)
    cache = torch.load(private / "all_features.pt", weights_only=True)
    rows, measurements = [], []
    with tempfile.TemporaryDirectory(prefix="cog-memory003-durable-") as temporary:
        environment = {"TMPDIR": os.environ.get("TMPDIR"), "temporary_directory": temporary,
                       "filesystem": subprocess.run(["findmnt", "-n", "-o", "TARGET,FSTYPE,SOURCE", "--target", temporary],
                                                     check=True, capture_output=True, text=True).stdout.strip(),
                       "python": platform.python_version(), "torch": str(torch.__version__),
                       "torch_threads": torch.get_num_threads()}
        for panel in SELECTION["panels"]:
            for length in SELECTION["lengths"]:
                chosen = [p for p in data[panel] if p["length"] == length][:4]
                texts = {e["text"] for p in chosen for w in p["worlds"] for e in w["events"]} | {
                    w["query"] for p in chosen for w in p["worlds"]}
                cached = predictions(adapter, cache, texts)
                expected = {r["lineage"]: r for r in evaluate_pairs(chosen, cached)}
                raw_write_times, durable_write_times, raw_answer_times, durable_answer_times = [], [], [], []
                pair_strict = pair_parity = 0
                for pair in chosen:
                    correct = parity = True
                    raw_ops, durable_ops, actual_answers = [], [], []
                    for world_i, world in enumerate(pair["worlds"]):
                        path = Path(temporary) / f"{panel}-{pair['index']}-{world_i}.sqlite3"
                        local_ledger = [UNKNOWN] * 8
                        with MemoryStore(path, principal_id="synthetic-evaluation") as store:
                            memory = FactualMemory(store, adapter, session_id="first-chat", authority="experimental-model-policy")
                            for event_i, event in enumerate(world["events"]):
                                raw_start = time.perf_counter()
                                op = adapter.predict_events([event["text"]])[0]
                                execute_operation(local_ledger, op)
                                raw_write_times.append(time.perf_counter() - raw_start)
                                durable_start = time.perf_counter()
                                result = memory.observe(event["text"], request_id=f"event-{event_i}")
                                durable_write_times.append(time.perf_counter() - durable_start)
                                actual = []
                                for record in range(8):
                                    snapshot = store.fact_snapshot(fact_key(record))["record"]
                                    actual.append(UNKNOWN if snapshot is None else payload_value(record, snapshot["value"]))
                                parity &= result.operation == op == cached[event["text"]][0] and actual == local_ledger
                                correct &= result.operation == event["operation"] and actual == event["ledger"]
                                raw_ops.append(op); durable_ops.append(result.operation)
                        # A new connection and chat exercise durable state rather
                        # than Python ledger reuse. Fresh PROCESS is independently
                        # checked by the root integration runner.
                        with MemoryStore(path, principal_id="synthetic-evaluation") as store:
                            memory = FactualMemory(store, adapter, session_id="second-chat", authority="experimental-model-policy")
                            raw_start = time.perf_counter()
                            route = adapter.predict_queries([world["query"]])[0]
                            raw_answer = adapter.predict_answers([read_ledger(local_ledger, route)])[0]
                            raw_answer_times.append(time.perf_counter() - raw_start)
                            durable_start = time.perf_counter()
                            answer = memory.ask(world["query"], request_id="question")
                            durable_answer_times.append(time.perf_counter() - durable_start)
                            parity &= answer.answer_index == raw_answer and answer.route == route == cached[world["query"]][1]
                            correct &= answer.answer_index == world["answer"]
                            actual_answers.append(answer.answer_index)
                    parity &= correct == expected[pair["lineage"]]["strict_pair"]
                    pair_strict += bool(correct); pair_parity += bool(parity)
                    rows.append({"panel": panel, "lineage": pair["lineage"], "index": pair["index"],
                                 "length": length, "strict_pair": bool(correct),
                                 "cached_and_raw_and_durable_parity": bool(parity),
                                 "raw_operations": raw_ops, "durable_operations": durable_ops,
                                 "answers": actual_answers})
                measurements.append({"panel": panel, "length": length, "pairs": len(chosen),
                                     "strict_pairs": pair_strict, "all_path_parity_pairs": pair_parity,
                                     "raw_observe": stats(raw_write_times), "durable_observe": stats(durable_write_times),
                                     "raw_answer": stats(raw_answer_times), "durable_answer": stats(durable_answer_times)})
                print(json.dumps(measurements[-1]), flush=True)
    result = {"protocol": SELECTION, "protocol_sha256": file_sha(HERE / "DURABLE_PROTOCOL.json"),
              "source_sha256": file_sha(Path(__file__).resolve()), "environment": environment,
              "model_identity": adapter.model_identity(), "measurements": measurements,
              "all_path_parity": all(r["cached_and_raw_and_durable_parity"] for r in rows),
              "load": load_time, "total": timing(started), "rows": rows,
              "caveat": "Fixed small serving sample; native/raw then durable order is not randomized. No speedup claim."}
    save_json(HERE / "DURABLE_RESULTS.json", result)
    save_json(private / "DURABLE_RESULTS.json", result)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--kit", type=Path, required=True)
    p.add_argument("--private", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    run(args.kit, args.private)
