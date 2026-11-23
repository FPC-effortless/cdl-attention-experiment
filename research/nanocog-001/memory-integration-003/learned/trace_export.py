"""Post-fit diagnostic export; never selects or modifies a fitted model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from runner import (ARMS, SEEDS, HERE, NATIVE_SHA, clock, file_sha, load_adapter,
                    load_corpus, predictions, save_json, sha, texts_in, timing)


def run(kit: Path, private: Path):
    started = clock()
    calibration = json.loads((private / "CALIBRATION.json").read_text())
    train_sha = file_sha(private / "train_features.pt")
    if train_sha != calibration["extraction"]["sha256"]:
        raise ValueError("training feature cache no longer matches its extraction receipt")
    expected = json.loads((private / "all_features.json").read_text())["sha256"]
    actual = file_sha(private / "all_features.pt")
    if actual != expected:
        raise ValueError("evaluation feature cache no longer matches its extraction receipt")
    cache = torch.load(private / "all_features.pt", weights_only=True)
    data = load_corpus(private)
    all_texts = set().union(*(texts_in(v) for v in data.values()))
    if (cache["native_sha"] != NATIVE_SHA or cache["texts_sha256"] != sha(cache["texts"]) or
            set(cache["texts"]) != all_texts or not torch.isfinite(cache["features"]).all()):
        raise ValueError("native cache identity/text/finite-value validation failed")
    outputs = []
    destination = private / "predictions"
    destination.mkdir(exist_ok=True)
    for seed in SEEDS:
        for arm in ARMS:
            checkpoint = private / "checkpoints" / f"{arm}-{seed}.pt"
            adapter = load_adapter(kit, checkpoint)
            pred = predictions(adapter, cache, all_texts)
            path = destination / f"{arm}-{seed}.json"
            record = {"kind": "fixed-final-checkpoint-diagnostic-predictions", "arm": arm, "seed": seed,
                      "checkpoint_sha256": file_sha(checkpoint), "native_sha256": NATIVE_SHA,
                      "cache_sha256": actual, "model_identity": adapter.model_identity(),
                      "predictions": pred}
            save_json(path, record)
            outputs.append({"arm": arm, "seed": seed, "path": path.name,
                            "sha256": file_sha(path), "text_count": len(all_texts)})
    result = {"training_cache_sha256": train_sha, "evaluation_cache_sha256": actual,
              "cache_integrity_matches_extraction_receipts": True, "outputs": outputs,
              "timing": timing(started),
              "status": "post-fit diagnostics only; no training, tuning, or checkpoint selection"}
    save_json(HERE / "TRACE_EXPORT.json", result)
    save_json(private / "TRACE_EXPORT.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    run(args.kit, args.private)
