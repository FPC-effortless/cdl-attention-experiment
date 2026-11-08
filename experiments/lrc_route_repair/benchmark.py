"""Complete-call CPU cost measurement, after frozen fitting and scoring close."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch

from .run import SEEDS, check_frozen, digest, load_model, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    torch.set_num_threads(1)
    check_frozen(output)
    complete = json.loads((output / "complete.json").read_text())
    if complete["status"] != "complete":
        raise RuntimeError("scoring must finish before timing")
    closed = json.loads((output / "fits_closed.json").read_text())
    fits = {(r["arm"], r["seed"]): r for r in closed["receipts"]}
    for (arm, seed), receipt in fits.items():
        if digest(output / f"{arm}_{seed}.pt") != receipt["checkpoint_sha256"]:
            raise RuntimeError("checkpoint changed since fitting closure")
    train = torch.load(output / "train.pt", weights_only=True)
    panel = [(train["records"][i:i+1], train["query"][i:i+1]) for i in range(32)]
    arms = ("repair", "original_lrc", "memnet")
    raw, summary = {}, []
    with torch.inference_mode():
        for seed in SEEDS:
            models = {arm: load_model(output, arm, seed) for arm in arms}
            for model in models.values():
                for records, query in panel:
                    model(records, query)
            blocks = {arm: [] for arm in arms}
            for block in range(30):
                rotation = block % len(arms)
                order = arms[rotation:] + arms[:rotation]
                for arm in order:
                    model = models[arm]
                    started = time.perf_counter_ns()
                    for records, query in panel:
                        model(records, query)
                    blocks[arm].append((time.perf_counter_ns() - started) / 32 / 1e6)
            raw[str(seed)] = blocks
            reference_i = statistics.median(blocks["memnet"])
            reference_t = fits["memnet", seed]["fit_seconds"]
            for arm in arms:
                inference = statistics.median(blocks[arm])
                training = fits[arm, seed]["fit_seconds"]
                summary.append(dict(arm=arm, seed=seed, complete_call_ms=inference,
                    fit_seconds=training, inference_ratio_to_memnet=inference/reference_i,
                    training_ratio_to_memnet=training/reference_t,
                    J=12 * math.log(inference/reference_i) + math.log(training/reference_t)))
    result = dict(scope="complete batch1 calls including record encoding and index construction",
                  threads=1, panel="first32 TRAIN examples", warmups=32, blocks=30,
                  calls_per_block=32, clock="perf_counter_ns", raw_block_means_ms=raw,
                  results=summary, fitted_checkpoints_verified=True,
                  complete_sha256=digest(output / "complete.json"), finished_unix=time.time())
    write_json(output / "benchmark.json", result)
    print(json.dumps(result["results"], indent=2), flush=True)


if __name__ == "__main__":
    main()
