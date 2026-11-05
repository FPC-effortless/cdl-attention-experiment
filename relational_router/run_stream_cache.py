"""LRR-003B: frozen prefill-versus-repeated-query benchmark."""
from __future__ import annotations
import argparse
import json
import statistics
import time
from pathlib import Path

from .raw_causal import generate, train_model
from .stream_cache import CachedCausalIndex


def run(lengths=(6,32,128,512), repeats=24, seed=21) -> dict:
    began = time.perf_counter()
    model, metadata = train_model(seed, steps=700, batch_size=48)
    training_sec = time.perf_counter() - began
    report = {
        "experiment": "LRR-003B persistent cache",
        "model_seed": seed,
        "training": metadata,
        "training_wall_seconds": training_sec,
        "repeats_per_history": repeats,
        "cache_tail_size": 14,
        "rows": [],
    }
    for facts in lengths:
        example = generate(98_000_000 + facts, facts, "test")
        tokens = example.tokens
        history = " ".join(tokens[:-5])
        question = " ".join(tokens[-5:])
        model.eval()
        cache = CachedCausalIndex(model)
        prefill = cache.append(history)
        cached_times = []
        full_times = []
        addresses_equal = 0
        cached_hits = 0
        full_hits = 0
        first_result = None
        for _ in range(repeats):
            selected = cache.query(question)
            recomputed = model.route(example.text)
            cached_times.append(selected["elapsed_query_cpu_us"])
            full_times.append(recomputed["elapsed_total_cpu_us"])
            addresses_equal += int(selected["query_address"] == recomputed["query_address"])
            cached_hits += int(selected["token_indices"] == (example.gold_index,))
            full_hits += int(recomputed["token_index"] == example.gold_index)
            if first_result is None:
                first_result = selected
            elif selected["token_indices"] != first_result["token_indices"]:
                raise AssertionError("cache is not stable under repeated queries")
        c_median = statistics.median(cached_times)
        f_median = statistics.median(full_times)
        report["rows"].append({
            "facts": facts,
            "tokens_in_history": len(tokens)-5,
            "cached_encoded_tokens_per_query": first_result["encoded_token_count"],
            "gold_index": example.gold_index,
            "cached_correct_queries": cached_hits,
            "naive_correct_queries": full_hits,
            "query_address_matches": addresses_equal,
            "index_probes_per_cached_query": first_result["index_probes"],
            "prefill_wall_us": prefill["elapsed_ingest_cpu_us"],
            "cache_writes": prefill["writes"],
            "cache_distinct_addresses": prefill["distinct_addresses"],
            "cached_query_median_us": c_median,
            "naive_full_query_median_us": f_median,
            "cached_query_p95_us": sorted(cached_times)[int(.95*len(cached_times))-1],
            "naive_full_query_p95_us": sorted(full_times)[int(.95*len(full_times))-1],
            "median_naive_to_cached_ratio": f_median/c_median,
            "amortized_cached_wall_per_query_for_1_query_us": prefill["elapsed_ingest_cpu_us"]+c_median,
            "amortized_cached_wall_per_query_for_64_queries_us": prefill["elapsed_ingest_cpu_us"]/64+c_median,
        })
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("/tmp/lrr003b.json"))
    args = p.parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2)+"\n")
    print(json.dumps({
        "experiment": report["experiment"],
        "training": report["training"],
        "rows": report["rows"],
    }, indent=2))


if __name__ == "__main__":
    main()
