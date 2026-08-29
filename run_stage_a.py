import json
import random
from pathlib import Path

import pandas as pd

import app


OUT = Path("benchmark-output")
OUT.mkdir(exist_ok=True)

# Paired benchmark design: every case gets its own RNG determined only by
# (global seed, case index). Increasing candidate count therefore preserves
# the same underlying relation/entity/answer and the same initial hard
# distractors, then adds more distractors. This makes 6 -> 12 -> 24 a real
# scaling comparison instead of three unrelated random samples.
def paired_make_benchmark(n_cases: int, n_memories: int, seed: int = app.SEED):
    return [
        app.make_case(random.Random(seed + 104729 * i), n_memories)
        for i in range(n_cases)
    ]


app.make_benchmark = paired_make_benchmark

CONFIGS = [
    {"cases": 120, "memories": 6, "native_attention": True},
    {"cases": 120, "memories": 12, "native_attention": False},
    {"cases": 120, "memories": 24, "native_attention": False},
]

summary_rows = []
metadata_rows = []

for cfg in CONFIGS:
    print(f"Running paired Stage A: {cfg}", flush=True)
    results, examples, metadata_json = app.run_benchmark(
        cfg["cases"], cfg["memories"], cfg["native_attention"]
    )
    tag = f"paired_n{cfg['cases']}_m{cfg['memories']}"
    results.to_csv(OUT / f"results_{tag}.csv", index=False)
    examples.to_csv(OUT / f"examples_{tag}.csv", index=False)
    metadata = json.loads(metadata_json)
    metadata["paired_case_design"] = True
    metadata_rows.append(metadata)
    for row in results.to_dict(orient="records"):
        summary_rows.append({
            "cases": cfg["cases"],
            "memories": cfg["memories"],
            **row,
        })

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT / "summary.csv", index=False)
(OUT / "metadata.json").write_text(json.dumps(metadata_rows, indent=2))
print("\n=== Paired Stage-A summary ===")
print(summary.to_string(index=False))
