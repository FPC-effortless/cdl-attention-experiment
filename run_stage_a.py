import json
from pathlib import Path

import pandas as pd

import app


OUT = Path("benchmark-output")
OUT.mkdir(exist_ok=True)

CONFIGS = [
    {"cases": 40, "memories": 6, "native_attention": True},
    {"cases": 40, "memories": 12, "native_attention": False},
    {"cases": 40, "memories": 24, "native_attention": False},
]

summary_rows = []
metadata_rows = []

for cfg in CONFIGS:
    print(f"Running Stage A: {cfg}", flush=True)
    results, examples, metadata_json = app.run_benchmark(
        cfg["cases"], cfg["memories"], cfg["native_attention"]
    )
    tag = f"n{cfg['cases']}_m{cfg['memories']}"
    results.to_csv(OUT / f"results_{tag}.csv", index=False)
    examples.to_csv(OUT / f"examples_{tag}.csv", index=False)
    metadata = json.loads(metadata_json)
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
print("\n=== Stage-A summary ===")
print(summary.to_string(index=False))
