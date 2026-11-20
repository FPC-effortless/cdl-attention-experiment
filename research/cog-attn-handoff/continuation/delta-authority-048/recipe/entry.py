"""TrainLab driver for the frozen module and authority experiments."""
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

study,arm=sys.argv[1:]
seed=int(os.environ["TRAINLAB_SEED"])
out=Path(os.environ["TRAINLAB_OUTPUT_DIR"])
mod=importlib.import_module(study)
data=mod.run(seed,arm,out) if study=="bridge" else mod.run(seed,arm)
payload=json.dumps(data,separators=(",",":")).encode()
(out/"results.json").write_bytes(payload)
if study=="bridge":
    stable=data["profiles"]
    cohort=stable[0]["results"]["integrated"]
    score=sum(row["exact"] for row in cohort)/len(cohort)
else:
    stable=data
    cohort=[r for r in data["results"] if r["profile"] in ("unsigned_alias","enrolled_mirrors","replay_nonce")
            and r["policy"]==("scoped" if arm=="candidate" else "signed")]
    score=sum(r["admitted"] is None for r in cohort)/len(cohort)
(out/"results.sha256").write_text(hashlib.sha256(json.dumps(stable,sort_keys=True).encode()).hexdigest())
(out/"metrics.json").write_text(json.dumps({"score":score}))
