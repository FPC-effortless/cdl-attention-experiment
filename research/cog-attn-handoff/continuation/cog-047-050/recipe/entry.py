"""TrainLab driver for original-utterance and context-cost experiments."""
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

suite,arm=sys.argv[1:]
seed=int(os.environ["TRAINLAB_SEED"])
out=Path(os.environ["TRAINLAB_OUTPUT_DIR"])
results=importlib.import_module(suite).run(seed,arm,out)
payload=json.dumps(results,separators=(",",":")).encode()
(out/"results.json").write_bytes(payload)
if suite=="authentic":
    predictions=results["result"]["test"]["predictions"]
    score=results["result"]["test"]["exact"]/len(predictions)
else:
    predictions=[r["steps"] for r in results]
    score=sum(s["raw"]==s["gold"] for block in predictions for s in block)/sum(map(len,predictions))
(out/"results.sha256").write_text(hashlib.sha256(json.dumps(predictions,sort_keys=True).encode()).hexdigest())
(out/"metrics.json").write_text(json.dumps(dict(score=score)))
