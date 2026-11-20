import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

suite,arm=sys.argv[1:]
out=Path(os.environ["TRAINLAB_OUTPUT_DIR"])
result=importlib.import_module(suite).run(int(os.environ["TRAINLAB_SEED"]),arm)
serialized=json.dumps(result,separators=(",",":")).encode()
(out/"results.json").write_bytes(serialized)
(out/"results.sha256").write_text(hashlib.sha256(serialized).hexdigest())
if suite=="binding":
    row=next(r for r in result[0]["rows"] if r["cohort"]=="joint")
    score=row["exact"]/512
else:
    rows=[r for r in result["rows"] if r["scenario"]=="constant" and r["channel"]=="independent" and
          r["budget"]==4 and r["policy"]==("active" if arm=="candidate" else "uncorrected")]
    score=sum(r["correct"] for r in rows)/len(rows)
(out/"metrics.json").write_text(json.dumps(dict(score=score)))
