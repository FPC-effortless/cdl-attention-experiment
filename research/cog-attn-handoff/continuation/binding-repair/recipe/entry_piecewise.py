import hashlib
import json
import os
import sys
from pathlib import Path

from bias_piecewise import run

out=Path(os.environ["TRAINLAB_OUTPUT_DIR"])
result=run(int(os.environ["TRAINLAB_SEED"]),sys.argv[-1])
data=json.dumps(result,separators=(",",":")).encode()
(out/"results.json").write_bytes(data)
(out/"results.sha256").write_text(hashlib.sha256(data).hexdigest())
rows=[row for row in result["rows"] if row["scenario"]=="piecewise" and row["budget"]==4]
(out/"metrics.json").write_text(json.dumps({"score":sum(row["validation"] and row["correct"] for row in rows)/len(rows)}))
