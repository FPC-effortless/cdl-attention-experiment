import hashlib
import json
import os
import sys
from pathlib import Path

from implicit import run

out=Path(os.environ["TRAINLAB_OUTPUT_DIR"])
result=run(int(os.environ["TRAINLAB_SEED"]),sys.argv[-1])
data=json.dumps(result,separators=(",",":")).encode()
(out/"results.json").write_bytes(data)
(out/"results.sha256").write_text(hashlib.sha256(data).hexdigest())
row=next(r for r in result if r["cohort"]=="shared_wording")
(out/"metrics.json").write_text(json.dumps({"score":row["exact"]/512}))
