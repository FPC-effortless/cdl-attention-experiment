import hashlib
import json
import os
import sys
from pathlib import Path

from nli_transfer import run

out=Path(os.environ["TRAINLAB_OUTPUT_DIR"])
data=run(int(os.environ["TRAINLAB_SEED"]),sys.argv[-1])
serialized=json.dumps(data,separators=(",",":")).encode()
(out/"results.json").write_bytes(serialized)
# Hash discrete predictions separately from floating-point scores.
payload=[(m["mode"],[(k,v["pred"]) for k,v in m["rows"].items()]) for m in data]
(out/"results.sha256").write_text(hashlib.sha256(json.dumps(payload).encode()).hexdigest())
correct=data[0]["rows"]["implicit"]["correct"]
(out/"metrics.json").write_text(json.dumps({"score":sum(correct)/len(correct)}))
