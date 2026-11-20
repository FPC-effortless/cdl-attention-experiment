import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

suite,arm=sys.argv[1:]
seed=int(os.environ["TRAINLAB_SEED"])
artifact=importlib.import_module(suite).run(seed,arm)
out=Path(os.environ["TRAINLAB_OUTPUT_DIR"])
(out/"results.json").write_text(json.dumps(artifact,separators=(",",":")))
# Keep results separately from the stable fingerprint and exclude memory/timing observations.
def stable(x):
    if isinstance(x,dict):
        return {k:stable(v) for k,v in x.items() if not k.endswith(("_ns","_kib")) and k!="worker_stats"}
    if isinstance(x,list):
        return [stable(v) for v in x]
    return x
payload=json.dumps(stable(artifact),sort_keys=True).encode()
(out/"result.sha256").write_text(hashlib.sha256(payload).hexdigest())
if suite=="language":
    summary=next(r for r in artifact[-1]["rows"] if r["cohort"]=="syntax")
    score=summary["effects"]/summary["count"]
elif suite=="epistemic":
    rows=artifact["shifts"][0]["worlds"]
    score=1-sum(r["wrong"] for r in rows)/(1024*len(rows))
else:
    score=sum(row[2:]==[1,1] for row in artifact["race"]["journal"])/400
(out/"metrics.json").write_text(json.dumps(dict(score=score)))
