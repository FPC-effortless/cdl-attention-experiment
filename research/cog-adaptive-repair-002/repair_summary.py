"""Verify matched continuation inputs and collect per-seed held-out readouts."""
import csv
import hashlib
import json
from pathlib import Path
import sys


def collect(root, out):
    out.mkdir(exist_ok=True, parents=True)
    hashes = {}
    panels = {}
    rows = []
    for path in sorted(root.glob("*/metrics.json")):
        r = json.loads(path.read_text())
        model = path.with_name("weights.pt")
        with model.open("rb") as f:
            assert hashlib.file_digest(f,"sha256").hexdigest() == r["trained_weights_sha256"]
        key = (r["seed"],r["data"])
        assert hashes.setdefault(key,r["training_sha256"]) == r["training_sha256"]
        s = r["assessment"]
        row = {k:r[k] for k in ("arm","data","regime","seed","parameters","steps","training_sha256","parent_sha256","trained_weights_sha256","training_cpu_seconds")}
        assert row["steps"] == 256 and r["question_streams"] == 2048 and r["real_text_replay_targets"] == 32768
        row["dev_bpb"] = s["dev_bpb"]
        row["cached_decode_median_ms"]=s["profile"]["decode_step_median_ms"]
        for group in ("panels","confirmation"):
            for name,v in s[group].items():
                assert panels.setdefault(name,v["sha256"])==v["sha256"] and v["worlds"]==64
                row[name]=v["both_correct"]
                row[name+"_off"]=r.get("read_disabled",{}).get(group,{}).get(name,{}).get("both_correct")
        rows.append(row)
    with (out/"whole_model.csv").open("w") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    (out/"digests.json").write_text(json.dumps({"train":{str(k):v for k,v in hashes.items()},"panels":panels},indent=2))
    return rows


if __name__ == "__main__":
    collect(Path(sys.argv[1]),Path(sys.argv[2]))
