"""Validate matched exposure and summarize per-seed grouping effects."""
import json
from pathlib import Path
import statistics
import sys

def summarize(directory,arms=("C","GQA","GRU")):
    summary={}
    for arm in arms:
        groups={c:[] for c in ("grouped","mixed")}
        for seed in (9731,9743,9769,9791):
            for condition,rows in groups.items():
                d=json.loads((directory/f"{arm}-{seed}-{condition}.json").read_text())
                assert (arm,seed,condition,256)==(d["arm"],d["seed"],d["condition"],d["steps"])
                rows.append(d)
            a,b=(rows[-1] for rows in groups.values())
            assert all(a[k]==b[k] for k in ("pool_sha","replay_sha"))
            assert all(a["transfer"][p]["sha256"]==b["transfer"][p]["sha256"] for p in a["transfer"])
        summary[arm]={}
        for c,rows in groups.items():
            summary[arm][c]={"bpb":statistics.mean(r["bpb"] for r in rows)}
            for domain in ("symbol","english"):
                summary[arm][c][domain]={metric:[r[domain][metric] for r in rows] for metric in ("grids","single","mean_margin")}
            summary[arm][c]["transfer"]={p:[r["transfer"][p]["both_correct"] for r in rows] for p in rows[0]["transfer"]}
    return summary

if __name__=="__main__":
    print(json.dumps(summarize(Path(sys.argv[1]),sys.argv[2:] or ("C","GQA","GRU")),indent=2))
