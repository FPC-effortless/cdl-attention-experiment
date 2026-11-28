"""Validate completeness and report matched-seed means."""
import json
from pathlib import Path
import statistics
import sys

def summarize(root):
    arms=("C","write","read","both","GQA","GRU")
    seeds=(9731,9743,9769,9791)
    records={(a,s):json.loads((root/f"{a}-{s}.json").read_text()) for a in arms for s in seeds}
    for s in seeds:
        rows=[records[a,s] for a in arms]
        assert len({r["training_sha256"] for r in rows})==1
        assert len({r["parameters"] for r in rows[:4]})==1
        for a,r in zip(arms,rows):
            assert (r["arm"],r["seed"],r["steps"])==(a,s,256)
            assert r["profile"]["batch_vs_stream_max_abs_logit_error"]<0.002
    summary={}
    for a in arms:
        rows=[records[a,s] for s in seeds]
        mean=statistics.mean
        summary[a]=dict(symbol_grids=[r["symbol"]["grids"] for r in rows],
                        english_grids=[r["english"]["grids"] for r in rows],
                        symbol_single=mean(r["symbol"]["single"] for r in rows),
                        english_single=mean(r["english"]["single"] for r in rows),
                        symbol_margin=[r["symbol"]["mean_margin"] for r in rows],
                        english_margin=[r["english"]["mean_margin"] for r in rows],
                        bpb=mean(r["bpb"] for r in rows),
                        state=rows[0]["profile"]["final_carried_tensor_state"]["logical_bytes"],
                        median_step_ms=statistics.median(r["profile"]["decode_step_median_ms"] for r in rows),
                        confirmation={k:[r["confirmation"][k]["both_correct"] for r in rows] for k in rows[0]["confirmation"]})
    return summary

if __name__=="__main__":
    print(json.dumps(summarize(Path(sys.argv[1])),indent=2))
