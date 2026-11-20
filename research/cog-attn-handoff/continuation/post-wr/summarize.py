import json,statistics
from pathlib import Path
rows=json.loads(Path("audit.json").read_text())
for fname in ("validation","kl-validation","matched-validation"):
    data=json.loads(Path(fname+".json").read_text())
    print(fname)
    for c in sorted({r["cohort"] for r in data}):
        subset=[r for r in data if r["cohort"]==c]
        print(c,[round(sum(x[k] for x in subset)/sum(x["n"] for x in subset)*100,2) for k in ("teacher","student")])
matched=[r for r in rows if r["series"]=="matched_trial.py"]
for family,gate in sorted({(r["family"],r["mode"]) for r in matched}):
    sub=[r for r in matched if r["family"]==family and r["mode"]==gate]
    print(family,gate,{k:sum(r[k] for r in sub) for k in ("errors","correct","asks","rollbacks","kv_hits")},sum(r["rtt_ns"] for r in sub)/sum(r["asks"] for r in sub)/1000)
for w in (1,10,50,90):
    for gate in ("0.8","0.995"):
        sub=[r for r in matched if r["mode"]==gate and r["ratio"]==w]
        speed=[next(x["ns"] for x in sub if x["family"]=="transformer" and x["seed"]==r["seed"])/r["ns"] for r in sub if r["family"]=="wr"]
        print(w,gate,[round(x,2) for x in speed],round(statistics.median(speed),2))
