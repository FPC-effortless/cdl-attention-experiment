"""Recompute aggregates and validate nested routing from retained per-cue observations."""
import json
from pathlib import Path
import sys
import pandas as pd

root = Path(sys.argv[1])
raw = pd.read_csv(root/"cues.csv.gz")
scores = pd.read_csv(root/"scores.csv")
ids = ["seed","load","arm"]
expected = {(seed, load, arm) for seed in (11,23,37) for load in (29,64,256)
            for arm in ["full", *[f"{tables}tables-{bits}bits" for tables in (0,1,2,4) for bits in (16,32)]]}
assert len(scores) == 81 and set(map(tuple,scores[ids].to_numpy())) == expected
assert set(map(tuple,raw[ids].drop_duplicates().to_numpy())) == expected
assert len(raw)==3*3*9*2048
assert not raw.duplicated(ids+["cue"]).any()
assert raw.groupby(ids).cue.nunique().eq(2048).all()
assert raw.correct.eq(raw.prediction.eq(raw.target)).all()
assert raw.correct.le(raw.retained).all() and raw.retained.le(raw.routed).all()
assert raw.loc[raw.arm.ne("full"),"dots"].le(32).all()
assert raw.groupby(["seed","load","cue"]).target.nunique().eq(1).all()
raw["overflows"]=raw.unique.gt(32).astype(int)
raw["packet"]=raw.cue//4
for row in scores.to_dict("records"):
    panel = raw[(raw.seed==row["seed"])&(raw.load==row["load"])&(raw.arm==row["arm"])]
    for field in ("routed","retained","correct","visits","unique","hamming","dots","overflows"):
        assert int(panel[field].sum())==row[field]
    for field in ("routed","retained","correct"):
        assert int(panel.groupby("packet")[field].all().sum())==row[field+"_packets"]
pivot = raw.pivot(index=["seed","load","cue"],columns="arm",values="routed")
for width in (16,32):
    assert (pivot[f"1tables-{width}bits"]<=pivot[f"2tables-{width}bits"]).all()
    assert (pivot[f"2tables-{width}bits"]<=pivot[f"4tables-{width}bits"]).all()
    assert pivot[f"0tables-{width}bits"].eq(1).all()
paired=[]
for (seed,load), group in raw.groupby(["seed","load"]):
    packets=group.groupby(["arm","packet"]).correct.all().unstack(0)
    for arm in packets:
        base=packets["1tables-16bits"]
        paired.append(dict(seed=int(seed),load=int(load),arm=arm,
                           improved=int((packets[arm]&~base).sum()),
                           worsened=int((~packets[arm]&base).sum()),
                           max_union=int(group[group.arm.eq(arm)].unique.max())))
print(json.dumps(paired,indent=2))
