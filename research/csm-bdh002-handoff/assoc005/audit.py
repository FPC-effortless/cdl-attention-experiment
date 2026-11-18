"""Audit saved integration counts, packet identities and paired timing coverage."""
from pathlib import Path
import json
import sys
import pandas as pd

root=Path(sys.argv[1])
s=pd.read_csv(root/"scores.csv")
t=pd.read_csv(root/"timing.csv")
ids=["seed","noise","arm","episode"]
assert len(s)==3*3*2*256 and not s.duplicated(ids).any()
assert s.groupby(["seed","noise","episode"]).gold.nunique().eq(1).all()
assert s.packet_exact.eq(s.gold.eq(s.decoded)).all()
assert (s.state_exact+s.rejected+s.wrong_commit).eq(1).all()
assert s[s.noise.eq(0)].state_exact.eq(1).all()
assert len(t)==3*3*12*2 and not t.duplicated(["seed","noise","block","arm"]).any()
assert t.ns.gt(0).all()
assert t.groupby(["seed","noise","block"]).position.nunique().eq(2).all()
timing=t.pivot(index=["seed","noise","block"],columns="arm",values="ns")
paired=timing.filtered/timing.full
report={"summary":s.groupby(["noise","arm"])[["packet_exact","state_exact","rejected","wrong_commit"]].sum().reset_index().to_dict("records"),
        "operations":s.groupby(["noise","arm","op"])[["state_exact","rejected","wrong_commit"]].sum().reset_index().to_dict("records"),
        "timing_ratio_filtered_over_full":{"median":float(paired.median()),"min":float(paired.min()),"max":float(paired.max())},
        "timing_us_per_packet":(t.groupby(["noise","arm"]).ns.median()/256/1000).reset_index().to_dict("records")}
def kind(token):
    return "operation" if token<4 else "register" if token<20 else "literal" if token<255 else "padding"
confusion={}
for row in s.itertuples():
    for field,(gold,decoded) in enumerate(zip(json.loads(row.gold),json.loads(row.decoded))):
        key=f"{row.noise}/{row.arm}/{field}/{kind(gold)}/{kind(decoded)}/"+("exact" if gold==decoded else "changed")
        confusion[key]=confusion.get(key,0)+1
report["token_confusions"]=confusion
print(json.dumps(report,indent=2,sort_keys=True))
