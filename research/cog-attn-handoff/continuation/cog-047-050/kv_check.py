"""Independent output replay on every corrected multi-head fixture turn."""
import json
import sys
from pathlib import Path

import torch
from trainlab.runner import verify_run

root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/"recipe"))
from causal_v3 import Attention,LENGTHS,data,decode

runs=root/"ecosystem/research_lab/runs"
report=[]
for entry in runs.glob("*/experiment.json"):
    exp=json.loads(entry.read_text())
    if exp["command"][-2:]!=["continual_v3","control"]:
        continue
    run=entry.parent
    assert verify_run(run)["valid"]
    seed=exp["seed"]
    for mode in ("full","scoped"):
        model=Attention().eval()
        model.load_state_dict(torch.load(run/"output"/(mode+".pt"),weights_only=True))
        for length in LENGTHS:
            x,_=data(torch.Generator().manual_seed(seed+4300),48,length)
            prefix=x[:1,:-4].clone()
            maximum=0.
            with torch.inference_mode():
                cached=model.cache(prefix) if mode=="full" else None
                for row in x[:,-4:]:
                    z=model.cached(row[None],cached)
                    q=model(torch.cat((prefix,row[None]),1) if cached is not None else row[None])
                    maximum=max(maximum,float((z-q).abs().max()))
                    assert decode(z)==decode(q)
            assert maximum<=1e-5
            report.append(dict(mode=mode,seed=seed,length=length,max_abs=maximum,turns=48))
assert len(report)==30
(root/"kv-audit.json").write_text(json.dumps(report,indent=2))
print("Validated",sum(r["turns"] for r in report),"decoder turns. Max difference",
      max(r["max_abs"] for r in report))
