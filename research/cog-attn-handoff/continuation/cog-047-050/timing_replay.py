"""Repeat measured end-to-end costs with unchanged checkpoints and paired arm order."""
import json
import random
import statistics
import sys
from pathlib import Path
import torch
from trainlab.runner import verify_run

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"recipe"))
from causal_v3 import Attention,LENGTHS,Local
from continual_v3 import runner

paths={}
for f in sorted((ROOT/"ecosystem/research_lab/runs").glob("*/output/results.json")):
    meta=json.loads((f.parents[1]/"experiment.json").read_text())
    if meta["command"][-2]!="continual_v3":
        continue
    assert verify_run(f.parents[1])["valid"]
    for cell in json.loads(f.read_text()):
        paths[cell["seed"],cell["mode"],cell["length"],cell["charge_ms"]]=(f.parent,cell)
records=[]
for seed in (947,971,997):
    models={}
    for mode in ("local","full","scoped"):
        folder,_=paths[seed,mode,128,0.]
        model=Local() if mode=="local" else Attention()
        model.load_state_dict(torch.load(folder/f"{mode}.pt",weights_only=True))
        models[mode]=model.eval()
    for length in LENGTHS:
        for delay in (0.,.002):
            for rep in range(3):
                order=list(models)
                random.Random(seed+length+rep).shuffle(order)
                with torch.no_grad():
                    for mode in order:
                        _,old=paths[seed,mode,length,delay*1000]
                        row=runner(models[mode],mode,seed,length,delay,old["fit_ns"])
                        assert json.loads(json.dumps(row["steps"]))==old["steps"]
                        assert row["kv_parity"]<=1e-5
                        assert row["total_ns"]==row["fit_ns"]+row["cold_ns"]+12*row["elapsed_ns"]
                        records.append(dict(rep=rep,**{k:v for k,v in row.items() if k!="steps"}))
assert len(records)==270
summary=[]
for length in LENGTHS:
    for delay in (0.,2.):
        for baseline in ("full","scoped"):
            ratios=[]
            stage=[]
            for seed in (947,971,997):
                matched=[r for r in records if (r["length"],r["charge_ms"],r["seed"])==(length,delay,seed)]
                local=[r for r in matched if r["mode"]=="local"]
                peer=[r for r in matched if r["mode"]==baseline]
                ratios.append(statistics.median(r["total_ns"] for r in peer)/
                              statistics.median(r["total_ns"] for r in local))
                stage.append(statistics.median(r["elapsed_ns"] for r in peer)/
                             statistics.median(r["elapsed_ns"] for r in local))
            summary.append(dict(length=length,delay=delay,baseline=baseline,total_ratio=statistics.median(ratios),
                                online_ratio=statistics.median(stage),paired_total=ratios,paired_online=stage))
(ROOT/"timing-replay.json").write_text(json.dumps(dict(records=records,summary=summary),indent=2))
for r in summary:
    print(r["length"],r["delay"],r["baseline"],round(r["total_ratio"],3),round(r["online_ratio"],3))
