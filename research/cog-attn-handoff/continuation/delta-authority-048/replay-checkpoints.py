"""Reload six pinned scope/TAC/CSM checkpoint sets and reproduce their held routes."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from trainlab.runner import verify_run

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"recipe"))
import bridge_scope
from scope import FAMILY_FEATURES, Scope, features

rows=[]
for m in json.loads((ROOT/"module-manifest.json").read_text()):
    run=ROOT/"ecosystem/research_lab/runs"/m["run_id"]
    assert verify_run(run)["valid"]
    for file,expected in m["checkpoints"].items():
        assert hashlib.sha256((run/"output"/file).read_bytes()).hexdigest()==expected
    seed=m["seed"]
    slot=bridge_scope.SEEDS.index(seed)
    tac=bridge_scope.TAC().eval()
    tac.load_state_dict(torch.load(run/"output/tac.pt",weights_only=True))
    reader=bridge_scope.reader.Reader(seed,True)
    with np.load(run/"output/csm.npz",allow_pickle=False) as data:
        reader.E,reader.a,reader.b=data["E"],data["a"],data["b"]
    learned=None
    if m["arm"]=="candidate":
        _,queries,_=bridge_scope.csm_data(seed+883,1000,FAMILY_FEATURES)
        entities={bridge_scope.reader.V[n] for n in bridge_scope.reader.WORDS}
        # Reconstruct DictVectorizer's sorted columns without loading Python
        # objects from the original names member.
        names=sorted({name for pair in queries for query in pair
                      for i,t in enumerate(query) if t in entities for name in features(query,i)})
        with np.load(run/"output/scope.npz",allow_pickle=False) as data:
            coeffs,bias=data["coef"],float(data["intercept"])
        assert len(names)==len(coeffs)
        learned=Scope(names,coeffs,bias)
        folder=ROOT/"portable"
        folder.mkdir(exist_ok=True)
        np.savez(folder/f"scope-{seed}.npz",names=np.array(names,dtype=str),coef=coeffs,intercept=bias)
    saved=json.loads((run/"output/results.json").read_text())["profiles"]
    for profile in saved:
        rebuilt=bridge_scope.held(seed,tac,reader,profile["profile"],(11,23,37)[slot],901+slot,learned)
        assert rebuilt==profile
    rows.append(dict(seed=seed,arm=m["arm"],cases=768))
assert len(rows)==6
(ROOT/"checkpoint-replay.json").write_text(json.dumps(rows,indent=2))
print(sum(r["cases"] for r in rows),"held module cases reproduced from checkpoint bytes")
