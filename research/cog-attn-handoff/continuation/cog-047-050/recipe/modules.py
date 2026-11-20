"""Bridge original frozen artifacts into the versioned store without changing weights."""
import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from original_lrc.model import ContentAddressedLRC
from original_lrc.bridge_data import bridge_dataset
from state_store import World

ROOT=Path(__file__).resolve().parent


def csm():
    path=ROOT/"original_csm"
    sys.path.insert(0,str(path))
    reader=importlib.import_module("reader")
    roles=importlib.import_module("roles")
    results=[]
    for seed in (11,23,37):
        model=reader.Reader(seed,True)
        with np.load(path/f"recurrent-{seed}.npz",allow_pickle=False) as weights:
            model.E,model.a,model.b=[weights[key] for key in ("E","a","b")]
        with np.load(path/f"commit-{seed}.npz",allow_pickle=False) as weights:
            gate=weights["predicate"]
        records,queries,truth,_=reader.data(93000+seed,128,"held")
        vals=model.forward(records)[0].reshape(-1,8,2,32).sum(2)
        q=model.forward(queries)[0]
        selected=(q@vals.transpose(0,2,1)).argmax(2)
        allowed=0
        world=World()
        for ix,(guess,label) in enumerate(zip(selected.flatten(),truth.flatten(),strict=True)):
            cells=[None]*10
            cells[int(label)]=("v",4)
            cells[8],cells[9]=("v",7),("v",3)
            allow,result,observed=roles.outcome(roles.State(tuple(cells)),"copy",int(guess),gate)
            if allow:
                assert world.commit(f"csm:{ix}",world.epoch(),{"observation":observed})=="committed"
                allowed+=1
        results.append(dict(seed=seed,n=truth.size,correct=int((selected==truth).sum()),commits=allowed,
                            predicted=selected.flatten().tolist(),truth=truth.flatten().tolist()))
    return results


def lrc():
    torch.set_num_threads(1)
    path=ROOT/"original_lrc"
    output=[]
    for seed in (901,902,903):
        file=path/f"repair_{seed}.pt"
        digest=hashlib.sha256(file.read_bytes()).hexdigest()
        assert digest==json.loads((path/f"repair_{seed}.receipt.json").read_text())["checkpoint_sha256"]
        model=ContentAddressedLRC().eval()
        model.load_state_dict(torch.load(file,weights_only=True)["model"])
        batches=bridge_dataset(92000+seed,64)
        with torch.inference_mode():
            out=model(batches["records"],batches["query"])
        answers=out.logits.argmax(-1)
        world=World()
        admitted=0
        for i,(pred,route) in enumerate(zip(answers,out.selected,strict=True)):
            if (route>=0).all():
                world.commit(f"lrc:{i}",0,{"result":int(pred)})
                admitted+=1
        output.append(dict(seed=seed,n=len(answers),exact=int((answers==batches["answer"]).sum()),
                   held_paths=int((out.selected==batches["support"]).all(1).sum()),
                   admitted=admitted,sha256=digest,pred=answers.tolist(),truth=batches["answer"].tolist(),
                   selected=out.selected.tolist(),gold_support=batches["support"].tolist()))
    return output


def run():
    return dict(lrc=lrc(),csm=csm())
