"""Label-isolated prediction and exact candidate trace for one frozen arm."""
from collections import defaultdict
import argparse
import json
from pathlib import Path

import torch
from .model import build_decoder, set_routing
from .objective import Tokenizer
from .data import ANSWERS


def read_actor(path):
    rows=[json.loads(x) for x in path.read_text().splitlines()]
    expected={"family_id","world_index","prompt"}
    if (not rows or any(set(x)!=expected or type(x["world_index"]) is not int or
                        x["world_index"] not in (0,1) or not isinstance(x["prompt"],str)
                        or not x["prompt"].strip() for x in rows)):
        raise ValueError("strict raw-text actor file required")
    k=[(r["family_id"],r["world_index"]) for r in rows]
    if len(k)!=len(set(k)) or any({w for f,w in k if f==g}!={0,1} for g in {f for f,w in k}):
        raise ValueError("incomplete or duplicate actor pair")
    return rows


@torch.inference_mode()
def predict_rows(model, tokenizer, actor, batch_size=8, max_prompt_tokens=512):
    """IDs only join output. No answer labels or locations in the actor closure."""
    model.eval()
    groups=defaultdict(list)
    traces={}
    callbacks=[]
    for i, layer in enumerate(model.mixers):
        if layer.arm!="C":
            continue
        def intercept(_,args,kw,ret,*,level=i,l=layer):
            # Retain exact output/state equality even when tapping route indices.
            x,s=args
            y,_,cand=l.route.forward_with_candidates(x,s,**kw)
            torch.testing.assert_close(y,ret[0],rtol=0,atol=0)
            traces[level]=cand[:,-1].reshape(cand.size(0),4).clone()
        callbacks.append(layer.route.register_forward_hook(intercept,with_kwargs=True))
    try:
        for idx,row in enumerate(actor):
            x=tokenizer.encode(row["prompt"])
            if len(x)>max_prompt_tokens:
                raise ValueError("actor prompt exceeds declared budget; no truncation")
            groups[len(x)].append((idx,x))
        records={}
        colors=[tokenizer.index[a] for a in ANSWERS]
        for length,rows in groups.items():
            for i in range(0,len(rows),batch_size):
                slate=rows[i:i+batch_size]
                xx=torch.tensor([t for _,t in slate],dtype=torch.long)
                set_routing(model,True)
                traces.clear()
                base,_=model(xx)
                selected={k:v.tolist() for k,v in traces.items()}
                if any(l.arm=="C" for l in model.mixers):
                    set_routing(model,False); traces.clear()
                    ablated,_=model(xx)
                else:
                    ablated=None
                for j,(idx,_) in enumerate(slate):
                    ll=base[j,-1]
                    out={k:actor[idx][k] for k in ("family_id","world_index")}
                    out.update(prompt_length=length,prediction=int(ll.argmax()),
                               answer_scores=ll[colors].tolist(),
                               route_positions={str(k):v[j] for k,v in selected.items()})
                    if ablated is not None:
                       al=ablated[j,-1]
                       out.update(no_route_prediction=int(al.argmax()),no_route_scores=al[colors].tolist())
                    records[idx]=out
        return [records[i] for i in range(len(actor))]
    finally:
        for h in callbacks:
            h.remove()
        set_routing(model,True)


def save_rows(path, rows):
    with path.open("x",encoding="utf8") as f:
        for r in rows:
            f.write(json.dumps(r,sort_keys=True,allow_nan=False,separators=(",",":"))+"\n")


def main():
    import hashlib
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint","actor","output"):
        parser.add_argument("--"+name,required=True,type=Path)
    for name in ("checkpoint-sha256","actor-sha256"):
        parser.add_argument("--"+name,required=True)
    a=parser.parse_args()
    for file,expected in ((a.checkpoint,a.checkpoint_sha256),(a.actor,a.actor_sha256)):
        with file.open("rb") as f:
            if hashlib.file_digest(f,"sha256").hexdigest()!=expected:
                raise ValueError("prediction input digest mismatch")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    c=torch.load(a.checkpoint,weights_only=True,map_location="cpu")
    if c.get("format")!=1 or c.get("updates")!=c["plan"]["steps"]:
        raise ValueError("checkpoint is not the prescribed endpoint")
    t=Tokenizer(tuple(c["tokenizer_words"]))
    if t.size!=c["vocab_size"]:
        raise ValueError("checkpoint tokenizer geometry mismatch")
    plan=c["plan"]
    m=build_decoder(t.size,plan["arm"],
                    **{n:plan[n] for n in ("width","layers","block","seed","route_mode","retained_blocks")}).eval()
    m.load_state_dict(c["state_dict"],strict=True)
    out=predict_rows(m,t,read_actor(a.actor),max_prompt_tokens=plan["max_tokens"])
    save_rows(a.output,out)


if __name__=="__main__":
    main()
