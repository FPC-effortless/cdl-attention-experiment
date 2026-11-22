"""Read-only prediction worker. Never takes truth or an optimizer state."""
import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys

import torch

from .role_judge import checked, require
from .role_mixers import ARMS, build_arm
from .objective import Tokenizer

def predict(source, source_sha256, weights, weight_sha256):
    actor = checked(source, source_sha256)
    require(set(actor) == {"schema","prompts"} and
            actor["schema"] == "cog-comm-role-001.actor.v1", "wrong prediction projection")
    raw = weights.read_bytes()
    require(sha256(raw).hexdigest() == weight_sha256, "checkpoint digest differs")
    ck = torch.load(BytesIO(raw), map_location="cpu", weights_only=True)
    require(ck["schema"] == "cog-comm-role-001.checkpoint.v1" and ck["format"]==1 and
            ck["plan"]["arm"] in ARMS and ck["seed"] in (1171,1271,1371) and
            ck["seed"] == ck["plan"]["seed"] and ck["updates"] in (128,1280),
            "wrong model identity")
    tok = Tokenizer(tuple(ck["tokenizer_words"]))
    require(tok.size == ck["vocab_size"] == 287, "tokenizer mismatch")
    m = build_arm(tok.size,ck["plan"]["arm"],ck["seed"]).eval()
    require(all(w.dtype == torch.float32 and bool(torch.isfinite(w).all())
                for w in ck["state_dict"].values()), "invalid checkpoint tensor")
    m.load_state_dict(ck["state_dict"], strict=True)
    prompts = actor["prompts"]
    require(len(prompts) in (960,2560) and len(set(prompts)) == len(prompts),
            "wrong or duplicate actor prompts")
    ids = [tok.encode(s)[1:-1] for s in prompts]
    require(all(len(p)==7 and min(p)>=262 for p in ids), "unexpected encoding")
    out = []
    with torch.inference_mode():
        for i in range(0,len(ids),8):
            v,_ = m(torch.tensor(ids[i:i+8], dtype=torch.long), state=None)
            v = v[:,-1]
            require(v.shape == (8,287) and bool(torch.isfinite(v).all()), "wrong output")
            out.extend(v.tolist())
    return dict(actor_sha256=source_sha256,checkpoint_sha256=weight_sha256,
                seed=ck["seed"],arm=ck["plan"]["arm"],updates=ck["updates"],
                vocab_size=tok.size, tokenizer_words=list(tok.words),logits=out)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("actor","checkpoint"):
        p.add_argument("--"+k,type=Path,required=True)
        p.add_argument("--"+k+"-sha256",required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    require(sys.version_info[:2] == (3,12) and torch.__version__ == "2.10.0+cpu",
            "unsupported runtime")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    result = predict(a.actor,a.actor_sha256,a.checkpoint,a.checkpoint_sha256)
    with a.out.open("x") as f:
        json.dump(result,f,separators=(",",":"),allow_nan=False)
        f.write("\n")
if __name__=="__main__":
    main()
