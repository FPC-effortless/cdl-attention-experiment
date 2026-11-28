"""Reconstruct the training pools without a model; cross-check labels, digests and held coverage."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from train import batches, blocks, cases
from repair_objective import trace_targets

def audit(root,arms,four):
    for seed in (9731,9743,9769,9791):
        checksum=hashlib.sha256()
        streams={d:blocks(seed,d,four) for d in ("symbol","english")}
        held={h for d in streams for a,b,_,_ in cases(192,957427,True,d,independent=True) for h in (a,b)}
        histories=set()
        for section in range(64):
            domain=("symbol","english")[section%2]
            block=next(streams[domain])
            if not four:
                for j in range(0,8,2):
                    if block[j][0][1]==block[j+1][0][1]:
                        block[j+1]=block[j+1][2:]+block[j+1][:2]
            def history(text):
                return text.split("#")[0] if domain=="symbol" else text.split("What digit")[0]
            for world in block:
                hs={history(text) for text,_ in world}
                assert len(hs)==2 and hs.isdisjoint(histories) and hs.isdisjoint(held)
                histories.update(hs)
            a,b=batches(block,True),batches(block,False)
            for left,right in zip(a,b):
                assert len({history(t) for t,_ in left})==4
                assert len({history(t) for t,_ in right})==8
                assert Counter(t.split("#")[-1] if domain=="symbol" else t[-4:] for t,_ in left)==Counter(t.split("#")[-1] if domain=="symbol" else t[-4:] for t,_ in right)
                assert Counter(v for _,v in left)==Counter(v for _,v in right)
            for text,v in sorted(sum(a,[])):
                assert trace_targets(text.encode())[1]==v
                checksum.update(text.encode()+bytes([v]))
        for arm in arms:
            for c in ("grouped","mixed"):
                d=json.loads((root/f"{arm}-{seed}-{c}.json").read_text())
                assert d["pool_sha"]==checksum.hexdigest()
    print("Verified training labels, conditional update composition, disjoint held histories and model-reported digests.")

if __name__=="__main__":
    root=Path(sys.argv[1])
    audit(root/"results",("C","GQA","GRU"),False)
    audit(root/"balanced",("C",),True)
