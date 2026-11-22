"""Pure-standard-library readout audit for the public synthetic role panel."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path

from .role_control import CELLS, COLORS, families, oracle, world

def checked(path, digest):
    data = path.read_bytes()
    if sha256(data).hexdigest() != digest:
        raise ValueError("pinned input digest mismatch")
    return json.loads(data)

def require(test, why):
    if not test:
        raise ValueError(why)

def score(panel, packet):
    require(set(panel) == {"panel","families"} and panel["panel"] in ("fit","control"),
            "invalid judge projection")
    keys = families()[panel["panel"]]
    require(len(panel["families"]) == len(keys), "wrong family count")
    words = packet["tokenizer_words"]
    vocab = packet["vocab_size"]
    require(type(words) is list and len(words) == len(set(words)) and
            all(c in words for c in COLORS) and vocab == 262 + len(words), "wrong output ID space")
    ix = {w:i+262 for i,w in enumerate(words)}
    logits = packet["logits"]
    require(len(logits) == 8 * len(keys), "incomplete predictions")
    out = {g:Counter() for g in ("all","seen_name_pair","unseen_name_pair")}
    for j,(key,row) in enumerate(zip(keys,panel["families"])):
        require(row["key"] == list(map(list,key)) and row["rows"] == list(range(j*8,j*8+8)),
                "judge did not match a unique ordered family")
        a,b = key[1]
        exact, rel, margins = [], [], []
        expected = [oracle(world(key,c)) for c in CELLS]
        require(row["answers"] == expected, "wrong judge answer")
        for i,y in zip(row["rows"],expected):
            values = logits[i]
            require(len(values) == vocab and all(type(v) in (int,float) and math.isfinite(v)
                                                 for v in values), "missing/nonfinite logits")
            m = values[ix[a]] - values[ix[b]]
            require(math.isfinite(m), "nonfinite margin")
            margins.append(m)
            exact.append(values[ix[y]] > max(values[:ix[y]]+values[ix[y]+1:]))
            rel.append(m > 0 if y == a else m < 0)
        require(all(not f or r for f,r in zip(exact,rel)), "score containment")
        record = dict(families=1, worlds=8, exact_worlds=sum(exact), exact_families=all(exact),
                      relative_worlds=sum(rel), relative_families=all(rel),
                      relative_ties=sum(m==0 for m in margins),
                      query_both_exact=sum(exact[i] and exact[i+1] for i in (0,2,4,6)),
                      order_both_exact=sum(exact[i] and exact[i+2] for i in (0,1,4,5)))
        for label,power in [("role",lambda s,o,q:s*q),("position", lambda s,o,q:s*o),
                             ("identity",lambda s,o,q:s),("bias",lambda s,o,q:1)]:
            record[label+"_contrast_sum"] = math.fsum(m*power(*c)/8 for m,c in zip(margins,CELLS))
        for g in ("all","seen_name_pair" if row["familiar_name_pair"] else "unseen_name_pair"):
            out[g].update(record)
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("panel","prediction"):
        p.add_argument("--"+k,type=Path,required=True)
        p.add_argument("--"+k+"-sha256",required=True)
    p.add_argument("--expected-actor-sha256",required=True)
    p.add_argument("--expected-checkpoint-sha256",required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    packet=checked(a.prediction,a.prediction_sha256)
    require(packet["actor_sha256"] == a.expected_actor_sha256 and
            packet["checkpoint_sha256"] == a.expected_checkpoint_sha256,
            "prediction identity mismatch")
    result=score(checked(a.panel,a.panel_sha256), packet)
    with a.out.open("x") as f:
        json.dump(result, f, sort_keys=True, allow_nan=False)
        f.write("\n")

if __name__ == "__main__":
    main()
