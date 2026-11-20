"""Qualification of a quorum restricted to a separately acquired source set."""
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

from policy import adjust,reply_tape,score

ROOT=Path(__file__).resolve().parent
PROFILES=("hidden","outage","compromised_reserve")


def tape_for(seed,index,profile,truth):
    responses=reply_tape(seed+9000,index,"hidden",truth)
    rng=random.Random((seed+9000)*1000003+index*7919)
    shared_fault=rng.random()<.15
    fake=[rng.randrange(4),*rng.sample(range(3),2)]
    for reply in responses:
        if reply["service"]<4:
            continue
        if profile=="outage":
            reply.update(status="missing",value=None)
        if profile=="compromised_reserve" and shared_fault and reply["status"]!="missing":
            slot=reply["slot"]
            # The validator sees current metadata on a semantically wrong snapshot.
            val=fake[slot] if truth is None else truth[3-slot] if slot else truth[slot]
            reply.update(status="answer",value=val)
    return responses


def run(seed,arm):
    items=json.loads((ROOT/f"scores-{seed}.json").read_text())["items"]
    output=[]
    for profile in PROFILES:
        for i,item in enumerate(items):
            prior=np.array(item["posterior"])
            truth=item["truth"]
            tape=tape_for(seed,i,profile,truth)
            if arm=="candidate":
                tape=[r for r in tape if r["service"]>=4]
            decision=score(prior,tape,"quorum_roots" if arm=="candidate" else "quorum_labels")
            action=decision["action"]
            output.append(dict(mode="reserve" if arm=="candidate" else "public",profile=profile,i=i,cohort=item["cohort"],
                               wrong=action is not None and action!=truth,
                               effect_error=action is not None and adjust(action,item["state"])!=adjust(truth,item["state"]),
                               requests=len(decision["events"]),**decision))
    return dict(seed=seed,rows=output)


if __name__=="__main__":
    seed=int(os.environ["TRAINLAB_SEED"])
    result=run(seed,sys.argv[-1])
    out=Path(os.environ["TRAINLAB_OUTPUT_DIR"])
    data=json.dumps(result,separators=(",",":")).encode()
    (out/"results.json").write_bytes(data)
    (out/"results.sha256").write_text(hashlib.sha256(data).hexdigest())
    eligible=[r for r in result["rows"] if r["profile"]=="hidden" and r["cohort"]!="ambiguous"]
    score_value=sum(r["action"] is not None and not r["wrong"] for r in eligible)/len(eligible)
    (out/"metrics.json").write_text(json.dumps(dict(score=score_value)))
