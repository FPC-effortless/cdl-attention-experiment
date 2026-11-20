"""TrainLab recipe over frozen model score banks and paired reply tapes."""
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np

from policy import MODES,PROFILES,adjust,reply_tape,score

ROOT=Path(__file__).resolve().parent
CUTOFFS=(.8,.9,.95,.975,.99,.995,.999,.9999,1.01)


def calibrate(samples):
    valid=[]
    for cutoff in CUTOFFS:
        chosen=[]
        for item in samples:
            post=np.array(item["posterior"])
            tape=reply_tape(item["seed"],item["index"],"exact",item["truth"])
            ret=score(post,tape,"staged_roots",cutoff)
            if ret["action"] is not None:
                chosen.append(ret["action"]==item["truth"])
        if (len(chosen)-sum(chosen))*100<=len(chosen):
            valid.append((len(chosen),cutoff))
    n,cutoff=max(valid)
    return dict(n=n,cutoff=cutoff,total=len(samples))


def run(seed,arm):
    bank=json.loads((ROOT/f"scores-{seed}.json").read_text())
    assert bank["seed"]==seed
    calibration=calibrate(json.loads((ROOT/"prior-calibration.json").read_text()))
    output=[]
    for mode in MODES[arm]:
        for profile in PROFILES:
            for i,item in enumerate(bank["items"]):
                prior=np.array(item["posterior"])
                truth=item["truth"]
                tape=reply_tape(seed,i,profile,truth)
                decision=score(prior,tape,mode,calibration["cutoff"],truth)
                answer=decision["action"]
                wrong=answer is not None and answer!=truth
                actual=adjust(answer,item["state"])
                expected=adjust(truth,item["state"])
                # The upper bound deliberately consumes three typed fields.
                requests=3 if mode=="verified" else len(decision["events"])
                output.append(dict(mode=mode,profile=profile,i=i,cohort=item["cohort"],wrong=wrong,
                                  effect_error=answer is not None and actual!=expected,requests=requests,**decision))
    return dict(seed=seed,calibration=calibration,rows=output)


if __name__=="__main__":
    seed=int(os.environ["TRAINLAB_SEED"])
    arm=sys.argv[-1]
    result=run(seed,arm)
    path=Path(os.environ["TRAINLAB_OUTPUT_DIR"])
    data=json.dumps(result,separators=(",",":")).encode()
    (path/"results.json").write_bytes(data)
    (path/"results.sha256").write_text(hashlib.sha256(data).hexdigest())
    mode="quorum_roots" if arm=="candidate" else "quorum_labels"
    observed=[r for r in result["rows"] if r["mode"]==mode and r["profile"]=="shared" and r["cohort"]!="ambiguous"]
    score_value=sum(r["action"] is not None and not r["wrong"] for r in observed)/len(observed)
    (path/"metrics.json").write_text(json.dumps(dict(score=score_value)))
