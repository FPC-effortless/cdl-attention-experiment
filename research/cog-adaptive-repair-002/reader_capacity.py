"""Check bounded store behavior on held single-word names at increasing load."""
import argparse
import json
from pathlib import Path
import time
import statistics
import numpy as np
import torch

from structured_memory import StreamingStore
from structured_transfer import load_tagger


def stream(store, text):
    result = None
    for x in text.encode("ascii"):
        result = store.step(x)
    return result


def check(checkpoint):
    model = load_tagger(checkpoint)
    rng = np.random.default_rng(459041)
    names = set()
    while len(names)<40:
        names.add("".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz"),5)).title())
    people=sorted(names)
    result={}
    for n in (4,16,32,33,40):
        store=StreamingStore(model)
        expected={}
        t0=time.perf_counter()
        for i,name in enumerate(people[:n]):
            value=(3*i)%4
            stream(store, f"The log lists {name} at {value}. ")
            stream(store, f"A rumor puts {name} at {(value+1)%4}. ")
            expected[name]=value
        accepted=list(store.bank)
        timings=[];correct=0;absent=0
        for name,value in expected.items():
            start=time.perf_counter()
            y=stream(store,f"What digit does the log give for {name}?")
            timings.append(time.perf_counter()-start)
            should_hold=name in people[max(0,n-32):n]
            if should_hold and y==48+value:correct+=1
            if not should_hold and y is None:absent+=1
        last=people[n-1]
        stream(store,f"The log lists {last} at {(expected[last]+1)%4}.")
        revised=stream(store,f"What digit does the log give for {last}?")
        result[n]={"keys":len(accepted),"retained_correct":correct,"retained_expected":min(32,n),
            "evicted_abstentions":absent,"expected_evictions":max(0,n-32),
            "last_revised_correct":revised==48+(expected[last]+1)%4,
            "median_parser_query_ms":1e3*statistics.median(timings),"elapsed_s":time.perf_counter()-t0}
    return result


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("checkpoint",type=Path);p.add_argument("out",type=Path)
    a=p.parse_args();torch.set_num_threads(1)
    a.out.write_text(json.dumps(check(a.checkpoint),indent=2))
