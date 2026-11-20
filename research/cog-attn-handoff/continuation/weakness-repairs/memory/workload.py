"""Persistent public CompanyWorld values updated by raw commands interpreted by frozen checkpoints."""
import hashlib
import json
import os
import random
import statistics
import sys
import time
from collections import deque
from pathlib import Path

from grounding import dataset, features, np, predict
from investigation_world.benchmark.policies import PublicEvidenceReferencePolicy
from research_lab.ecosystem import _fixture

ROOT = Path(__file__).resolve().parent
CHECKPOINT = {181:13, 193:37, 211:61}
ARMS = {"control":("fresh","semantic","eager","unassisted"),
        "candidate":("lazy","delayed","versioned")}
LENGTH = 1536


def setup(seed):
    fit = CHECKPOINT[seed]
    spec = json.loads((ROOT/f"data-32-{fit}.json").read_text())
    with np.load(ROOT/f"analytic-32-{fit}.npz") as ck:
        weights = [ck[f"arr_{i}"] for i in range(3)]
    bank = []
    for i,cohort in enumerate(("familiar","held_pair","recombine","novel_verb")):
        strings, _, _, actions = dataset(20000+seed*4+i,64,32,np.array(spec["aliases"]),cohort)
        bank.extend(zip(strings, map(int,actions)))
    return weights,spec["vocabulary"],bank


def trace(seed,ratio):
    rng = random.Random(seed*100+ratio)
    ops = []
    for _ in range(LENGTH):
        if rng.randrange(100)<ratio:
            # Recurrent commands and a longer tail; hot pool includes all four cohorts.
            idx = rng.randrange(4)*64 + (rng.randrange(4) if rng.random()<.75 else rng.randrange(64))
            ops.append(("write",idx,rng.random()<.25))
        else:
            ops.append(("read",rng.randrange(32),False))
    return ops


def fixture(seed,key):
    return _fixture(seed*300+key*3+key%2)


def world(seed):
    payloads = []
    for key in range(32):
        payload = fixture(seed,key).public_payload()
        source,projection = payload["records"]
        field = next(iter(source["fields"]))
        source["fields"][field] = key%4
        payload["records"] = [dict(projection,object_id=f"EXTRA-{i}") for i in range(128)]+[source,projection]
        payloads.append(payload)
    return payloads


def digest_authority(payload):
    target = payload["task"]["target_object_id"]
    winner = next(r for r in payload["records"] if r["object_id"]==target and r["record_type"]!="system_projection")
    return winner["record_id"],tuple(winner["fields"].items())


def run(mode, ops, payloads, weights, vocab, bank):
    start = time.perf_counter_ns()
    policy = PublicEvidenceReferencePolicy()
    semantic, answers = {}, {}
    deps = [digest_authority(p) for p in payloads] if mode=="eager" else []
    versions = [0]*32
    notices = deque()
    state = [k%4 for k in range(32)]
    queries, writes = [], []
    hits = compiled = asks = 0
    for step,(kind,key,lost) in enumerate(ops):
        while notices and notices[0][0]<=step:
            _,dirty = notices.popleft()
            answers.pop(dirty,None)
        if kind=="write":
            command,gold = bank[key]
            if mode!="fresh" and command in semantic:
                action = semantic[command]
                compiled += 1
            else:
                x,_ = features([command],vocab)
                s,d,o = predict(weights,x)
                action = int(o.argmax()*1024+s.argmax()*32+d.argmax())
                confidence = float(o.max()*s.max()*d.max())
                if confidence<.80 and mode!="unassisted":
                    action = gold  # Simulated truthful clarification for this invocation.
                    asks += 1
                elif confidence>=.80 and mode!="fresh":
                    semantic[command] = action
            op,pair = divmod(action,1024)
            src,dst = divmod(pair,32)
            value = (state[src]+op*state[dst])%4
            state[dst] = value
            fields = payloads[dst]["records"][-2]["fields"]
            fields[next(iter(fields))] = value
            if mode=="versioned":
                versions[dst] += 1
            if mode=="eager":
                new = digest_authority(payloads[dst])
                if new!=deps[dst]:
                    answers.pop(dst,None)
                    deps[dst]=new
            if mode=="lazy":
                answers.pop(dst,None)
            if mode in ("delayed","versioned") and not lost:
                notices.append((step+8,dst))
            writes.append(action)
        else:
            entry = answers.get(key)
            if entry and (mode!="versioned" or entry[0]==versions[key]):
                result = entry[1]
                hits += 1
            else:
                result = policy(payloads[key])
                if mode in ("eager","lazy","delayed","versioned"):
                    answers[key] = versions[key],result
            queries.append(result)
    elapsed = time.perf_counter_ns()-start
    return dict(ns=elapsed,hits=hits,compiled=compiled,asks=asks,actions=writes,
                reads=[r.model_dump(mode="json") for r in queries])


def main():
    seed = int(os.environ["TRAINLAB_SEED"])
    weights,vocab,bank = setup(seed)
    rows = []
    for ratio in (1,10,50,90):
        ops = trace(seed,ratio)
        timings = {m:[] for m in ARMS[sys.argv[1]]}
        rng = random.Random(seed+ratio)
        last = {}
        for _ in range(3):
            modes = list(timings)
            rng.shuffle(modes)
            for mode in modes:
                last[mode] = run(mode,ops,world(seed),weights,vocab,bank)
                timings[mode].append(last[mode].pop("ns"))
        for mode,counts in last.items():
            rows.append(dict(mode=mode,ratio=ratio,times=timings[mode],**counts))
    primary = "lazy" if sys.argv[1]=="candidate" else "semantic"
    cell = next(r for r in rows if r["mode"]==primary and r["ratio"]==10)
    out = Path(os.environ["TRAINLAB_OUTPUT_DIR"])
    (out/"results.json").write_text(json.dumps(rows,separators=(",",":")))
    stable = sorted([(r["mode"],r["ratio"],r["actions"],r["reads"]) for r in rows])
    (out/"responses.sha256").write_text(hashlib.sha256(json.dumps(stable,sort_keys=True).encode()).hexdigest())
    (out/"metrics.json").write_text(json.dumps({"score":LENGTH*1e9/statistics.median(cell["times"])}))


if __name__=="__main__":
    main()
