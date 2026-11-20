"""Warm process, cold answer cache. Every write reaches the candidate's synchronous hook."""
import hashlib
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

from investigation_world.benchmark.policies import PublicEvidenceReferencePolicy
from research_lab.ecosystem import _fixture

N = 16
LENGTH = 2048
KINDS = ("revision", "swap", "projection", "lower_revision")


def trace(seed, write_percent):
    rng = random.Random(seed * 100 + write_percent)
    return [(rng.randrange(N), rng.choice(KINDS) if rng.randrange(100) < write_percent else "read",
             rng.randint(1,99)) for _ in range(LENGTH)]


def episode(seed, key):
    return _fixture(seed * 300 + key * 3 + key % 2)


def inputs(seed, padding):
    payloads = []
    for key in range(N):
        payload = episode(seed,key).public_payload()
        source, projection = payload["records"]
        field = next(iter(source["fields"]))
        backup = dict(source, record_id=source["record_id"]+"-B", fields={field:source["fields"][field]+200})
        fillers = [dict(projection, object_id=f"OTHER-{i}") for i in range(padding)]
        payload["records"] = fillers + [source,backup,projection]
        payloads.append(payload)
    return payloads


def dependency(payload):
    task = payload["task"]
    kind = "policy_rule" if task["task_type"]=="INVESTIGATE_AUTHORITY_BREACH" else "carrier_manifest"
    winner = next(row for row in payload["records"] if row["object_id"]==task["target_object_id"]
                  and row["record_type"]==kind)
    return winner["record_id"], tuple(winner["fields"].items())


def execute(payloads, operations, guarded):
    start = time.perf_counter_ns()
    policy = PublicEvidenceReferencePolicy()
    cache = {}
    dependencies = [dependency(p) for p in payloads] if guarded else []
    results = []
    hits = 0
    for key,kind,delta in operations:
        payload = payloads[key]
        records = payload["records"]
        if kind=="read":
            if guarded and key in cache:
                answer = cache[key]
                hits += 1
            else:
                answer = policy(payload)
                if guarded:
                    cache[key] = answer
            results.append(answer)
            continue
        if kind=="swap":
            records[-3],records[-2] = records[-2],records[-3]
        else:
            row = records[{"revision":-3,"lower_revision":-2,"projection":-1}[kind]]
            field = next(iter(row["fields"]))
            row["fields"][field] += delta
        if guarded:
            current = dependency(payload)
            if current != dependencies[key]:
                cache.pop(key,None)
                dependencies[key] = current
    elapsed = time.perf_counter_ns()-start
    return elapsed,hits,results


def main():
    seed = int(os.environ["TRAINLAB_SEED"])
    guarded = sys.argv[1]=="candidate"
    cells = []
    for padding in (0,128):
        for percent in (1,10,50,90):
            ops = trace(seed,percent)
            timings = []
            for _ in range(5):
                elapsed,hits,answers = execute(inputs(seed,padding),ops,guarded)
                timings.append(elapsed)
            cells.append(dict(padding=padding,percent=percent,times_ns=timings,hits=hits,
                              answers=[a.model_dump(mode="json") for a in answers]))
    out = Path(os.environ["TRAINLAB_OUTPUT_DIR"])
    (out/"results.json").write_text(json.dumps(cells,separators=(",",":")))
    responses = json.dumps([c["answers"] for c in cells],sort_keys=True).encode()
    (out/"responses.sha256").write_text(hashlib.sha256(responses).hexdigest())
    # Primary endpoint is throughput in the predeclared sparse-write, padded workload.
    primary = next(c for c in cells if c["percent"]==10 and c["padding"]==128)
    (out/"metrics.json").write_text(json.dumps({"score":LENGTH*1e9/statistics.median(primary["times_ns"])}))


if __name__=="__main__":
    main()
