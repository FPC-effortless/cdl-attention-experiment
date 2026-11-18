"""Paired multi-table retrieval diagnostic with observable candidate losses."""
import csv
import gzip
import json
import os
from pathlib import Path

import numpy as np
from panel import make, MASK, PACKETS


def signatures(values, width):
    dtype = np.uint16 if width == 16 else np.uint32
    weights = np.left_shift(dtype(1), np.arange(width, dtype=dtype))
    return (values[:, 12:12+width] >= 0).astype(dtype) @ weights


def index(keys, tables):
    return [[np.flatnonzero(((keys[:, 3*t:3*t+3] >= 0) @ MASK) == bank)
             for bank in range(8)] for t in range(tables)]


def candidates(query, banks, load):
    if not banks:
        return np.arange(load), load
    ordered = []
    seen = set()
    visits = 0
    for table, groups in enumerate(banks):
        coordinates = query[3*table:3*table+3]
        primary = int((coordinates >= 0) @ MASK)
        adjacent = primary ^ (1 << np.argmin(abs(coordinates)))
        for group in (groups[primary], groups[adjacent]):
            visits += len(group)
            for item in group:
                if int(item) not in seen:
                    seen.add(int(item))
                    ordered.append(item)
    return np.asarray(sorted(ordered), dtype=int), visits


def evaluate(seed, load, table_count, width, full, raw):
    _, keys, values, targets, queries, _ = make([seed, load, 1], load)
    banks = index(keys, table_count)
    packed = signatures(keys, width) if not full else None
    query_codes = signatures(queries, width) if not full else None
    arm = "full" if full else f"{table_count}tables-{width}bits"
    total = dict(routed=0, retained=0, correct=0, visits=0, unique=0, hamming=0, dots=0, overflows=0)
    events = []
    for cue, (target, query) in enumerate(zip(targets, queries)):
        union, visits = candidates(query, banks, load)
        subset = union
        hamming = 0
        if len(union) > 32 and not full:
            distances = np.bitwise_count(packed[union] ^ query_codes[cue])
            subset = union[np.argsort(distances, kind="stable")[:32]]
            hamming = len(union)
        prediction = int(subset[np.argmax(keys[subset] @ query)]) if len(subset) else -1
        routed, retained, correct = int(target in union), int(target in subset), int(prediction == target)
        assert correct <= retained <= routed
        assert full or len(subset) <= 32
        raw.writerow((seed, load, arm, cue, int(target), prediction, routed, retained, correct,
                      visits, len(union), hamming, len(subset)))
        events.append((routed, retained, correct))
        for name, value in zip(total, (routed,retained,correct,visits,len(union),hamming,len(subset),
                                      int(len(union)>32))):
            total[name] += value
    packet_events = np.asarray(events).reshape(PACKETS,4,3).all(axis=1).sum(axis=0)
    storage = keys.nbytes + values.nbytes
    extra = 0 if full else packed.nbytes + table_count * (load + 9) * 8
    return dict(seed=seed, load=load, arm=arm, **total,
                routed_packets=int(packet_events[0]),retained_packets=int(packet_events[1]),
                correct_packets=int(packet_events[2]), numeric_bytes=storage+extra)


def main():
    output = Path(os.environ["TRAINLAB_OUTPUT_DIR"])
    rows = []
    with gzip.open(output/"cues.csv.gz","wt") as stream:
        raw = csv.writer(stream,lineterminator="\n")
        raw.writerow("seed load arm cue target prediction routed retained correct visits unique hamming dots".split())
        for seed in (11,23,37):
            for load in (29,64,256):
                for count in (1,2,4,0):
                    for width in (16,32):
                        rows.append(evaluate(seed,load,count,width,False,raw))
                rows.append(evaluate(seed,load,0,16,True,raw))
    with (output/"scores.csv").open("w") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (output/"metrics.json").write_text(json.dumps({"panels":9}))


if __name__ == "__main__":
    main()
