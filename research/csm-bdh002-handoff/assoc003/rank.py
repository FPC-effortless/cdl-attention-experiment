import csv
import os
from pathlib import Path
import numpy as np
from panel import make, MASK, PACKETS


def signatures(values):
    return ((values[:, 3:19] >= 0).astype(np.uint16) @
            (1 << np.arange(16, dtype=np.uint16))).astype(np.uint16)


def measure(seed, load):
    _, keys, _, targets, queries, _ = make([seed,load,1], load)
    bank_id = (keys[:,:3] >= 0) @ MASK
    banks = [np.flatnonzero(bank_id == i) for i in range(8)]
    stored = signatures(keys)
    routed = []
    survived = [[], []]
    correct = [[], []]
    dot_count = hamming_count = 0
    for q, signature, target in zip(queries, signatures(queries), targets):
        primary = int((q[:3] >= 0) @ MASK)
        secondary = primary ^ (1 << np.argmin(abs(q[:3])))
        candidates = np.concatenate((banks[primary], banks[secondary]))
        routed.append(target in candidates)
        ranked = candidates
        if len(candidates) > 32:
            distance = np.bitwise_count(stored[candidates] ^ signature)
            ranked = candidates[np.argsort(distance, kind="stable")[:32]]
            hamming_count += len(candidates)
        dot_count += min(32,len(candidates))
        for i, subset in enumerate((candidates[:32], ranked)):
            survived[i].append(target in subset)
            picked = subset[np.argmax(keys[subset] @ q)] if len(subset) else -1
            correct[i].append(picked == target)
    def packets(cues):
        return np.reshape(cues, (PACKETS,4)).all(axis=1).sum()
    for i, arm in enumerate(("insertion","hamming16")):
        yield (seed,load,arm,sum(routed),packets(routed),sum(survived[i]),sum(correct[i]),
               packets(correct[i]),dot_count,hamming_count*i,stored.nbytes*i)


if __name__ == "__main__":
    out = Path(os.environ["TRAINLAB_OUTPUT_DIR"])
    with (out/"panel.csv").open("w") as f:
        writer = csv.writer(f,lineterminator="\n")
        writer.writerow("seed load arm routed routed_packets survived correct packets dots hamming extra_bytes".split())
        for seed in (11,23,37):
            for load in (29,64,256):
                writer.writerows(measure(seed,load))
    (out/"metrics.json").write_text('{"panels":9}')
