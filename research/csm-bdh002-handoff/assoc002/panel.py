import csv
import json
import os
from pathlib import Path

import numpy as np

D = 64
BITS = 8
PACKETS = 512
MASK = 1 << np.arange(3)


def normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def make(seed, load):
    rng = np.random.default_rng(seed)
    keys = normalize(rng.normal(size=(load, D)))
    ids = rng.choice(256, load, replace=False)
    values = 2 * ((ids[:, None] >> np.arange(BITS)) & 1) - 1
    targets = rng.integers(load, size=4 * PACKETS)
    queries = normalize(keys[targets] + rng.normal(0, .09, size=(len(targets), D)))
    matrix = np.zeros((BITS, D))
    for key, value in zip(keys, values):
        matrix += np.outer(value - matrix @ key, key)
    return rng, keys, values, targets, queries, matrix


def run():
    output = Path(os.environ["TRAINLAB_OUTPUT_DIR"])
    with (output / "panel.csv").open("w") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("seed","load","arm","correct_cues","correct_packets","lookups","key_comparisons","numeric_state_bytes"))
        for seed in (11, 23, 37):
            for load in (29, 64, 256):
                _, _, _, _, cq, cm = make([seed,load,0], load)
                threshold = np.median(abs(cq @ cm.T).min(axis=1))
                rng, keys, values, targets, queries, matrix = make([seed,load,1], load)
                # All arms share one memory and identical with-replacement packets per cell.
                assert np.array_equal((keys @ keys.T).argmax(axis=1), np.arange(load))
                expected = values[targets] > 0
                delta = queries @ matrix.T
                weak = abs(delta).min(axis=1) < threshold
                addresses = ((keys[:,:3] >= 0) @ MASK)
                banks = [np.flatnonzero(addresses == i) for i in range(8)]
                indices, random_indices, counts = [], [], []
                for q in queries:
                    bank = int((q[:3] >= 0) @ MASK)
                    adjacent = bank ^ (1 << np.argmin(abs(q[:3])))
                    subset = np.concatenate((banks[bank], banks[adjacent]))[:32]
                    counts.append(len(subset))
                    indices.append(subset[np.argmax(keys[subset] @ q)] if len(subset) else -1)
                    control = rng.choice(load, len(subset), replace=False)
                    random_indices.append(control[np.argmax(keys[control] @ q)] if len(control) else -1)
                counts = np.array(counts)
                def read(indices):
                    predictions = values[indices].copy()
                    predictions[np.asarray(indices) < 0] = 0
                    return predictions
                banked = read(indices)
                n = len(queries)
                exact = values[(queries @ keys.T).argmax(axis=1)]
                store = keys.nbytes + values.nbytes
                bank_bytes = store + sum(b.nbytes for b in banks)
                for arm, raw, calls, compares, storage in (
                    ("hebbian", queries @ (values.T @ keys).T, 0, 0, matrix.nbytes),
                    ("delta",delta,0,0,matrix.nbytes),
                    ("full",exact,n,n*load,store),
                    ("shifted_control",values[((queries @ keys.T).argmax(axis=1)+1)%load],n,n*load,store),
                    ("random",read(random_indices),n,counts.sum(),store),
                    ("banked",banked,n,counts.sum(),bank_bytes),
                    ("hybrid",np.where(weak[:,None],banked,delta),weak.sum(),counts[weak].sum(),bank_bytes+matrix.nbytes),
                ):
                    correct = ((raw > 0) == expected).all(axis=1) & (raw != 0).any(axis=1)
                    writer.writerow((seed, load, arm, correct.sum(), correct.reshape(-1,4).all(axis=1).sum(),
                                     calls, compares, storage))
    (output / "metrics.json").write_text(json.dumps({"panels":9}))


if __name__ == "__main__":
    run()
