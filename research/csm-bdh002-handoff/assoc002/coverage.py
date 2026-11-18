"""Post-run decomposition of candidate loss at the highest load."""
import csv
import sys
import numpy as np
from panel import MASK, make

writer = csv.writer(sys.stdout, lineterminator="\n")
writer.writerow(("seed","cues","in_two_banks","in_first_32"))
for seed in (11, 23, 37):
    _, keys, _, targets, queries, _ = make([seed, 256, 1], 256)
    ids = (keys[:, :3] >= 0) @ MASK
    before = after = 0
    for query, target in zip(queries, targets):
        bank = int((query[:3] >= 0) @ MASK)
        adjacent = bank ^ (1 << np.argmin(abs(query[:3])))
        candidates = np.concatenate((np.flatnonzero(ids == bank), np.flatnonzero(ids == adjacent)))
        before += int(target in candidates)
        after += int(target in candidates[:32])
    writer.writerow((seed, len(targets), before, after))
