# CSM-SKILL-003 — Defer SWAP until a copy would lose a needed value

Compare primitive-only, eager SWAP and deferred SWAP controllers on the complete four-record roster from study 002. Deferred SWAP first checks for a final-value copy that preserves every sole remaining source still required. If no such copy exists, it attempts to resolve a reciprocal pair with SWAP. All arms share the scratch fallback. Require exact final state on every case and reproduce both prior arms before timing. Compare dispatch counts and expanded primitive counts. Measure the median wall time per goal across 30 blocks, balancing the six arm orders. Include planning and execution inside the planner and exclude training, goal construction, and result validation.

All three arms passed all 256 goals. Deferral removed 144 of the eager controller's 147 extra primitive operations while retaining 69 fewer dispatches than primitive-only planning.

| Controller | Dispatches | Expanded primitives | Median µs per goal |
| --- | ---: | ---: | ---: |
| Primitive only | 841 | 841 | 2.69 |
| Eager SWAP | 700 | 988 | 3.04 |
| Deferred SWAP | 772 | 844 | 2.59 |

Timing is a local Python microbenchmark over one small roster. The roughly 4% median difference between deferred and primitive-only execution needs independent replication. The controller priority was supplied by hand. A subsequent experiment could learn that priority from primitive cost and test transfer to larger record sets.

Replay from this directory with Python 3.12. The correctness run checks the previous study's per-goal counts and the exact final state. The timing file contains all 90 measured batches.

```sh
python experiment.py --time > /tmp/csm-skill003.csv 2> /tmp/csm-skill003-timing.csv
diff -u scores.csv /tmp/csm-skill003.csv
```
