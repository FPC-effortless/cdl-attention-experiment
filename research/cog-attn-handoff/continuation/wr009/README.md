# WR-009 mixed-read/write authority cache

Research Lab OS registered six TrainLab jobs on seeds 131, 149 and 167. Each used a public CompanyWorld scripted solver for shipment and approval questions and a stream over 16 objects. Both arms applied the same writes. Fresh resolution ran the solver on every read. The candidate captured each object's winning source identity and fields at startup, resolved that dependency again after each write to the object, evicted stale answers, and filled cold or evicted entries on the next read.

The workload includes direct and lower-priority revisions, projection changes, and precedence swaps. Each cell ran 2,048 operations five times from a cold answer cache. Timing includes dependency initialization, writes, invalidation, read misses, hits and answer collection. It excludes public fixture and trace generation, payload serialization, downstream scoring and external transport. Input records are in memory, delivery is synchronous and the target object of every write is supplied. This is a cache around a scripted reference solver. No trained model is involved.

| Write probability | Median speedup, 3 records | Median speedup, 131 records |
| ---: | ---: | ---: |
| 1% | 27.23× | 33.47× |
| 10% | 7.13× | 7.19× |
| 50% | 1.24× | 1.03× |
| 90% | 0.45× | 0.18× |

Each number is the median across seeds of the paired fresh/cached time ratios, after taking the median of five runs within each cell. At 50% writes with 131 records the seed ratios ranged from 0.93 to 1.22. The primary 10%-write, 131-record comparison met the prespecified threshold of 1,000 additional operations per second on every seed. Both arms traverse the records on a fresh solve. Only the candidate also traverses them to resolve authority on each write. This explains why a longer record list can hurt a write-heavy workload. Arms ran in separate local jobs, so wall-clock ratios also reflect machine and run-order variation.

The same-host audit checked six TrainLab receipts and reconstructed an oracle from the fixed operation streams. All 60,916 saved read submissions received full CompanyWorld reward. Only the final repetition's submissions were retained for each cell. The audit shares the seeded trace generator with the benchmark but tracks authority values and precedence independently. All four Veritas ecosystem packages came from upstream revision eec439492407b0aa26922c23b5ae40d95289f4eb. Forge's separate reference comparison passed ecosystem reverification.

## Replay

Install the four pinned ecosystem packages with Python 3.12. In a fresh directory containing launch.py, audit.py and recipe/workload.py, run:

    veritas-ecosystem run "$PWD/ecosystem"
    python launch.py
    python audit.py
    veritas-ecosystem verify "$PWD/ecosystem"

The evidence bundle contains the registered protocol, code snapshots, local lab database, run receipts, raw time samples, saved submissions and audit results. The timings are local observations. Independent replication and end-to-end learned runtime qualification remain open.
