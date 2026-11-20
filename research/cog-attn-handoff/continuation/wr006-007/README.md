# WR-006 and WR-007 initial pilots

See [protocol.md](protocol.md) for workloads, gates and assumptions. The scripts use Python 3.12.14 and NumPy 2.3.5. Run python wr006.py and python wr007.py in a copy to preserve timings. The WR-007 JSON replayed byte for byte.

## WR-006 bitemporal indexing

The path-copied segment tree indexes valid time with a per-entity root at each transaction. Lookup selects the last visible root and finds its valid-time predecessor. A correction at a repeated timestamp updates its new root while preserving the older answer. Both indexes matched full scan on all 12,288 benchmark queries.

| Writes | Worst one-dimensional candidate checks | Worst tree node visits | One-dimensional/tree query time | Tree payload divided by packed baseline estimate |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 103 | 13 | 0.38× | 8.67× |
| 2,048 | 1,543 | 20 | 2.65× | 12.67× |
| 32,768 | 24,729 | 27 | 34.12× | 16.67× |

Candidate checks and node visits count different operations. Node visits exclude the binary search for the visible root. The predecessor walk takes logarithmic work in the valid-time universe. Each write allocates logarithmically many nodes.

At 32,768 writes the tree took 166.55 ms to build versus 18.99 ms for the list. Its 512 queries took 3.09 ms versus 105.38 ms. Including construction, the tree took longer over the tested query volume. Packed payload estimates exclude Python overhead and spare capacity. The prototype assumes a bounded integer valid-time range, strictly increasing transaction IDs and nonnegative payloads. It covers append-only corrections.

## WR-007 criticality-aware promotion

Eight paired seeds give 16,384 high-criticality trials per regime. Policies saw the same simulated verification sequences. The adaptive policy estimates verifier error from labeled calibration and uses dynamic programming under the stated cost model.

| High-criticality rule, independent noise | Wrong promotions | Promotions | Verifications | Realized cost |
| --- | ---: | ---: | ---: | ---: |
| Original posterior threshold | 268 | 16,256 | 39,656 | 32,876.40 |
| Criticality threshold, pooled calibration | 35 | 15,918 | 60,322 | 13,014.30 |
| Adaptive, pooled calibration | 35 | 15,918 | 60,017 | 12,968.55 |
| Adaptive, context calibration | 1 | 8,175 | 27,133 | 12,378.95 |

The context-aware policy deferred all high-criticality cases in the 15% noise context without requesting verification. The lower error count comes with lower coverage. Under fully correlated errors, the pooled criticality threshold made 1,678 false promotions, equal to the original rule. The context-aware policy made 388 wrong promotions among 8,192 admissions. Repeated reads from the same corrupted source inflate confidence under the modeled independence assumption. The simulation assumes two known meanings and correct entity binding. It leaves compiled runtime integration and acquisition of independent evidence open.
