# COG-ARCHIVE-ASCII-001 | Local exact lexical fast path

**The identical broad retrieval ran about twice as fast on this synthetic
ASCII-heavy workload.** Full rendered contexts matched byte for byte in all
40 timed pairs. The two-line isolated candidate has not changed production.

The candidate uses ordinary lowercase and the original token regex when
the input is entirely ASCII. Casefolding and NFD with combining-mark
removal have identical output for ASCII. The Unicode path is unchanged.
This does not change FTS matching, ranking, scope filters, tie order,
provenance, character budgets, write acceptance, or SQLite durability.

| Request | Matched pairs | Original median / p95, ms | Candidate median / p95, ms | Median paired speedup | Candidate faster |
|---|---:|---:|---:|---:|---:|
| Broad, 8,000-character cap | 24 | 129.94 / 148.00 | 60.17 / 65.78 | 2.12× | 24/24 |
| Broad, 256-character cap | 8 | 124.36 / 141.67 | 55.36 / 59.45 | 2.18× | 8/8 |
| Rare needle, 8,000-character cap | 8 | 0.253 / 0.457 | 0.198 / 0.351 | 0.94× | 3/8 |

p95 is empirical nearest-rank; particularly for eight observations it is the
maximum, not a robust tail estimate. Paired ratios are computed on each
matched case and then summarized; coincident medians need not define the
paired effect. The rare-query timing is inconclusive and not a win.

## What was actually matched

The offline reconstruction matches the published cumulative operation
stream hashes at 1k, 10k and 50k in [PR #94 at its exact commit](https://github.com/FPC-effortless/cdl-attention-experiment/tree/ffb842adc176496d8838b774d0fe412aec5d1d20/research/nanocog-001/memory-validation-002).
The 50k-stream digest is
`473418a93cdcb46b832e7cd9f91df838b2866c6c71c5ba0f3baaa892c25bcf35`.
The initial lexical control plus sequence of 53,227 mutations yielded 256
current facts and 1,613 revisions. Both variants queried one shared, warm
temporary database on Linux tmpfs with WAL/FULL, Python 3.12.14 and
SQLite 3.53.1. This reuses the archived fixture, not all scale validation
workloads. Alternating the order of paired calls reduces simple warm-order
effects but does not establish performance on other hosts.

Every broad 8k result was 7,645 characters; the broad 256-cap result was
the 86-character empty envelope. A separate untimed callback census
counted 6,250 lexical scoring calls on the broad query in one scope,
versus one for a rare needle. The patch reduces per-match normalization
work. It does not bound the number of candidates, eliminate archive reads,
or make the result independent of stored text length.

Preflight passed 2,059 deterministic tokenizer/score cases and 144 context
comparisons spanning ties, mixed Unicode, source exclusion, foreign scope,
deletion, caps and restored backup. Timed full JSON equality passed 40/40.
No parity gate failed. These checks cover this patch and frozen panels;
they are not a requalification of every storage behavior.

## Costs and what is still unmeasured

The isolated whole run took 15.765 s wall / 15.710 s CPU. Fixture ingestion
took 6.336 s wall / 6.312 s CPU (not included in the query medians).
Peak process RSS was 37,224 KiB. Immediately after ingestion, main DB/WAL/
SHM bytes were 44,023,808 / 7,712,672 / 32,768, not backup footprint or
combined peak memory. Every paired call's wall time, CPU time, order,
context dimensions and hash are preserved in `RAW_TIMINGS.json`.

This ran on tmpfs, not characterized durable disk. It did not change or
measure the separate 49–83 ms one-token durable decoder path. That path
checks full model identity twice, reads and validates receipts/checkpoint,
runs the model, encodes and commits state/logits with CAS/WAL/FULL, then
decodes the authoritative receipt. Neither its crypto overhead nor its disk
sync time can be inferred by subtracting unrelated medians. Absent caller
wiring, those decoder logits did not respond to changed ledger facts in
the source study. No model executed here, and no LLM, KV-cache, lifelong
memory, asymptotic, semantic retrieval, or production latency claim follows.

Replay from the extracted private experiment directory with
`python -B bench_ascii.py NEW_RAW_TIMINGS.json` on a compatible isolated
Python/SQLite system with a tmpfs directory at /tmp. The harness validates
exact source, candidate and fixture hashes, checks parity, writes a new raw
timing result and discards the temporary synthetic database. Timing itself
is not expected to reproduce byte for byte.
