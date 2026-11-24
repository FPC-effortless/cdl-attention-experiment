# COG-1M-LATENT-005: bounded reuse depends on locality

**First-demand caching beat normal text recomputation on repeatedly read records. Waiting for a second read cut rebuilds during churn, but was not a better default.** The trial reused two frozen native decoders; no model was trained. Each of eight synthetic roots was crossed over eight workloads, with the same inputs across four policies. Roots, not repeated requests or checkpoints, are the independent units.

The numbers below are paired median milliseconds **saved versus one-call canonical text** per edit plus all subsequent reads. Negative means slower. Values are shown for checkpoints 20371 / 20471.

| Reads per edit | Eager, unbounded | Eager, 4-state cap | Selective, same 4-state cap |
| --- | ---: | ---: | ---: |
| 1 | -4.06 / -3.48 | -4.03 / -3.90 | -1.52 / -0.67 |
| 2 | -0.97 / +0.11 | -2.59 / -1.33 | -4.74 / -3.88 |
| 3 | +2.58 / +3.30 | +1.59 / +0.82 | -2.91 / -2.20 |
| 4 | +4.47 / +7.04 | +1.98 / -0.49 | -1.07 / -1.30 |
| 16 | +35.15 / +35.49 | +15.91 / +22.73 | +12.61 / +11.76 |

Bounded eager first showed a positive paired median at **three** reads on both checkpoints, but the small four-read reversal on one checkpoint means this is not a reliable threshold. At three reads its absolute p50/p95 was 48.3/69.9 and 46.2/58.7 ms, versus 51.2/76.7 and 48.3/68.3 ms for text. The selective policy first saved time on both at **16 among the tested workloads**, not necessarily at exactly 16.

**Access order changed the winner.** These sequences used the same query counts for six keys. Counts are over 288 queries in each locality condition per checkpoint.

| Locality | Eager-bounded builds / hits / evictions | Selective builds / hits / evictions | Selective savings versus eager-bounded | Selective savings versus live text |
| --- | ---: | ---: | ---: | ---: |
| Hot, mostly stable | 32 / 256 / 24 | 48 / 208 / 24 | -32.03 / -26.27 ms | -1.04 / +15.21 ms |
| Churn | 160 / 128 / 152 | 80 / 144 / 56 | +29.69 / +25.38 ms | -23.99 / -17.29 ms |
| Shifting | 64 / 224 / 56 | 72 / 160 / 48 | -30.23 / -26.08 ms | -9.81 / -23.34 ms |

Unbounded eager stayed faster than live across those three traces, but reached seven resident states versus four for either bounded path. Peak actual logical storage was 74,785 bytes unbounded, 42,742 bytes eager-bounded and 42,808 bytes selective (payload and metadata). Both bounded modes stayed under the shared 45,040-byte charged limit. Median closed SQLite file per traffic session was 45,056 bytes live, 94,208 with either bounded policy and 131,072 unbounded; temporary WAL and shared-memory files are separate. The selective policy also durably maintained a bounded list of recently seen revisions, so its saved rebuilds were not free.

**Correctness stayed the same.** All 2,368 native answers per policy and checkpoint matched the same checkpoint’s live text path in greedy tokens and stop behavior. The maximum logit drift was 5.43×10⁻⁶. Every policy answered 2,288/2,304 and 2,295/2,304 direct requests correctly and 48/64 alias requests correctly on both checkpoints. The source guard rejected 576 deliberate stale state offers per checkpoint, including 192 same-value reversions, without decoding or changing admission order. The 512 refused read observations per checkpoint across four policies never reached the decoder. Reuse persisted across new worker processes. This was not physical cold-disk I/O or a concurrent commit/emission test.

**Practical choice:** favor canonical text when reuse is uncertain, bounded eager for demonstrably hot records, and test adaptive admission on realistic trace mixes before enabling it. This trial supports a cache policy decision, not a broad architecture or serving-speed claim.
