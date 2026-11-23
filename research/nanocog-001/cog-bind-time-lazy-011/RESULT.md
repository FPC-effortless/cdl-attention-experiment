# COG-BIND-TIME-LAZY-011: skip joint search when greedy evidence is certified

**Outcome: exactly equivalent packets and answers on the archived tape.** The conditional decoder matched always-joint on all 5,760 passage-by-seed packets (including abstentions) and all 23,040 query-by-seed outputs. It skipped 4,287/5,760 searches (74.43%); 1,473 had structurally infeasible greedy evidence and 0 failed only the precision certificate. No annotations, checkpoint weights or model execution are needed for the replay.

## Decoder-stage cost, not full inference

| Pane / seed | Skipped joint searches | Always-joint median pass (ms) | Standalone gate median (ms) | Reusing existing greedy packet median (ms) | Ratios: always-joint / standalone / handoff |
|---|---:|---:|---:|---:|---|
| dev:20371 | 291/384 | 90.64 | 68.97 | 28.57 | 1.31× standalone; 3.17× handoff |
| dev:20471 | 309/384 | 87.34 | 62.39 | 23.68 | 1.40× standalone; 3.69× handoff |
| dev:20571 | 277/384 | 90.26 | 72.69 | 32.79 | 1.24× standalone; 2.75× handoff |
| held-familiar:20371 | 603/768 | 186.08 | 145.61 | 55.57 | 1.28× standalone; 3.35× handoff |
| held-familiar:20471 | 547/768 | 182.45 | 142.43 | 63.98 | 1.28× standalone; 2.85× handoff |
| held-familiar:20571 | 570/768 | 199.12 | 159.93 | 67.88 | 1.25× standalone; 2.93× handoff |
| held-novel:20371 | 593/768 | 185.11 | 142.27 | 66.33 | 1.30× standalone; 2.79× handoff |
| held-novel:20471 | 521/768 | 208.77 | 155.00 | 71.97 | 1.35× standalone; 2.90× handoff |
| held-novel:20571 | 576/768 | 202.09 | 142.51 | 67.06 | 1.42× standalone; 3.01× handoff |

**Predeclared decoder-stage cost gate: PASS.** This required at least 10% improvement in the aggregate of medians and at most 5% regression in any held pane/seed.

Across the nine cases, the sum of median decoder-pass times was 1431.86 ms always-joint, 1091.80 ms standalone and 477.82 ms using the verified handoff (1.31× and 3.00× respectively). This is an aggregate of per-case medians, not a separate timed run. The timed section receives already tokenized inputs and detected cue locations. Standalone charges reconstruction of the greedy grouping; handoff charges only certification and fallback because it assumes an actual existing, exact grouping. Neither includes the model, tokenization, I/O, query composition, prompt re-reads or attention's KV accesses. The source implementation is Python, pinned single-thread CPU. End-to-end speed, real-language accuracy, changing live context, durable agent memory and prompt/KV traffic remain untested.

## Why the gate preserves the reference

A valid greedy packet uses each role’s unique greatest raw value. Any different feasible packet has a lower exact total. The guard uses the gap to each selected value’s next lower binary32 number and a conservative binary64 sum-error margin. If rounding could erase the distinction, the method returns to the unchanged exhaustive-equivalent joint decoder rather than assuming a winner. Synthetic tests include an extreme case where two mathematically different totals really do tie after summation. Agreement on the archived tape is exact equality to the reference, not a new scientific qualification or safety test.

## Next measurement

Keep the certified gate in a live incremental inference harness. Measure complete prompt ingestion, repeated turns, state restore/revision, tokens and state bytes touched, and quality at the same workload and hardware budget. Do not claim that this component result eliminates any prompt reprocessing or KV reads.
