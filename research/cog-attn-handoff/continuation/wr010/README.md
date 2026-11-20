# WR-010 learned grounding under mixed reads and writes

The runtime uses three actual frozen WR-005 analytic checkpoints (training seeds 13, 37 and 61) on fresh workload seeds 181, 193 and 211 respectively. Each raw command predicts operation, source and destination. Its action mutates persistent CompanyWorld shipment or policy values through the copy/add modulo-four executor. Reads use the public reference solver or its cached answer. The audit reconstructs the intended state from gold commands and separately reconstructs the state produced by executed commands. This distinguishes semantic drift from serving an outdated answer.

At joint confidence of at least 0.80, the compiled path can reuse an exact-string binding. Below that threshold the assisted arms query a simulated truthful clarification on each invocation. The unassisted ablation uses the model's prediction. This experiment uses a fixed escalation threshold and known synthetic vocabulary, with some novel verbs. The active path models a perfect reply without human waiting time. Model fitting and checkpoint selection are upstream.

Each 1,536-operation trace uses 32 objects, 128 distractor records per object and recurrent plus long-tail commands. Fresh inference, semantic-only caching, eager authority checks and the lazy write hook all use the same checkpoint and clarification rule. Lazy invalidation evicts only the written object's answer in constant time; the next read resolves it. No-op writes also evict. The delayed variant receives notifications eight operations late and loses 25% in expectation. The versioned variant additionally reads a synchronous per-object version incremented by the writer.

| Write probability | Median lazy speedup over fresh | Over semantic-only | Over eager |
| ---: | ---: | ---: | ---: |
| 1% | 11.37× | 11.86× | 1.30× |
| 10% | 2.92× | 2.66× | 1.15× |
| 50% | 1.65× | 1.28× | 1.27× |
| 90% | 1.51× | 1.06× | 1.26× |

These are medians of paired per-seed time ratios after three repetitions per cell. At 10% writes the throughput gain over semantic-only exceeded the frozen threshold of 1,000 ops/s on all seeds. Timing includes tokenization, checkpoint inference on misses, command execution, writes, routing, cache initialization and answers. Public world creation, file loading, verifier scoring, response serialization and external clarification or version service latency are excluded. Arm order was shuffled within each TrainLab job. Different jobs may experience different host load.

| Route | Incorrect writes / 6,962 | Stale reads / 11,470 | Full-reward reads / 11,470 |
| --- | ---: | ---: | ---: |
| Fresh, semantic-only, eager or synchronous lazy (each) | 4 | 0 | 11,344 |
| Lazy with delayed/lost notifications | 4 | 826 | 10,521 |
| Delayed/lost with atomic version check | 4 | 0 | 11,344 |
| Unassisted learned grounding with fresh answers | 1,021 | 0 | 9,334 |

The assisted runs each requested 4,034 clarifications. Four incorrect writes sufficed for 126 incorrect reads on the fresh and synchronous paths. A stale answer is judged against the executed state; verifier reward is judged against the intended state. The two error counts therefore need not add. The version lookup assumes the version and authoritative state are synchronously visible in the same process.

Research Lab OS froze six jobs, TrainLab executed seven modes across the four write rates, and the audit checked all six receipts and rescored 80,290 final-repetition read submissions with Veritas CompanyWorld. The audit shares the trace and language generator and independently implements state updates. The separately verified Forge-backed ecosystem reference comparison is included. This is a constructed learned-runtime integration with exact-string memoization and a scripted read solver. Distributed coherence, real clarification cost, trained TAC/LRC integration, and independent scientific replication remain open.

## Reproduction

Use Python 3.12, NumPy 2.3.5 and the four Veritas ecosystem packages from commit eec439492407b0aa26922c23b5ae40d95289f4eb. From a fresh directory with launch.py, audit.py and the recipe inputs, run:

    veritas-ecosystem run "$PWD/ecosystem"
    python launch.py
    python audit.py
    veritas-ecosystem verify "$PWD/ecosystem"

The archive contains snapshotted recipe inputs, checkpoints and vocabulary, raw timing samples and submissions, run receipts, SQLite evidence memory and audit results.
