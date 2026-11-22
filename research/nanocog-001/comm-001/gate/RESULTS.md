# COG-COMM-001 pre-fit gates: literal packet survival and local learnability

These are separate CPU diagnostics, not main COG-COMM model results. The main CAL/DEV roles and all protected evaluation were left unopened by these workers.

## Initialized literal evidence survives too rarely

The initialized C model ran on all 640 audited synthetic TRAIN worlds with three seeds and two layers (3,840 layer-world observations). Every source considered below was eligible under the expanded age envelope. There were no updates. The reference counts are based on original literal positions, not abstract information encoded in later payloads.

| Evidence at its original positions | Ever published | Resident before answer query | Selected by learned subscriber | Selected by cyclic subscriber on the same scratch |
|---|---:|---:|---:|---:|
| Exact answer-color token | 2,647 / 3,840 | 2 / 3,840 | 0 / 3,840 | 1 / 3,840 |
| Any position from the supporting sentence | 3,840 / 3,840 | 30 / 3,840 | 0 / 3,840 | 15 / 3,840 |

All survivors occurred in one seed's second layer, nearly all in the zero-distractor stratum, and none at four or eight distractors. Twelve seed-layer-family cases retained at least one position from the support sentence in both worlds, but the learned subscriber selected no complete family. There were no two-world retained pairs for the exact value token. The cyclic rule was evaluated on the same cache, not by running a separately trained control.

The dominant literal support loss happened between publication and retention, before subscription. Extending nominal age reach alone did not protect entries from an eight-slot per-token writer. This rules out treating the baseline as a demonstrated direct-position route; it does not establish that trained weights cannot learn retention or that information never reaches the reader through contextual summaries or later payloads. A 59.17-second single-thread native subprocess included 240 ordinary model calls and 480 instrumentation calls; this is not a latency benchmark. A weight-free auditor verified saved positions, eligibility and aggregate arithmetic. It did not independently execute the writer.

### Decision for main experiments

Do not spend the full main fit budget on an unqualified direct-retrieval premise. The next retention comparison should keep the **same storage and read budget**, and change only what the write path admits: unmodified per-token publishing versus a raw-input-boundary policy that can publish at an observed separator. Charge token/event parsing and indexing. Track whether messages summarize a whole earlier statement rather than count a separator as if it were the literal answer token. Include a recency-matched blind read on each cache and do not supply support spans or oracle entity IDs to the model. If the write cap still systematically evicts support, a distinct capacity or replacement-policy comparison must address that, not silently infer that more age is sufficient.
