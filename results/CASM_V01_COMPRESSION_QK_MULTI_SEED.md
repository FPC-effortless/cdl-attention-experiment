# CASM v0.1 compression-trained Q/K — multi-seed replication

## Status after benchmark audit

**The earlier promotion decision is withdrawn.**

A post-run audit found that the original `graph_reachability` generator always emitted `answer yes`. The apparent multi-seed compression-QK advantage was therefore dominated by a defective task that did not require distinguishing reachable from unreachable graphs.

Compression-QK remains an efficiency-motivated hypothesis, not a validated generally superior router.

## Runs

Four controlled 800-step seeds are included: 20260901, 20260921, 20260922, and 20260923. Each seed trains:

- `qk-memory`: ordinary Q/K memory, no compression supervision;
- `compression-qk`: ordinary Q/K inference with compression-derived training supervision;
- `compression`: Q/K + pairwise compression-score MLP at inference, with compression supervision.

All use the same ~1.347M-parameter configuration.

## Original aggregate result

| Seed | Q/K memory | Compression-QK | Full compression |
|---|---:|---:|---:|
| 20260901 | 1.68017 | 1.62488 | 1.63477 |
| 20260921 | 1.73554 | 1.66997 | 1.71296 |
| 20260922 | 1.68027 | 1.67230 | 1.67577 |
| 20260923 | 1.72459 | 1.68099 | 1.69780 |
| Mean | 1.70514 | 1.66203 | 1.68032 |

This gave `compression-qk - qk-memory = -0.04311 nats` over the six-task average.

## Why that aggregate is not valid evidence of general improvement

Mean per-task NLL difference, compression-QK minus Q/K:

| Task | Mean delta NLL |
|---|---:|
| graph reachability | **-0.26459** |
| state tracking | -0.01383 |
| reverse/copy | +0.00094 |
| arithmetic | +0.00149 |
| associative recall | +0.00367 |
| rule induction | +0.01365 |

Because the six tasks are equally weighted, the defective graph task contributes about `-0.26459 / 6 = -0.04410 nats` to the overall mean—essentially the entire reported `-0.04311` advantage.

Excluding graph reachability, the mean compression-QK minus Q/K difference across the other five tasks is:

`+0.00118 nats`

That is effectively a tie/slight regression, not evidence of a general compression-routing advantage.

## Additional audit finding

Free autoregressive generation on balanced reachable/unreachable graphs using the old checkpoints shows that neither model generates `no` correctly on unreachable examples. The old task had never trained that distinction.

## Efficiency result still stands as an implementation fact

From the original paired CPU benchmark:

- Q/K memory: ~38.4k tokens/s
- compression-QK: ~39.1k tokens/s
- full compression: ~34.6k tokens/s

So if compression-derived supervision eventually proves useful on a correct benchmark, amortizing it into ordinary Q/K geometry remains preferable to retaining the pairwise compression MLP at inference.

## Corrective action

A new branch, `experiment/casm-v02-corrected-curriculum`, now:

1. balances graph reachability between reachable and unreachable cases;
2. validates labels against actual directed reachability;
3. lengthens training sequences so hard state-tracking cases can enter training;
4. compares only ordinary Q/K vs compression-trained Q/K;
5. adds long-horizon state and associative-memory stress tests.

No architecture promotion should occur until that corrected experiment completes.
