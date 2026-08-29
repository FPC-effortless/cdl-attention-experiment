# CASM v0.1 compression-trained Q/K — multi-seed replication

## Decision

Promote **compression-trained Q/K** to the current CASM routing baseline.

Compression-derived predictive utility is used during training to shape ordinary Q/K routing scores. The runtime pairwise compression-score MLP is removed.

## Runs

Four controlled 800-step seeds are included: the original efficiency experiment (20260901) plus independent replications 20260921, 20260922, and 20260923. Each seed trains three variants from compatible initialization:

- `qk-memory`: ordinary Q/K memory, no compression supervision;
- `compression-qk`: ordinary Q/K inference with compression-derived training supervision;
- `compression`: Q/K + pairwise compression-score MLP at inference, with compression supervision.

All variants use the same ~1.347M parameter configuration.

## Heterogeneous hard-task answer NLL

| Seed | Q/K memory | Compression-QK | Full compression |
|---|---:|---:|---:|
| 20260901 | 1.68017 | **1.62488** | 1.63477 |
| 20260921 | 1.73554 | **1.66997** | 1.71296 |
| 20260922 | 1.68027 | **1.67230** | 1.67577 |
| 20260923 | 1.72459 | **1.68099** | 1.69780 |
| **Mean** | **1.70514** | **1.66203** | **1.68032** |

Compression-QK beats ordinary Q/K on answer NLL in **4/4 seeds**.

Mean difference:

`compression-qk - qk-memory = -0.04311 nats`

Compression-QK also beats the full runtime compression scorer in **4/4 seeds**.

Mean difference:

`compression-qk - compression = -0.01829 nats`

## Packed-stream hard answer NLL

Mean across four seeds:

- Q/K memory: 2.08684
- **Compression-QK: 2.05783**
- Full compression: 2.07512

## Exact correctness

Exact teacher-forced answer accuracy is effectively tied in aggregate:

- Q/K memory: 0.17448
- Compression-QK: 0.17344
- Full compression: 0.17448

Therefore the replicated result is currently a **probabilistic/calibration improvement**, not a demonstrated increase in discrete solve rate.

## Per-task mean NLL difference: compression-QK minus Q/K

| Task | Mean delta NLL | Direction |
|---|---:|---|
| graph reachability | **-0.26459** | strong improvement |
| state tracking | **-0.01383** | modest improvement |
| reverse/copy | +0.00094 | neutral |
| arithmetic | +0.00149 | neutral |
| associative recall | +0.00367 | neutral/slight regression |
| rule induction | +0.01365 | regression |

The present gain is therefore concentrated in structural / stateful routing, especially graph reachability. It is not yet a universal reasoning improvement.

## Efficiency result

From the original paired runtime benchmark (CPU, batch 8 x 192 bytes):

- Q/K memory: ~38.4k tokens/s
- **Compression-QK: ~39.1k tokens/s**
- Full compression: ~34.6k tokens/s

Absolute CPU timings are implementation-dependent. The architectural conclusion is more robust: the pairwise compression-score MLP is unnecessary at inference; compression supervision can be amortized into ordinary Q/K geometry.

## Next experimental gate

Do **not** immediately scale parameter count.

Next:

1. longer-horizon structural extrapolation (deeper graphs, longer state histories, more cross-chunk interference);
2. determine whether the gain persists outside the current graph-heavy regime;
3. only then revisit selective write/erase memory management.

The first selective-write v0.2 candidate was stable and test-green but underperformed compression-QK in a 100-step paired smoke test, so it remains an unpromoted branch rather than the new baseline.
