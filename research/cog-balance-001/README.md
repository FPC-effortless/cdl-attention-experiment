# COG-BALANCE-001

## Matched-write diagnostic

Run `OPENBLAS_NUM_THREADS=1 python research/cog-balance-001/gate_probe.py`. Two local runs produced byte-identical JSON. Each condition uses 4,096 worlds for each of seeds 9601, 9611 and 9629. Fit the linear scorer using separate labeled synthetic worlds seeded at 1729. Each block contains one genuine write and seven distractors, with an externally supplied key. Sparse controls commit once after buffering all eight inputs. Dense controls consume all eight inputs and include a posterior-vote reference that knows the generator.

Both-correct requires getting the same query right in two worlds that differ only in the value of the genuine write. The modulo operators are given and invertible, so this metric measures memory revision.

| Cue strength (standard deviations) | Learned gate | Periodic | Random | Signal-aware dense | Oracle event |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 12.05% | 12.57% | 12.68% | 17.17% | 100% |
| 1 | 37.53% | 12.57% | 12.68% | 36.12% | 100% |
| 2 | 70.66% | 12.57% | 12.68% | 70.23% | 100% |
| 4 | 98.71% | 12.57% | 12.68% | 98.71% | 100% |

Commit counts measure writes to persistent state. Each sparse arm sees the full block and the learned arm scores all events. Buffer cost, compute, latency and energy are separate outcomes.

## Full-model follow-up

Cross the Native HH and slow-state SS backbones with the learned event gate versus its matched static schedule. Include untouched HH and SS, GQA Transformer and GRU trained on the same examples, and use the existing TRACE-007 gated-delta reference and unified model as additional comparisons where compatible. Freeze budget and all hyperparameters before opening held results. Compare final-only and intermediate supervision with the same total cognitive weight. Keep optimizer fixed for mechanism attribution. Run three paired seeds at approximately 1.08M first and repeat the same comparisons at 8.8M for scaling. Report matched-data results and separately report a matched-compute budget. Add Mamba-2 when the reference implementation and cost accounting can be validated.

Require a gain over the static schedule at matched writes on natural-language and synthetic paired revisions, and report a confidence interval clustered by source document or world. Separate oracle write and oracle address, track query-target accuracy and measure false writes, missed writes, held compositions, BPB, persistent bytes including router buffers, operation counts and measured latency. Give all baselines the same input, excluding privileged targets at inference. Include randomized gates and future-token perturbation checks. For CPU runs, count scoring and buffered reads in the measured runtime.

This branch contains the local diagnostic. Full-model and scale comparisons are prospective.
