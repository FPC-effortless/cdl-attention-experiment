# VENTILATION-ALERT-008: learned requests under randomized timing

This is a private-Kaggle GPU benchmark of simulated building telemetry; no actual indoor-air deployment or general architecture qualification. The endpoint is **severe under-alerts accumulated across all observed post-offset ticks**, with false severe alerts and observation cost alongside it.

## Scientific result

Frozen multi-cell gate counts: **COG 4/4; SDPA 4/4; last-only MLP 4/4**. Across the four equally weighted cells, lowest learned post-offset severe-miss rate: **6.31% (transformer)**, versus reserve rule **12.75%** and always fresh **5.78%**. These unweighted summaries are descriptive; shared families are not independent across cells.

## Design and validity

24,000 synthetic FIT families; 8,000 disjoint HELD families replayed in four check availability/noise regimes. Counterfactual family worlds share noise, witness offset and check opportunities while final correct alert differs. Witness offset starts at varying ticks 2–12; query terminates at varying ticks 10–15, hidden from policy. For onset after termination, the episode contributes terminal counts but no post-offset ticks. Realized end is not passed as an input. The observation is a noisy synthetic independent estimate of *witness offset*, not latent concentration or a true class. The simulated policy cannot inspect it before taking a check. All policies use the same cutoff rule after requests. There is no physical control.

Three seeds and 1,024 updates each for tied COG directed communication, tied causal SDPA and pointwise MLP, 384 worlds per update; 192,000 stacked training histories per fit (24,000 families × two worlds × four cells), with randomized FIT acquisition traces and causal observable inputs. FIT targets are weighted one-step loss reduction from obtaining the current check. Held labels are not passed to the request policy. All learners use 48 hidden channels and the same request cap; equal width is not equal parameter count or equal FLOPs. See replay for per-arm counts. Frozen reserve comparator delays the third request until tick ten, while periodic uses early spacing. Realized observation cost may differ even with the same cap, so compare the actual mean requests.

## Per-cell performance

Each cell below is **post-offset severe misses % / post-offset false severe % / terminal complete pairs % / requests per history**. Learner rates average the same three seeds; rules evaluated once. Lower except terminal pairs. The replay contains integer denominators. No evaluation model or future information is supplied to the policies.

| Policy | dense-accurate | dense-degraded | gap-accurate | gap-degraded |
|---|---:|---:|---:|---:|
| Never | 31.20 / 17.37 / 33.05 / 0.00 | 31.20 / 17.37 / 33.05 / 0.00 | 31.20 / 17.37 / 33.05 / 0.00 | 31.20 / 17.37 / 33.05 / 0.00 |
| First check only | 31.41 / 17.28 / 33.56 / 1.00 | 29.92 / 16.37 / 36.65 / 1.00 | 31.41 / 17.28 / 33.56 / 1.00 | 29.92 / 16.37 / 36.65 / 1.00 |
| Periodic | 12.75 / 7.64 / 62.51 / 3.00 | 15.41 / 9.44 / 55.67 / 3.00 | 12.05 / 7.43 / 73.19 / 2.96 | 15.11 / 9.37 / 62.45 / 2.96 |
| Reserve | 9.38 / 6.13 / 85.56 / 3.00 | 13.28 / 8.72 / 71.41 / 3.00 | 12.66 / 7.79 / 81.97 / 2.72 | 15.70 / 9.86 / 69.28 / 2.72 |
| Always fresh | 1.01 / 1.49 / 94.09 / 13.53 | 6.37 / 5.14 / 77.51 / 13.53 | 5.86 / 4.02 / 89.51 / 9.34 | 9.86 / 7.05 / 74.00 / 9.34 |
| Tied COG | 2.06 / 2.46 / 87.76 / 1.79 | 5.53 / 5.49 / 78.28 / 1.93 | 7.67 / 5.18 / 81.70 / 1.65 | 10.26 / 7.37 / 73.37 / 1.75 |
| Tied SDPA | 2.03 / 3.14 / 89.61 / 1.87 | 5.53 / 6.58 / 78.95 / 2.01 | 7.61 / 5.66 / 83.52 / 1.72 | 10.07 / 8.20 / 73.97 / 1.82 |
| Last-only MLP | 4.49 / 4.37 / 88.12 / 2.18 | 9.22 / 7.50 / 75.15 / 2.31 | 9.03 / 6.36 / 82.79 / 1.96 | 12.62 / 8.66 / 70.88 / 2.07 |

## Paired differences against reserve

Descriptive percentile intervals from 1,000 paired-family resamples of the fixed evaluated seeds. Values are severe-miss percentage points; lower is better. They do not generalize beyond this synthetic distribution. The prespecified gate also requires at least two points fewer severe misses in every cell, no more than two points worse false severe or terminal pairs, at most 0.10 extra checks/history, and repeat direction in all three seeds.
| Policy | dense-accurate | dense-degraded | gap-accurate | gap-degraded |
|---|---:|---:|---:|---:|
| Periodic | +3.37 [+2.73, +4.01] | +2.13 [+1.34, +2.92] | -0.61 [-1.20, -0.09] | -0.59 [-1.29, +0.14] |
| Tied COG | -7.32 [-7.81, -6.82] | -7.75 [-8.34, -7.16] | -4.99 [-5.60, -4.45] | -5.44 [-6.13, -4.82] |
| Tied SDPA | -7.35 [-7.84, -6.85] | -7.75 [-8.34, -7.13] | -5.06 [-5.64, -4.56] | -5.63 [-6.28, -4.99] |
| Last-only MLP | -4.89 [-5.39, -4.39] | -4.05 [-4.67, -3.47] | -3.63 [-4.21, -3.07] | -3.08 [-3.76, -2.43] |

## Reproduction

Kaggle version 1; reported **Tesla T4**, 2 visible GPU device(s), Torch 2.11.0+cu130, 70.49 seconds for the script's benchmark region including fixture generation, training and scoring, not serving latency. Notebook is private with internet disabled. Script fails closed if CUDA is absent. The separate local implementation reproduced all 560 auditable endpoint counts (14 policy/seed identities × four cells × ten quantities) from exported trajectories and regenerated held fixtures, checked observation and reserve caps and validated exact source and artifact hashes. Native worker was the explicitly requested external Kaggle runtime, not a TrainLab qualification receipt. Model weights were not exported. No new sealed or real-world data, model promotion or serving claim.

- Frozen source SHA-256: 1c281008bf08fecfa8b2eaf81abf48dbee3883ecff7ee6e430920a369bec9fd6
- Native report SHA-256: f682a43881d3aa159739cd780346ee61674508fc3f6f18bf8517387f8780bdf9
- ACT raw trajectories SHA-256: 6ca6e43958ecf8b3a5159789e33dc32cf230943a4d215c546023ebc6a0862590
- Detached recount SHA-256: 8abf7e79aea4598fc635646fe88ea61b9b6ead110b7103016d3ec409ae844ade

