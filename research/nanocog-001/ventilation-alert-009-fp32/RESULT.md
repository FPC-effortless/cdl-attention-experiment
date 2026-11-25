# VENTILATION-ALERT-009-R: does model capacity improve check timing?

## Result

The prespecified width-scaling screen is independent of whether a policy beats a reserve rule. Lower post-offset severe-miss rate is better. Cell means below equally weight four regimes using the same family IDs; they are not extra independent replications.

| Model / width | Parameters | Mean severe-miss rate | 008-style screen vs reserve | Scaling gate vs width 48 |
|---|---:|---:|---:|---:|
| cog / 48 | 24,289 | 6.88% | 4/4 | baseline |
| transformer / 48 | 23,665 | 6.70% | 4/4 | baseline |
| cog / 384 | 1,484,545 | 7.11% | 4/4 | FAIL |
| transformer / 384 | 1,479,553 | 7.75% | 4/4 | FAIL |
| cog / 720 | 5,202,721 | 8.81% | 4/4 | FAIL |
| transformer / 720 | 5,193,361 | 13.49% | 0/4 | FAIL |

## Frozen method

Fresh 24,000 FIT and 8,000 held matched synthetic families, each repeated in four offer-gap/noise cohorts, 16-tick cap and randomized onset and terminal tick. For the same family, the world pair and nuisance realization are held fixed between all models and rules. Every policy can observe only current readings, offer flag and previous acquired checks; terminal tick is not an input. A noisy synthetic reference reports witness offset, not true concentration. Policies request calibration observations; no hardware, actuation, or deployment decisions.

Two tied passes using seeded COG directed blocks or causal SDPA, with widths 48, 384 and 720, three paired random seeds per width. 1,024 actual AdamW updates of 384 worlds each, compared under the same exposure but **not matched training FLOPs**. FP32 on CUDA with TF32 disabled, identical sampled rows within seed. No skipped optimizer updates. Held scores use fixed one-step risk utility threshold and reserve guard (maximum three checks, no third before tick ten); no held tuning. Class alerts use the shared engineered cutoff, not model-generated classes. Replay preserves all decisions, not fitted checkpoints.

Width-scale gate: cut severe misses by at least 1 point in ≥3/4 cells, no cell >0.5 points worse, false severe no >1 point worse, terminal pair no >1 point worse, and ≤0.10 extra checks per history. Original study's benefit/risk/cost screen against reserve is listed separately. Distinguish observed differences from architecture claims.

## Per-cell metrics

Post-offset severe misses % / post-offset false severe % / terminal pair accuracy % / realized mean checks per history. Means over three fixed seeds for the learned policies; rules once.

| Model | dense-accurate | dense-degraded | gap-accurate | gap-degraded |
|---|---:|---:|---:|---:|
| never | 31.94 / 17.87 / 31.00 / 0.00 | 31.94 / 17.87 / 31.00 / 0.00 | 31.94 / 17.87 / 31.00 / 0.00 | 31.94 / 17.87 / 31.00 / 0.00 |
| first | 31.71 / 17.84 / 31.38 / 1.00 | 29.53 / 16.36 / 36.20 / 1.00 | 31.71 / 17.84 / 31.38 / 1.00 | 29.53 / 16.36 / 36.20 / 1.00 |
| periodic | 13.34 / 7.71 / 61.61 / 3.00 | 16.25 / 9.40 / 55.74 / 3.00 | 12.98 / 7.50 / 72.10 / 2.96 | 15.84 / 9.29 / 62.24 / 2.96 |
| reserve | 9.98 / 6.12 / 84.65 / 3.00 | 13.43 / 8.45 / 70.84 / 3.00 | 13.63 / 7.90 / 80.80 / 2.70 | 16.56 / 9.54 / 68.60 / 2.70 |
| always | 1.10 / 1.51 / 93.78 / 13.45 | 6.75 / 5.12 / 77.09 / 13.45 | 6.43 / 4.12 / 88.55 / 9.27 | 10.91 / 6.84 / 73.89 / 9.27 |
| cog 48 | 2.47 / 2.44 / 89.45 / 1.84 | 5.95 / 5.36 / 79.36 / 1.99 | 8.35 / 5.12 / 82.83 / 1.70 | 10.73 / 7.20 / 73.49 / 1.80 |
| transformer 48 | 2.32 / 2.58 / 90.79 / 1.87 | 5.82 / 5.57 / 80.64 / 2.00 | 8.21 / 5.23 / 84.47 / 1.71 | 10.45 / 7.36 / 75.42 / 1.81 |
| cog 384 | 2.69 / 2.45 / 92.94 / 1.90 | 6.22 / 5.27 / 82.85 / 2.00 | 8.53 / 5.08 / 86.25 / 1.73 | 10.99 / 7.11 / 76.94 / 1.79 |
| transformer 384 | 3.56 / 3.57 / 87.87 / 1.89 | 6.84 / 6.30 / 76.59 / 1.97 | 9.37 / 6.23 / 81.08 / 1.72 | 11.22 / 8.17 / 71.32 / 1.77 |
| cog 720 | 4.71 / 3.53 / 87.84 / 1.93 | 8.22 / 6.03 / 77.40 / 2.03 | 10.14 / 6.26 / 81.34 / 1.75 | 12.18 / 7.89 / 72.08 / 1.82 |
| transformer 720 | 10.88 / 13.11 / 65.01 / 1.61 | 12.55 / 13.87 / 59.18 / 1.67 | 14.99 / 14.05 / 61.66 / 1.53 | 15.54 / 14.36 / 57.07 / 1.57 |

## Paired severe-miss differences and profiler scope

Descriptive paired-family percentile intervals, fixed three seeds, 1,000 resamples. Values in points. Negative is fewer severe misses for the first named arm.
| Contrast | dense-accurate | dense-degraded | gap-accurate | gap-degraded |
|---|---:|---:|---:|---:|
| cog_384_minus_48 | +0.23 [+0.08, +0.38] | +0.27 [+0.06, +0.50] | +0.17 [-0.03, +0.37] | +0.25 [-0.00, +0.49] |
| cog_720_minus_48 | +2.24 [+2.03, +2.48] | +2.27 [+1.95, +2.58] | +1.79 [+1.52, +2.09] | +1.45 [+1.15, +1.76] |
| cog_minus_sdpa_384 | -0.87 [-1.03, -0.70] | -0.61 [-0.86, -0.36] | -0.85 [-1.07, -0.63] | -0.23 [-0.55, +0.04] |
| cog_minus_sdpa_48 | +0.15 [+0.02, +0.27] | +0.13 [-0.04, +0.32] | +0.14 [+0.01, +0.29] | +0.28 [+0.10, +0.48] |
| cog_minus_sdpa_720 | -6.17 [-6.54, -5.82] | -4.33 [-4.74, -3.94] | -4.85 [-5.21, -4.49] | -3.36 [-3.77, -2.97] |
| transformer_384_minus_48 | +1.24 [+1.04, +1.41] | +1.02 [+0.77, +1.26] | +1.16 [+0.94, +1.39] | +0.76 [+0.51, +1.02] |
| transformer_720_minus_48 | +8.56 [+8.09, +9.02] | +6.73 [+6.23, +7.21] | +6.78 [+6.32, +7.28] | +5.09 [+4.62, +5.59] |

Forward operator FLOPs at shape 1×16 are profiler-supported counts, not whole execution FLOPs. Fused attention, reductions, elementwise operations, and data movement can be omitted. Do not compare as total compute.

| Width | COG supported forward FLOPs | SDPA supported forward FLOPs |
|---|---:|---:|
| 48 | 1,546,800 | 1,494,544 |
| 384 | 94,949,424 | 94,531,600 |
| 720 | 332,858,928 | 332,075,536 |

## Verification

Private offline Kaggle version 1, CUDA required, Tesla T4, Torch 2.11.0+cu130, 1669.50 seconds measured script time including fixture generation, all 18 training/evaluation fits and rules. Script time is not serving latency. GPU native receipt is from Kaggle, not a TrainLab qualification. Detached implementation reproduced 920 numeric endpoints in 92 arm/seed/regime cells and checked source/actor hashes, deterministic fixture bytes, actual optimizer steps, data-draw equality and causal request caps.

- source SHA-256: 16abb2028f4ea21fdb18d762d78b5304c6afe9cef345d38536bb95d6905f2aff
- native REPORT SHA-256: 7f152a824e9a2c425e74ad3e918cd3f75cbbe41f696c52850a918f462e205149
- ACT SHA-256: d6853b72f178ea0af1327dee7873616f686b4a8206a031fedc70c9e5ac5feba2
- detached RECOUNT SHA-256: d626c2f78edcf4cbfef0e3c94d31b637edce414a90527a4991bf569672594065

