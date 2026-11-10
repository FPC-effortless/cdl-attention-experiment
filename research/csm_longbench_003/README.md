# CSM-LONGBENCH-003 results

Twelve final checkpoints; three seeds; four equal-capacity controls. Each fit used 262,144 training targets, 32 times the initial pilot. Final DEV covers 32,768 targets distributed across the exposed split. No protected-test scoring.

| Arm | Mean NLL ↓ | Seed range | Accuracy | ECE ↓ |
|---|---:|---|---:|---:|
| instant | 3.364819 | 3.347198–3.373972 | 15.08% | 0.01084 |
| single | 3.361740 | 3.332891–3.391481 | 15.06% | 0.00934 |
| multi | 3.346581 | 3.297599–3.371231 | 15.20% | 0.00750 |
| multi_tb64 | 3.317109 | 3.267241–3.371697 | 15.10% | 0.00686 |

## Paired differences from immediate

Negative NLL differences favor the candidate. Three seeds do not support precise uncertainty estimates.

| Arm | Seed11 | Seed23 | Seed37 |
|---|---:|---:|---:|
| single | +0.018194 | +0.013649 | -0.041081 |
| multi | -0.002374 | +0.024033 | -0.076374 |
| multi_tb64 | -0.060898 | +0.024499 | -0.106731 |

## Memory interventions

Post-token256 NLL change relative to the untouched rollout. Positive means the intervention hurts. Trace swap is a one-time cyclic swap across different real-text streams. Record reset zeros values, preserving links, cursor and trace. These out-of-distribution interventions measure total downstream effects, not isolated semantic memory.

| Arm | Trace reset | Trace swap | Record reset |
|---|---:|---:|---:|
| instant | +0.000000 | +0.000000 | +0.007207 |
| single | +0.021579 | +0.000253 | +0.006797 |
| multi | -0.001653 | -0.000138 | +0.003208 |
| multi_tb64 | -0.000643 | +0.000042 | +0.002459 |

## Reproduction and boundaries

`experiment.py --seed 11` (also23/37) repeats fits. `test_contracts.py` verifies split and state contracts. `replay_and_benchmark.py --timing` independently reloads final checkpoints and measures isolated CPU steps; run after training exits. All checkpoint hashes and source/data/schedule identifiers are preserved.

This is an exposed-DEV study on one small real corpus, with fixed budget and no tuning. The longer runs also broaden the sampled training data, so improvement over the old pilot cannot be attributed to update count alone. The three main arms have equal learned-array shapes and nominal operation counts; hardware FLOPs are unmeasured. The tb64 arm changes backward truncation and thus does not have an identical backward computation graph. CPU training ran in concurrent independent processes; throughput is not a clean architecture speed comparison. Deployment performance requires an optimized runtime.

Historical pilot CI failed its exact cross-host floating-point assertion without reporting a delta; its root cause is unconfirmed. The revised replay reports numerical errors with absolute tolerance1e-6, registered before these fits. A subsequent independent run passed all nine pilot replays with zero observed error. Historical failure records remain available. Resumed single-timescale fits have missing early per-update loss/time telemetry; those entries are null, not reconstructed.

## Interpretation and reference baselines

The strongest mean result is multi_tb64, but its advantage is seed-dependent. The longer 256-token gradient window does not improve the multiple-timescale condition here. Clearing the multiple-timescale trace slightly improves mean post256 loss, so this result does not demonstrate that beneficial retained history explains the gain.

Train-only unigram mean NLL: 3.353891; bigram: 2.504695. These count controls see the same sampled targets, but are not parameter- or compute-matched. The bigram has 65,536 count cells. They establish practical reference performance, not a fair architecture-superiority claim.

## Exploratory post-fit collapse diagnostics

All 12 models choose ASCII space as their most frequent top-1 prediction. Its prediction share ranges from 95.86% to 99.98%. NLL can improve while the most likely token barely changes; this is why the loss improvement is not yet useful language competence.

| Seed | Arm | Space prediction share | Distinct top-1 bytes | Effective destinations | Effective operators |
|---:|---|---:|---:|---:|---:|
| 11 | instant | 99.90% | 5 | 5.59 | 1.91 |
| 11 | single | 95.86% | 4 | 1.60 | 1.57 |
| 11 | multi | 99.98% | 6 | 1.49 | 1.00 |
| 11 | multi_tb64 | 99.56% | 7 | 2.54 | 1.52 |
| 23 | instant | 99.64% | 7 | 4.22 | 1.97 |
| 23 | single | 99.16% | 6 | 2.69 | 2.59 |
| 23 | multi | 99.32% | 8 | 2.61 | 1.00 |
| 23 | multi_tb64 | 99.95% | 7 | 3.39 | 1.00 |
| 37 | instant | 97.35% | 11 | 1.04 | 1.01 |
| 37 | single | 99.79% | 7 | 3.71 | 1.98 |
| 37 | multi | 97.43% | 6 | 1.28 | 1.17 |
| 37 | multi_tb64 | 98.46% | 8 | 1.48 | 1.18 |

Effective counts are exp(entropy) of selection frequencies, not proof of semantic specialization. Several runs effectively use only one operator or destination. Complete raw frequencies and routing margins are in prediction_diagnostics.json. These checks were added after inspecting the first completed fits and are explicitly exploratory.

## Replay and measured inference

All 12 final checkpoints replayed with zero observed local NLL and per-excerpt error. Model/optimizer/RNG/schedule/state are saved. No training occurs during replay.

| Arm | Mean of seed median step μs | Mean of seed p95 μs | Retained tensor bytes |
|---|---:|---:|---:|
| instant | 286.08 | 408.24 | 1616 |
| single | 288.08 | 407.12 | 1616 |
| multi | 282.37 | 401.65 | 1616 |
| multi_tb64 | 284.25 | 398.50 | 1616 |

Timings are isolated single-thread CPU eager supplied-token steps after training exited, with warmup; they exclude sampling and are not mobile-device or optimized-kernel measurements. The immediate adapter still allocates its trace; a specialized deployment could omit that state.

## Next decision

Do not promote this as a strong language model or scale compute blindly. First test whether CSM can overfit a small real training excerpt beyond a count baseline, then isolate hard routing, admission, and the readout bottleneck using matched ablations. Check gradient magnitudes and routing margins against prediction diversity. Keep real-data quality, semantic record use, and state-preservation tests separate.
