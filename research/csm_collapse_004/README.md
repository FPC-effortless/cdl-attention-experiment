# CSM-COLLAPSE-004 results

Twelve matched-parameter mechanistic fits on one authentic 128-target training excerpt. Final endpoint only; no selection by development scores. Each fit sees the same excerpt 256 times (32,768 target instances).

| Variant | Mean fit NLL ↓ | Mean fit accuracy | Fit criterion passes | Adjacent TRAIN NLL | Exposed DEV NLL |
|---|---:|---:|---:|---:|---:|
| hard | 0.104485 | 98.18% | 3/3 | 4.593166 | 6.064517 |
| soft_routes | 0.112517 | 98.70% | 3/3 | 4.217646 | 5.818282 |
| soft_admission | 1.217310 | 69.79% | 2/3 | 3.926762 | 5.147215 |
| residual_front | 0.057807 | 97.40% | 3/3 | 4.362517 | 5.954064 |

Fit criterion fixed before training: NLL<0.7 and accuracy>=80%. Fitting one real excerpt is a debugging test, not a real-world generalization result.

| Seed | Variant | Fit NLL | Accuracy | Space share | Distinct predictions |
|---:|---|---:|---:|---:|---:|
| 11 | hard | 0.095037 | 99.22% | 12.50% | 32 |
| 11 | soft_routes | 0.167378 | 97.66% | 12.50% | 30 |
| 11 | soft_admission | 0.619821 | 85.94% | 13.28% | 26 |
| 11 | residual_front | 0.068228 | 97.66% | 11.72% | 31 |
| 23 | hard | 0.065490 | 99.22% | 12.50% | 31 |
| 23 | soft_routes | 0.059050 | 100.00% | 12.50% | 32 |
| 23 | soft_admission | 0.164394 | 97.66% | 12.50% | 31 |
| 23 | residual_front | 0.056484 | 96.88% | 12.50% | 31 |
| 37 | hard | 0.152928 | 96.09% | 11.72% | 30 |
| 37 | soft_routes | 0.111123 | 98.44% | 12.50% | 31 |
| 37 | soft_admission | 2.867716 | 25.78% | 51.56% | 3 |
| 37 | residual_front | 0.048709 | 97.66% | 12.50% | 30 |

## What was controlled

All variants have 34,479 learned parameters and identical initial arrays per seed, inputs, supervision, optimizer and update budget. Their computation differs; actual FLOPs are not matched. Soft routing changes address/update semantics and removes the original single-slot frame guarantee. Soft admission also removes the exact binary no-write decision at finite logits. The residual readout adds no parameters but changes the information and gradient path. There is no claim of equal-compute architectural superiority.

The hard variant exactly matches the archived baseline in forward values and backward gradients on checked fixtures. Causal-prefix, finite-gradient, initialization/count and independent-stream checks pass for the variants. Checkpoints include model, optimizer, RNG, curves and timing; they support resuming at saved presentation boundaries.

## Reproduction

`python research/csm_collapse_004/test_contracts.py` checks implementation contracts. `run.py --seed 11` (also 23/37) resumes missing fits and skips completed outputs. For a new fit use a separate source copy without seed output directories. `replay.py` validates all checkpoints and final panel scores without fitting. `frozen_audit.py` audits long-run checkpoints without optimization.

No protected TEST scoring, paid compute, GPU, merge or deployment. The adjacent TRAIN excerpt and exposed DEV are diagnostic transfer checks, not selection signals. The prior long-run findings are preserved unchanged.

## Frozen long-run audit

No optimizer updates. Measurements use the same authentic 128-target training excerpt. Proposal saturation is the fraction of pre-tanh activations with magnitude > 3; at this threshold tanh derivative is below 0.01. This correlation is not a causal attribution of collapse. Readout standard deviation is averaged over coordinates across time.

| Seed | Long-run arm | Proposal saturation | Readout coordinate std | Frontend gradient norm | Proposal-output gradient norm |
|---:|---|---:|---:|---:|---:|
| 11 | instant | 84.81% | 0.155343 | 0.104144 | 0.009453 |
| 11 | multi_tb64 | 41.55% | 0.184862 | 0.150097 | 0.075521 |
| 23 | instant | 12.33% | 0.245270 | 0.171266 | 0.051627 |
| 23 | multi_tb64 | 29.91% | 0.189620 | 0.086679 | 0.013490 |
| 37 | instant | 1.54% | 0.176870 | 0.263841 | 0.079146 |
| 37 | multi_tb64 | 37.99% | 0.222817 | 0.184583 | 0.045936 |

## Post-fit local-context control

A lookup fitted and scored on this same excerpt reaches the following accuracy using only recent bytes. This is a descriptive memorization control, not held-out evaluation or a parameter-matched model. It was added after observing successful fitting.

| Context bytes | Unique contexts | Training lookup accuracy | Empirical conditional NLL |
|---:|---:|---:|---:|
| 1 | 32 | 46.88% | 1.243725 |
| 2 | 83 | 85.16% | 0.243790 |
| 3 | 102 | 95.31% | 0.069070 |

The 3-byte lookup already explains 95.31% of training targets. Exceeding this on the same exact inputs requires distinguishing some histories sharing the last 3 bytes, but does not establish generalizable retrieval, record semantics, or compositional reasoning.

## Interpretation boundary

The unchanged model passed all three fits, ruling out an unavoidable constant-output limitation on this diagnostic. The frontend residual reached high accuracy earlier and the lowest mean final fitting loss, while soft admission failed in one seed. None establishes adequate capacity for general language modeling: transfer losses remain poor. This study changes learning rate and data repetition relative to LONGBENCH-003; it cannot isolate which caused the earlier collapse. Within this study the four variants have matched learning settings.

The residual readout changes both the available information and gradient path and the input distribution seen by the output head. Any advantage cannot uniquely identify one of these as the cause. Soft routing changes the forward state dynamics as well as the gradient estimator. Further controlled experiments are needed before selecting a replacement architecture.
