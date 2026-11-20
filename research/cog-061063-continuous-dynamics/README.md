# COG-061/063: useful compressed state, failed causal adaptation

**Executed and preserved.** The narrow 061 factor-recoverability gate passed; the 063 causal-adaptation gate failed. This is a controlled simulator pilot, not authentic task qualification or discovery of named concepts.

## What was tested

Learners saw only 16 continuous sensor channels and two dense controls. Four hidden coordinates were mixed into the sensors, then combined with colored nuisance and observation noise. Neither control corresponded to one factor. The comparison used exactly the same 8,064 effective three-lag training targets from 64 training episodes for every fit. There were 64 independent held development groups, 32 with a coupling shift and 32 without. Four cloned action branches shared each initial state and disturbance realization. No future branch observations, regime labels, latent values or category labels entered the learner. Rank four was supplied to the compact methods. Calibration probes saw labels only after model weights were frozen; no test set was accessed.

## 061: the compact predictive state retained more factor information than PCA

| Predictor representation | Factor 1 R² | Factor 2 R² | Factor 3 R² | Factor 4 R² | Mean R² | Stored scalars |
|---|---:|---:|---:|---:|---:|---:|
| Predictive compression | .854 | .231 | .926 | .932 | .736 | 672 |
| Matched-rank PCA | .593 | .140 | .707 | .557 | .499 | 672 |
| Uncompressed three-lag NARX | .995 | .995 | .997 | .995 | .995 | 11,264 |

The fixed 061 screen required all predictive-factor R² values above zero and mean better than PCA. It passed. Raw history retained more information; its probe had more available features. This was a gain at equal *compact* storage, not unique-axis identification, discovery of the number of physical factors, symbolic concepts, or a general capacity ranking. The gate is descriptive and no 061 confidence bound or separate distribution-shift claim was preregistered.

## 063: adaptation overreacted and failed the causal forecast gate

Primary cell: 32 changed-coupling groups, 16 observed support turns, eight-step open-loop forecast.

| Readout | Intervention contrast RMSE ↓ | Factorial interaction RMSE ↓ | Branch-conditioned absolute RMSE ↓ | Warm request ms ↓ |
|---|---:|---:|---:|---:|
| Predictive adaptive | .2103 | .6638 | .5125 | .805 |
| PCA adaptive | .2006 | .2807 | .4186 | .760 |
| NARX adaptive | .7887 | 1.1581 | .8587 | 1.455 |
| Predictive frozen | .0765 | .0917 | .2176 | .698 |
| Action-blind persistence | .0436 | .0165 | .2522 | .024 |

The paired predictive-adaptive minus PCA effect error was +.0098 (95% group-bootstrap interval −.0773 to +.0920), not better. It did beat the tested NARX adaptive pipeline on that contrast (−.5784, interval −.7756 to −.3462), but lost to its own frozen version (+.1339, interval +.0761 to +.1858). The interaction point estimate was worse than PCA; the 5% unchanged absolute-error guard passed (.3873 vs .4070). No post-hoc gate replacement.

A useful diagnostic, not a new pass: frozen predictive one-step effect RMSE was .0123 versus action-blind .1610. At eight steps the zero-effect comparator won. Since controls were applied only on the first four steps and the stable generator forgets perturbations, the zero comparator's measured effect error is itself the RMS gold effect and drops to .0436 at step 8 and .0013 at step 16. Thus the candidate learned some immediate response, but recursive predictions did not preserve action-effect decay. The attempted local residual update made things worse; overfitting or mismatch is plausible but not causally isolated here. The next experiment should isolate direct multihorizon effect prediction from recursive rollout and compare a stable-decay baseline before adding adaptation; freeze a new panel rather than tuning on this one.

The CPU clock is a median over three imperfectly rotated repeats, 192 requests/arm, four plans per request. It includes warm handling of sensor context, support fit, model and rollout, excludes load and serialization, and is not a statistically established speed advantage. Action-blind is a cheap diagnostic, not an equally capable control. Measured offline process CPU: predictive .0670s; PCA .0252s; NARX .2049s. Inference cost is prioritized 12:1, not combined with quality using a fabricated exchange rate.

## Evidence and qualification

- Forge separate content-bound constructed TRAIN and DEV envelopes; independent reconstruction matched 128 episodes and 768 query branches. Probe calibration and target truth were admitted only to the scorer. Forge envelope tests are not causal validity or contamination certification.
- One stage at a time against frozen manifest `0f775c17040e52d2ea712f1ae42d4ae448899d372535298a6eaddd7513a10b04`; pinned local Veritas ecosystem `eec439492407b0aa26922c23b5ae40d95289f4eb`. Exact independent approvals preceded fit, prediction, and scoring.
- TrainLab fit `20261120T201119-c1d86fe81f68`, prediction `20261120T201747-5efa8fa3a431`, verifier `20261120T201945-80b8ea313879`; all succeeded, post-run mechanical criteria met, outputs hash verified, independent evidence nodes retained. The custom Veritas scorer was tested prospectively on known effects and malformed fixtures.
- Full raw inputs/models/forecasts/label truth remained local. No paid resources, protected test, publication, push, merge or refit. The campaign remains **NOT_QUALIFIED**, and these results do not complete all objectives of 061/063.

Independent analyst recomputation from retained prediction and truth artifacts matched all primary contrast/interaction errors, all three paired intervals, all 12 probe R² values and both gate decisions. This was separate arithmetic, not a second training seed or hardware rerun.
