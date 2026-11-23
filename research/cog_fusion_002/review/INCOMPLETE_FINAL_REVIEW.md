# Final independent review of the incomplete campaign

Reviewed on 2026-11-23 after the real partial integrity audit and generation of the report and figure. **Accepted as an incomplete-study record. No learned capability or campaign qualification is established.**

## Evidence reviewed

This review examined the actual [incomplete-campaign audit](../evidence/INCOMPLETE_CAMPAIGN_AUDIT.json), the original campaign/recovery receipts, all five completed run receipts and logs, the two incomplete trace prefixes, bound checkpoint and optimizer file hashes, the locked sources, [RESULT.md](../RESULT.md), and the rendered [training figure](../evidence/COG_FUSION_002_LEARNING.png). It also inspected the separate [execution-handle observation](../evidence/EXECUTION_HANDLE_OBSERVATION.json), whose interpretation correctly stops at unavailable tool control and does not assert an operating-system exit or cause.

Before the partial audit ran, two input-validation gaps were corrected in that new audit alone: nonfinite completion clocks are now rejected, and the copied retry identity must match the identity admitted before the copy. Twelve focused fixtures passed. The final engineering receipt records 111 passes, zero failures, zero errors, and zero skips. The original all-six admission, training, evaluation, recovery, and summary sources were not relaxed.

## Independent reconciliation of the real output

The reviewer executed [reconcile_incomplete_evidence.py](reconcile_incomplete_evidence.py), which imports neither Torch nor the study's accounting/audit helpers. It independently rebuilt the Python draw sequence and supplied character, target, padding, generation-prefix, and rehearsal counts from the fixed TRAIN data. It reconciled all 6,000 completed-run rows plus the original 13-row and replacement 107-row prefixes, checked every input hash bound by the real audit, and verified the unchanged 22-source training lock and frozen summary.

The resulting [reconciliation receipt](INCOMPLETE_EVIDENCE_RECONCILIATION.json) passed. It agrees with the report's exact per-run totals:

| Retained completed fit | Updates | Training-loop CPU seconds | Physical token positions |
| --- | ---: | ---: | ---: |
| reset_training, seed 17 | 1,200 | 2,270.259846458 | 5,533,470 |
| continuous, seed 29 | 1,200 | 2,395.073955330 | 5,521,215 |
| reset_training, seed 29 | 1,200 | 2,262.215207951 | 5,521,201 |
| continuous, seed 41 | 1,200 | 2,641.009679133 | 5,525,339 |
| reset_training, seed 41 | 1,200 | 2,066.796605126 | 5,525,375 |
| **Completed total** | **6,000** | **11,635.355293998** | **27,626,600** |

The original continuous-seed-17 prefix contributes 13 recorded updates, 48.195140857 CPU seconds, and 61,417 positions. The permitted replacement snapshot contributes 107 recorded updates, 197.215294733 CPU seconds, and 492,043 positions. The observed combined lower bounds are therefore **11,880.765729588 training-loop CPU seconds and 28,180,060 positions**. The two prefixes are separate attempts initialized from the parent; their steps are not cumulative progress toward an endpoint.

For the completed fits, the replay confirms 1,834, 1,842, and 1,864 distinct TRAIN families sampled for seeds 17, 29, and 41. The corresponding fixed 128-family probe exposure is 113, 119, and 115 families. No probe was evaluated. The seed-17 full-exposure counts apply to its completed reset-training arm; they do not invent a completed continuous arm.

The real raw-payload audit reports all five endpoint configurations unchanged, 85 active optimizer states at step 1,200, complete finite AdamW moments, exact saved Python sampling states, and all 29 inactive native-board tensors byte-identical to their parents with no optimizer state. All 85 active parameter tensors changed numerically in each completed fit. The reviewer source-checked those tensor tests and bound their saved claims to the actual checkpoint/optimizer hashes; this second reconciliation did not deserialize the weights again. Numerical updates do not establish useful learning.

## Report and figure assessment

The report clearly identifies the missing continuous-seed-17 endpoint, separates its recorded original signal from the replacement's unknown status, and marks every planned scientific gate **Not evaluated**. The actual supplied-data and work-accounting tables agree with the raw evidence. Python sampling-state verification is distinguished from the observational Torch RNG identity. Actual generated characters remain model-dependent, and a period is consumed when emitted; termination may instead occur at the 12-character cap.

The figure is readable without clipped labels. It shows the five 1,200-update endpoints, the observed 107-update replacement, and 50-update averages of semantic training loss and TRAIN language-rehearsal NLL. Its curves are descriptive training observations under each arm's own training state policy. They do not compare held performance, establish language preservation, or demonstrate a specific advantage of continuous training. The 13-update original attempt is retained separately and is not added to replacement progress. The unchanged architecture count must not be confused with unchanged numerical weights.

## Remaining limits and disposition

The original negative return code supports termination of the original attempt. The replacement still has no supported liveness/termination receipt. Stable copies establish matching bytes during a bounded observation window; they do not prove that a producer is dead. No out-of-memory event, timeout, or other cause is established.

CPU totals cover recorded training loops. They exclude initialization, unreported work, calibration, tests, audit work, and orchestration. Concurrent process CPU is not elapsed campaign wall time. Generated training strings were not retained, so their logged accuracy and dynamic cost counts can be checked for consistency but cannot be independently reconstructed from this archive without rerunning inference.

The five individual integrity passes cannot satisfy the original six-endpoint gate or substitute for a three-seed scientific comparison. The preserved inference/evaluation implementations remain available, but no held learned-model scoring, TRAIN adequacy test, regression test, preservation score, state intervention, bootstrap comparison, or qualification has been run for this incomplete cycle. The [earlier disposition and conditional amendment proposal](INTERRUPTION_DISPOSITION_REVIEW.md) remains applicable and unexecuted.

The final review binding, including report, figure, source, and receipt hashes, is recorded separately in `INCOMPLETE_FINAL_REVIEW.json`.
