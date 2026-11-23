# Independent training-audit and diagnostic review

Reviewed on 2026-11-23 before the evaluation lock and before any learned held-panel scoring. This is a source and fixture review, not a completed-checkpoint audit. No real checkpoint was loaded and no candidate was scored during this review. The numerical replay below uses only the fixed Python sampling schedule and TRAIN inputs.

## Verdict

The independent audit is suitable to run once all six fixed endpoints and any admitted interruption-recovery receipts exist. Its checks support provenance, accounting, update and inactive-board claims. Passing it will not establish meaningful learning, task accuracy, generalization, or architecture adequacy. Those require the separate, unchanged qualification gates.

The audit initially allowed missing active optimizer states, incomplete AdamW moments, and active parameter steps below 1,200. These concrete gaps were repaired in the new audit and its fixtures. The repaired version requires all and only the active states, 1,200 steps per active parameter, finite moments with matching shape and dtype, and the declared AdamW defaults. The native/interface name mapping is valid for the pinned sources, which have no registered buffers, tied parameters, state-dict hooks, or duplicate parameter aliases. It is not a generic mapping algorithm for arbitrary models.

The saved Python RNG must match the exact draw replay. The saved Torch RNG receives an identity comparison between arms; that comparison is explicitly observational, not an independently reconstructed initialization RNG history. Numerical tensor changes are correctly described as updates, without equating changes to useful learning.

## Draws and accounting

The replay matches the locked trainer's call order: four family draws, then four Shakespeare-window starts, on each of 1,200 updates from `random.Random(20090017 + seed)`. The draw digest uses the same JSON representation. The LM start range matches the implementation, including its exclusion of the final otherwise possible window.

The audit distinguishes supplied observation/question characters, their padded tensor positions, teacher-forced decoder inputs, answer target characters including the period, the actual generation prefix, actual generated characters, and rehearsal positions. The actual generation path is charged independently of the discarded teacher-forced branch. Variable output lengths are checked against decoder-call and physical-position identities. Since generated training strings were not saved, this portion validates the logged dynamic counts and bounds; it does not reconstruct those strings or independently reproduce generation accuracy.

The fixed sampler implies the following coverage for both arms of each seed. These are prospective draw-replay counts, to be reconciled with completed traces and RNG payloads by the audit.

| Seed | Distinct families drawn from 2,048 | Sampled families in fixed 128-family probe | Probe families never sampled | Probe families sampled after the first 100 updates |
| --- | ---: | ---: | ---: | ---: |
| 17 | 1,834 | 113 | 15 | 110 |
| 29 | 1,842 | 119 | 9 | 117 |
| 41 | 1,864 | 115 | 13 | 113 |

The adequacy gate remains on all 128 fixed TRAIN-pool families under each arm's declared training state policy. It is a training-distribution adequacy gate, not a pure memorization-fit gate. Sampled/unsampled membership is descriptive and must not filter the probe, alter thresholds, or select an endpoint.

## Inactive board and endpoint admission

The audit compares every disabled native-board tensor byte-for-byte with the paired parent and rejects optimizer state on that board. It verifies the unchanged architecture/tokenizer inventory, 1,077,834 stored values, 80,160 inactive values, and 997,674 active values. Raw final and optimizer payloads are inspected only after the lightweight admission has checked all six completed endpoints, finite contiguous traces, original source/data hashes, and any recorded recovery provenance.

Successful retained training-loop CPU is separate from archived interrupted-attempt lower bounds. The whole-process CPU measured for a replacement attempt includes initialization and must not be added again to its retained training-loop CPU. A negative signal proves abnormal termination; it does not identify the underlying cause.

## Locked state interventions

The intervention script is correctly scoped to the two observations immediately following the first two generated answers in each of the first 32 TEST families. It compares the same text after natural, full-state reset, native-layer reset, native-hidden-only reset, and decoder reset. Modified states are discarded after that one encoding. The unchanged natural path supplies all subsequent state and ledger updates.

The pinned `FusionModel._backbone` and `encode_event` return new states rather than modifying the supplied state. Consequently, evaluating the natural diagnostic branch before the ordinary natural step does not advance the real state twice. Model evaluation mode disables dropout for the comparison.

Entity margins are interpretable only on the meaningful SET/DELETE subset. A HOLD operation has no factual key target, so the locked script's all-row entity-margin aggregate has no useful semantic interpretation and must not support a claim. All-row operation accuracy remains a legitimate separate measure. Decoder-reset semantic equality is a structural negative control for this fixed model, not evidence that recurrent decoder state is irrelevant in a different architecture.

These interventions condition on the actual prior generated history and its possibly wrong ledger. They diagnose the next observation parser. They do not evaluate an alternative repaired whole session, establish end-to-end accuracy under a reset policy, or justify a post-hoc scoring rescue. Two boundaries and two worlds from a family are correlated observations, not four independent families.

## Summary compatibility repair

The approved operational recovery can preserve one incomplete final log write. The prior summary reader attempted to parse every line and would reject an otherwise valid admitted archive. The unlocked summary now retains only valid complete rows, permits one final unterminated partial line, rejects malformed complete/interior lines, checks row order, clocks and dynamic cost counters, and reconciles retained rows with the admitted `original_trace` receipt. It reports discarded trailing bytes without inventing their compute cost.

The compatibility fixtures cover complete rows with and without a final newline, the allowed incomplete last write, completed/interior corruption, a blank completed row, a fragment with no complete row, and a noncontiguous update. The final focused summary suite passed 15 tests. The existing two Torch JIT deprecation warnings arise from fixture imports and do not change the checks.

Exact reviewed source hashes, fixture receipt hashes, and the unchanged training-lock verification are recorded in `STATISTICS_COMPATIBILITY_REVIEW.json`. No locked training, evaluation, data, protocol, or intervention source was changed by this review.
