# COG-PULSE-004A-R2: frozen value probe

**The frozen reader stores more value information than its original output head uses.** With the correct record supplied, a small linear probe on the 64-dimensional route × payload vector decoded 60, 62 and 60 of 64 current records whose owner–value pairing was absent from the probe's training records. The unchanged native head, also given the correct record at full strength, got 36, 41 and 21. Restricting that same head to the same 16 value choices raises it to 44, 51 and 35, leaving a gap in all three checkpoints.

## What ran

Three already-trained R2 content-aware readers (seeds 173, 271, 907) were frozen. With no query or key, timestamp, slot or ID side channel as probe features, a fixed ridge solver (λ=1, unpenalized intercept, TRAIN-only normalization) read either the 16-dimensional payload or the 64-dimensional ordered route × payload. The supervised value label was the visible source value, not a label derived from the query. The 304 distinct original TRAIN records included all occupied roles; duplicate presentations did not get extra weight. The original encoder generated TRAIN vectors. Before any CAL feature projection, each set of probe coefficients was sealed. Evaluation used only the archived exact CAL record vectors and fixed coefficients; no checkpoint or labels in that evaluator.

The original open CAL corpus is separate from the probe's TRAIN roster, but was previously exposed in the R2 research cycle. It contains 215 distinct occupied records. Across all roles, 57 exact records recur from TRAIN, 51 are exact-new with a seen owner–value pairing, and 107 have an entirely new owner–value pairing. The main **current-record** comparison is 64 exact raw records, representing 57 new owner–value pairs, from 22 constructed roots. All 16 value symbols were supported in TRAIN; this is transfer to new combinations, not new value vocabulary. The comparison uses the same factual contexts, and a native head record counts strictly only if all its contexts are correct. Each exact shared record had zero measured payload or route drift between batches, and exact repeats within CAL also had zero measured drift.

## Results

Exact classifications in the 64-record current cohort, in seed order:

| Readout | 173 | 271 | 907 |
|---|---:|---:|---:|
| Native head, correct record at full strength, full 48 outputs | 36 | 41 | 21 |
| Same native logits, restricted to the 16 value outputs | 44 | 51 | 35 |
| Linear probe, payload only (16 dimensions) | 56 | 51 | 55 |
| Linear probe, route × payload (64 dimensions) | **60** | **62** | **60** |
| Nearest centroid, same 64-dimensional features | 54 | 63 | 52 |
| Linear probe, same 64-dimensional features, labels shuffled in TRAIN | 2 | 0 | 3 |

Across all 107 occupied records with unseen owner–value pairs (regardless of their role), the 64-dimensional probe decoded 90, 103 and 91; payload alone decoded 89, 86 and 79. The 64-dimensional probe finished all factual cases in 19, 23 and 21 of 24 current/obsolete/foreign roots, versus 9, 11 and 3 of 24 for the native full-strength head (15, 19 and 8 after restricting the same logits). That rooted comparison still supplies the correct record for every read; it does not test whether the reader can identify it. In the strict 64-record panel the centroid beat ridge by one in seed 271, so the finding does not hinge on the particular ridge solver.

## Decision

A missing-value-code explanation is not sufficient: much of the value is already recoverable from the frozen packets, including unfamiliar owner–value combinations. The route expansion improved the tested ridge probe in every seed, but the expanded vector is larger and the fit is directly value-supervised. This result does not establish that the original head is the sole bottleneck, that the model can choose the right packet, or that it knows when to return NONE. The probe has fewer possible answers and no need to reject irrelevant evidence.

Next I would test a **value-preserving readout under ordinary joint answer training**, keeping the existing content router as a matched baseline and retaining externally supplied full-strength records only as a diagnostic ceiling. Preregister factual pair and genuine occupied no-match gates on new roots. That next test should ask whether the packet-to-answer gain survives native selection and unrestricted outputs, rather than treating this diagnostic decoder as a deployed memory system.

Six sequential isolated single-core diagnostic roles completed once (three TRAIN solves, three fixed CAL evaluations; ≤15 seconds per role). No PULSE parameters changed, no new protected evaluation or paid run occurred, and the measured role times are not a serving benchmark.
