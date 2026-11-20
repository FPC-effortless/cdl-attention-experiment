# COG-062: retaining rank one beat both rerankers

The B-informed binary diagnostic ran to completion under separate independent approval for fitting, lexical prediction, and gold scoring. The historically exposed publisher-TRAIN re-split was fixed; this is NOT_QUALIFIED, not a fresh external transfer result. No original publisher DEV or TEST was used by this pipeline. The earlier three-class failure remains retained and was never retried.

| On true changed slots | All experimental E (863) | Exact-clean dialogue groups (755) | Shortlist hits only (769) | Rescues / spoils on full E vs rank 1 |
|---|---:|---:|---:|---:|
| Retain shared proposer rank 1 | 521 | 451 | 521 | reference |
| Matched canonical-history reranker | 505 | 435 | 505 | 27 / 43 |
| Ordered-history reranker | 505 | 434 | 505 | 29 / 45 |
| Ordered weights with chronology removed at inference | 513 | 441 | 513 | 28 / 36 |

The primary ordered-minus-canonical difference was **0 percentage points** (descriptive paired 95% dialogue-group bootstrap interval −0.95 to +0.94 pp), missing the prespecified +3 pp/lower-bound-positive gate. Compared with rank 1, ordered is −1.85 pp (descriptive 95% interval −3.84 to +0.12 pp). Descriptive sensitivity excluding groups with any exact shared feature tuple retained the same direction against the rank-one reference. This is not semantic decontamination: Forge's original exact-content check was FAILED and retained.

Top-two proposal oracle coverage was 769/863. All 94 shortlist misses remained errors for all arms. Of 248 cases where gold was rank two, ordered rescued 29, canonical 27, and order-removed 28. This gain is offset by spoiling more rank-one answers. Against 21,891 unchanged eligible slots, false proposals were 94 rank-one, 94 canonical, 99 ordered and 93 order-removed. These are unverified write proposals, not committed or correct state writes. Neither full request inference cost nor actual write cost was measured.

Training mask candidate-hit rank2 rates were 195/11,224 (area) and 170/11,217 (price), both under 2%. The fixed 0.5 switch threshold was not tuned. Both six-classifier A/B fits met all execution criteria and converged; checkpoint and execution observations were byte-identical. Original three-class variant halted on 44 available B neither targets (100 required), and yielded no E outputs. Do not use newly exposed E to report a fresh confirmatory gain after choosing another switch threshold or cohort.

TrainLab wall: binary fit 41.215988 s, fit replay 39.582040 s, batch prediction 5.383055 s (11,378 turns), independent scoring subprocess 1.828391 s. These are child-stage wall measurements, not individual query, full application, or production write timings.

Separate TrainLab receipts were verified and a native Veritas OBSERVED, NOT_QUALIFIED private envelope was indexed in Research Lab. Private receipt and evidence identifiers are not included in this release candidate.

Recommendation: retain rank one as the stronger reference under this fixed design; investigate a better candidate generator or separately preregister a confidence-sensitive policy against genuinely newly acquired evaluation data. No switch policy is promoted by this study.

## Package boundary

Source, protocol, official source identifiers, dependency lock instructions, and aggregate results only. No dialogue or text rows, lexical input/label views, predictions, checkpoints, private reviewer records, execution receipts, private workspace inventories, or credentials. Scientific scripts match checked private copies; private admission wrappers are not shipped and must not be bypassed. See REPRODUCE.md.

The independent result review recomputed the changed-slot and unchanged-proposal counts against the retained predictions and evaluation labels, and checked the canonical native evidence receipt. Those underlying artifacts remain private.
