# COG-BIND-001V: learned token weighting did not fix temporal role binding

**Decision: 0/3 seeds qualified.** Both readers selected the action object on every held row whose target event actually included a locative modifier (512/512 per fit). That result did not transfer to unseen temporal constructions. Learned candidate-conditioned weighting was worse than uniform pooling in aggregate on the novel syntax, even when retrieval was not a problem because the scorer already knew the correct event.

The task is to answer which item an actor handled in the main action, not which item appears in the preceding trigger or the location phrase. Each strict pair swaps action and trigger objects in a sentence with identical word counts, fixed event order and unchanged question. Both versions must be answered correctly.

| Seed | Familiar strict pairs, pooled → learned (/512) | Novel strict pairs, pooled → learned (/512) | Locative strict pairs, pooled → learned (/512) |
| --- | ---: | ---: | ---: |
| 19371 | 512 → 512 | 256 → 256 | 512 → 512 |
| 19471 | 512 → 512 | 348 → 302 | 512 → 512 |
| 19571 | 510 → 512 | 254 → 256 | 510 → 512 |

Novel strict pairs fell from **858/1,536 to 814/1,536** pooled across seeds; these are repeated observations of the same held case families, not 1,536 independent cases. Paired effects by seed were 0.00, −8.98 and +0.39 percentage points. Family-clustered bootstrap intervals were [0.00, 0.00], [−14.45, −3.13] and [0.00, 1.17]. No learned seed reached 95% exact individual rows or 90% strict pairs on novel syntax. The learned arm's individual counts there were 512, 648 and 512 of 1,024; complete families were 0, 23 and 0 of 128.

## What failed

- In novel construction 6, two learned seeds picked the **trigger's object on all 512 rows each**. The same two seeds got all 512 rows in construction 7. The third learned seed solved all 512 rows in construction 6 but picked the trigger's object on 376/512 rows in construction 7. Across all 1,400 erroneous novel rows from the learned runs, the reader picked the trigger object, not a random token. This isolates a phase/role selection error on this fixture; two-noun scoring after the fact did not rescue it.
- On novel syntax, replacing learned weighting with uniform pooling at the same trained weights changed strict pairs 256→256, 302→332, and 256→256. Removing the aggregate while preserving the common query term produced 256, 336, and 256. Cyclic reassignment of weights across event tokens yielded 256, 326, and 256. Learned aggregation did not provide a reproducible net gain; aggregate counts can mask individual pair gains and losses.
- The correct item's peak anchor landed on the action verb only 512, 510 and 98 of 1,024 novel rows across seeds, and often landed on the trigger or item instead. The name “anchor” should not be mistaken for learned verb grounding. The same conclusion does not rule out learning an explicitly supervised phase distinction.

## Scope and custody

Six matched CPU fits, 14,655 trainable parameters each, 384 AdamW updates and 49,152 matching-event answer exposures per fit. Both arms use the same raw character-composed local stack, common question fusion and answer-position loss. The contrast is uniform versus learned candidate-conditioned aggregation across positions within a visible event boundary. It is not attention versus no attention, does not provide typed tokens, roles, or candidate masks at inference, and does not establish equal deployment cost. The native actor scores both events; the correct event is used only by the scorer. Test conditions are aligned views of 128 withheld families, not independent panels. There are 256 training and 64 development families, with disjoint actor/item role atoms by split. The letter inventory includes held spellings but no learned lookup embedding for held words.

First- and last-item shortcuts do not solve any complete held family. The exact finite temporal grammar does solve the fixture. It is an engineered validity witness, not learned generalization. The locative insertion uses familiar base syntax; this bounded success does not imply robustness to arbitrary paraphrases.

Native Forge diagnostics and independent input checks preceded TrainLab execution. All six native fit records met declared and realized fixed-budget criteria without scientific qualification. The initial actor pass in the first two runs stopped on a filename-metadata bug after writing only unlabeled development output. A separately locked actor-only correction reused the identical six checkpoints and produced byte-identical development arrays for both affected runs. No held predictions were repeated, no fit was repeated, and the final scorer joined truth to 24 sealed prediction bundles.

The next mechanism to test is an explicitly learned **trigger vs. main-action phase selector**, followed by object extraction, rather than another pooled-pointer variant. Compare phase supervision with a matched model given the same diverse training examples and only answer supervision, and include a competent small attention control. Cross timing cues and clause order during training, then hold out entire new syntax families and test phase-cue counterfactuals. Gate native event selection and full inference cost separately.

The independent recount matched all 24 bundles and the aggregate. The private Research Lab ledger records 14 evidence nodes and 19 provenance edges. Recount SHA-256 `510432d5fe02051c38cbd210918aa3e8cb2187cc993a8365d54db080c31f2bdc`.

Data pin `8d3c7aa4cee1684c6c2bac3991546e780f1be9b31289fccf8a9c2f2750ee340c`; source lock `3eb47c23565fd9a9b8ffa91deccb6254932e242c68a298e15bb5f05abe81dd54`; score `19f29127ee1afec23a53daa496d7d8f1229f87dd1a3b7112fcfc73c461680593`. The private evidence package retains exact data, checkpoints, wires, counts, and replay instructions.
