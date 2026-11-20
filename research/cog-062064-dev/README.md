# COG-062/064: expiry is not a learned recovery solution

Executed development diagnostic, 20 November 2026. **Do not promote either expiry policy.** This advances the persistent-error diagnosis (064), not learned language competence (062). No new model was trained. No official TEST data was requested or evaluated.

## What ran
Frozen language heads from merged PR #72 supplied the same current utterance and two prior utterances to seven read policies. We used all 7,374 user turns across the 1,000 official MultiWOZ 2.2 DEV dialogues, two hotel slots only. These DEV data had already selected the admission thresholds and PR #73 soft filter; they are exposed development data, not independent qualification.

The new intervention marks an unsupported provisional value **unknown** after two or four consecutive HOLD predictions. Unknown counts as wrong for accuracy and separately as abstention, never as correct empty state. Any non-HOLD top prediction renews the timer, even when low confidence or repetitive. Equality with the verified store does not expire. This deliberately tests a simple bounded-carry policy, not corroboration or a new proposal learner.

| Policy | Joint-correct turns / 7,374 | Wrong unchanged turns | False asserted slot outputs | Abstained slot outputs |
|---|---:|---:|---:|---:|
| Verified-only carry | 5,665 | 1,319 | 2,344 | 0 |
| Unrestricted carry | 6,593 | 620 | 880 | 0 |
| Previously DEV-selected soft filter | 6,650 | 580 | 820 | 0 |
| Expiry after 2 HOLD turns | 5,982 | 1,219 | 767 | 1,136 |
| Expiry after 4 HOLD turns | 6,255 | 955 | 818 | 630 |
| Always initial empty | 5,429 | 1,490 | 2,916 | 0 |
| Reset each turn; HOLD means unknown | 28 | 6,914 | 23 | 14,346 |

The no-op majority effect is substantial, so aggregate accuracy alone is inadequate. All policies retained exactly the same underlying verified patch and state stream. Neither expiry changes the consequential write gate.

## Negative result
Expiry-2 reduced false assertions by 113 relative to unrestricted carry (paired dialogue-bootstrap 95% interval: 68–163 fewer), but produced **611 fewer completely correct turns** (510–716 fewer) and 1,136 abstentions. Expiry-4 reduced false assertions by 62 (33–95 fewer), but lost 338 correct turns (269–410 fewer) and introduced 630 abstentions. Compared to the soft filter, false-assertion reductions for both expiry policies have intervals spanning zero, while accuracy losses are clear on this exposed cohort.

The worst observed incorrect run and false-assertion run remained **14 turns** for unrestricted, soft, and both expiry policies. Reducing assertions by withholding answers is not evidence that persistent mistakes were repaired.

Across 511 annotated-change slot episodes, total incorrect-turn debt until the next annotated change or dialogue end was 768 for unrestricted, 632 for soft, 1,791 for expiry-2, and 1,336 for expiry-4. Never-correct episodes were respectively 148, 123, 154, and 151. Censored episodes are not credited as recovered. There were 447 initial sets, 57 overwrites, and only seven annotated removals. This is insufficient evidence about explicit cancellation or robust repeated updating.

## Language proposal bottleneck
The frozen top proposal covered 344/511 annotated slot changes (67.3%). Correct labels occurred in the top two on 466/511 (91.2%) and top three on 484/511 (94.7%). Those are **oracle coverage diagnostics**, not performance of a learned selector. They motivate a separately admitted contextual reranking comparison. This run does not demonstrate synonym, implicit-role, or new-language generalization.

## Reproducibility and integrity
Source/protocol freeze: local commit ec000c5, before candidate outcomes. Scientific worker and protocol hashes are in PREFLIGHT_SHA256SUMS. Pinned ecosystem: Veritas eec439492407b0aa26922c23b5ae40d95289f4eb.

Forge froze a validation-only diagnostic view and checked its schema and content hash; this is not a leakage or source-authenticity certificate. The worker consumed that view and independently checked it against pinned official DEV bytes. Model inference received text only; annotations were used by scoring.

TrainLab primary and replay completed successfully in 15.68 and 15.98 seconds. All three scored artifacts were byte-identical. The first replay registration attempted to reuse an immutable Research Lab evidence-node ID and was rejected after execution; the replay was then registered as a separate node, preserving the original. This bookkeeping failure did not alter predictions or scores. A separate reviewer recomputed all seven arms, patch/state equivalence, error debt and censoring.

There were 7,000 midpoint snapshot/restoration checks, 7,000 rejected corrupted snapshots and 7,000 rejected duplicate-turn attempts. Exactly 20,510 post-restore outputs and complete runtime states matched uninterrupted execution. Three short dialogues per arm had no post-midpoint suffix; 997/1,000 had actual continuation comparisons. These are same-run trusted-snapshot checks, not authentication of arbitrary SQLite dumps, hostile-input safety, process-kill fault testing or production crash-durability certification.

## Cost and next gate
Full sequential request timing is recorded separately when complete: text feature construction, frozen-head inference, SQLite verified commit and selected provisional read, three interleaved repetitions per arm. Noop/reset intentionally execute the same verified path, so these are policy-instrumentation costs, not their cheapest standalone implementations. Snapshot/restore microbenchmarks are separate.

Inference receives the user's 12× priority, but no artificial quality–milliseconds exchange rate is imposed. Expiry already fails the useful-answer comparison; faster execution would not establish a useful recovery repair. The next genuine 062 experiment needs new TRAIN-only learned contextual proposals/reranking, equal information and training budgets, fresh grouped evaluation and preserved language/cost controls. Do not tune it on this cohort and relabel that as independent evidence.

### Model scope
This is the existing COG dialogue stack, not a new native CSM fit. The direct models are two frozen multinomial logistic-regression heads over 95,361 current/context TF-IDF features: 762,896 area coefficients/intercepts and 572,172 price coefficients/intercepts, 1,335,068 total. Original solver iteration counts are 34 and 43. No capacity sweep, new optimizer update, Transformer comparison or model-size-matched architectural conclusion occurred here.

### Scope of the 064 failure mode
The common high-confidence path made 84 DEV writes (61 area, 23 price), all matching annotated slot changes, only 84/511 annotated changes admitted. Thus this cohort exercises missed-update/read persistence but contains **no observed incorrect admitted write** to establish recovery from bad verified commits. Expiry cannot alter verified commits by construction. Its negative result must not be generalized to every persistent-error repair or presented as solving both halves of 064.

The top-two coverage includes HOLD among the competing classes and uses the correct annotation to select the winning candidate. A learned reranker would also have to preserve the many unchanged turns; this ceiling alone supplies no precision guarantee or fair improvement estimate.

## Source anchors
- [Merged COG milestone, PR #72](https://github.com/FPC-effortless/cdl-attention-experiment/pull/72)
- [Provisional belief follow-up, PR #73](https://github.com/FPC-effortless/cdl-attention-experiment/pull/73), frozen at `1f0c2a18ad0d49bd5e27bb079aec48e5eb0a55dd` for upstream belief code.
- [Pinned authentic DEV source](https://github.com/budzianowski/multiwoz/tree/fe0c8e65cfcd8462bd33c86e35f21addc84ca82b/data/MultiWOZ_2.2/dev)
- [Pinned Veritas ecosystem](https://github.com/FPC-effortless/veritas/tree/eec439492407b0aa26922c23b5ae40d95289f4eb)

## Executed cost result

All 21 complete sequential sessions finished (three per policy, each 7,374 turns), 446.25 seconds including bookkeeping and recovery microbenchmarks. Median full-session CPU time: soft 20.16s, unrestricted 22.62s, expiry-2 19.93s, expiry-4 20.39s, verified 22.45s. The corresponding measured averages are 2.73, 3.07, 2.70, 2.76 and 3.04ms per turn. These descriptive clocks include the same frozen text scorer and verified SQLite path, not model loading. Execution order was rotated, hardware was not exclusive, and unrestricted varied 20.45–25.82s. Do not infer a statistically established latency advantage from three repetitions. Neither expiry policy earns its correctness loss. Detailed measurements and separate snapshot/restore clocks are in timing-summary.json.
