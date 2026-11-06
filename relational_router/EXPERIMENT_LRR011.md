# LRR-011 — absolute read/write relation agreement versus self-confidence

Woyengi Intelligence Labs — 2026-11-06. PROSPECTIVE protocol, before any new fit. Parent: LRR-010 PR #46, exact head 7056df49f6cfa4a3c463bbf63438a5c7bd7004cc. All parent source and evidence remain unchanged. Local CPU execution only; no merge, paid compute, deployment or claim of externally independent review.

## Question
LRR-010's marginal usage entropy increased soft diversity but did not reduce semantic hard-code collapse. The inherited loss explicitly aligns two questions with the same answer and repels different-answer questions; the question-to-correct-write relationship enters through normalized answer-position likelihood. Normalized ranking can reward relative discrimination without requiring confident absolute read/write code equality. Test whether explicit alignment to the correct stored record improves hard binding, rather than merely sharpening whatever code is currently largest.

This is supervised answer-index learning: the label already used by the base loss selects the gold record's pre-write state. No relation-family IDs, slot labels, entity span labels, oracle value-only writer or fresh runtime annotations enter the network. The conjecture is not a new general contrastive-learning primitive.

## Model and single-variable arms
Unchanged inherited InvariantRegister/CausalRoleRegister, 4,748 parameters, eight relation slots. Runtime is the original hard role gates, hard categorical relation code and last-write dictionary lookup. Training still uses soft dense cross-position compatibility. No attention-free training, unrestricted-language understanding, or end-to-end efficiency claim.

Let r_q be the relation register just before the final query marker and r_g the register just before the labeled correct answer token. Both are categorical distributions over the same eight codes. L_base is the original two-sided relational loss and gate regularization, WITHOUT marginal usage entropy.

A — control: L_A = L_base.
B — self-confidence diagnostic control: L_B = L_base + 0.2 * mean[-0.5(log max_c r_q,c + log max_c r_g,c)]. The maximizers are the model's own choices, not semantic labels.
C — read/write binding, PRIMARY: L_C = L_base + 0.2 * mean[-log(sum_c r_q,c * r_g,c)]. This uses the existing answer-token label only to identify the correct pre-write state.

All probabilities in auxiliary logs are clamped below at 1e-9. Coefficient 0.2 is fixed before all fits, chosen to match the existing relational-credit coefficient; it is NOT selected from this study's TEST or a pilot. No added entropy term, code resets, EMA, temperature annealing, straight-through estimator, adaptive stopping or model-size change. B is secondary; a favorable B result cannot silently replace the primary C hypothesis.

For distributions p,q, p dot q <=1, with equality only for the same one-hot code. Thus C favors both confidence AND agreement; B permits confident but different codes. This algebra is not a generalization guarantee. Existing negative relation separation remains active to discourage all records collapsing together.

## Matched execution and partitions
20 fresh paired tuples: model seeds 12001–12020, data-bank seeds 22001–22020. Each arm within a tuple receives identical initial weights, all input examples, labels, filler masks/noise, and event order. All arms use interleaved clean/augmented events, 500-bank batches consumed once each clean and augmented: 1,000 AdamW updates, lr .004, clipping 2, 500 clean + 500 augmented batches, 48,000 labeled question-view exposures/model and 4,000 unique source histories/bank. Total 60 confirmation fits; NO pilot or tuning. Arm execution order is rotated by tuple to reduce ordering confounds.

Use unchanged LRR-008/009/010 bank generator and held-out alias restrictions. TEST ordered pair (1,3), DEV (2,1), all historical forbidden pairs excluded from every training triplet. This is replication on the same finite relation task with fresh histories/initializations, not unseen semantics. Generator-marked nuisance filler embedding substitution remains privileged training information. No mask enters hard inference. Literal entity identity, grammar, fixed code capacity, and shared UNK embedding remain explicit limitations.

DEV uses fresh model-seed-keyed panels: 128 regular histories, 128 same-position unfamiliar-in-vocabulary histories, and 128 independent training-pair diagnostic histories balanced 4 families x4 aliases x8. Record at updates 0/100/250/500/750/1000, never stop or alter training based on results. Save model and AdamW states at 0/250/500/750/1000. All 60 fits must finish before the first learned-model TEST evaluation. Nonfinite training aborts are preserved and reported, not replaced.

TEST: 64 fresh histories per tuple at H=7,32,128 and five conditions: regular, familiar filler permutation, unfamiliar in-vocabulary substitution (PRIMARY at H32), out-of-vocabulary filler substitution, added familiar filler positions. Arms/paired transformations share test histories. Seed formula 510000000000 + model_seed*1000000 + H*1000 + case_index is disjoint from all source TRAIN/DEV formulas. Negative H32 queries: familiar entity/predicate with no supporting record, and a single new predicate surface. Validate no support remains; report raw false reads even though no null/read gate is trained. No gate may conceal bad raw retrieval.

## Endpoints and diagnostic definitions
Primary estimand: ALL-20-seed mean paired C−A exact-token retrieval on H32 unfamiliar in-vocabulary fillers. Treat model/data-bank tuples as independent units; do not inflate n with cases or views.

Retain LRR-010 semantic-collapse definition: >=10% of all different-true-family query pairs share the same valid hard code on balanced DEV. Missing codes separate. Report severe starvation (effective codes<2 or largest share>=.90), code occupancy, marginal and conditional entropy, semantic collisions, missing query codes, query/gold address agreement, gold writer admission, non-value writes, wrong-address/overwrite/no-candidate failures. Family labels are evaluator-only. Qualification: >=90% regular DEV accuracy. Conditional noncollapsed/qualified performance is descriptive and post-treatment selected, not a substitute for the unconditional paired analysis.

Add read/write diagnostics: soft r_q dot r_g; entropy/max probability of q and g; hard relation equality separately from entity equality; disagreement between soft argmax position reading and unchanged hard indexed reading. No claim a soft diagnostic is an attention-free runtime. Log positive and negative auxiliary magnitudes plus all model/optimizer/source/data hashes.

## Statistics and prospective promotion
Primary C−A: 50,000 paired tuple bootstrap resamples, exact two-sided sign-flip enumeration over 2^20 assignments (exchangeability/symmetry assumption disclosed), per-pair differences and wins/ties/losses. Collapse difference: paired discordant counts and exact McNemar test; Wilson intervals for individual arm collapse rates. No optional stopping. Secondary B−A and C−B tests use Holm correction and are labeled secondary; all outcomes are preserved.

Promote C only if ALL: (1) primary mean gain >=5 percentage points and paired bootstrap lower bound>0; (2) at least three fewer semantically collapsed runs than A; (3) no more than 5pp aggregate deterioration in known-absent false-read rate or H32 OOV retrieval. Report unknown-predicate false reads separately; no blanket abstention guarantee. If A has fewer than three collapses, the collapse-reduction promotion condition is unsatisfied rather than relaxed after results. Any broader architecture claim still requires an appropriately qualified attention/recurrent language-model comparator and measured end-to-end resource advantage, absent here.

## Evidence and audit
Freeze local source/hash manifest before first fit; commit this protocol before fitting. Validate zero-weight value and ALL-parameter gradient equivalence to the inherited objective; analytical agreement/confidence fixtures, label/mask/causality integrity, fixed parameter budget and original hard runtime parity. Preserve every fit, milestone, prediction and false read. Same-investigator separate audit recomputes aggregates from raw records, source hashes, exact paired bank identity, optimizer counters, finite tensors, and original sequential-runtime replays. Do not call this external review or independent departmental execution. Draft PR and downloadable evidence only; no paid GPU or automatic merge.
