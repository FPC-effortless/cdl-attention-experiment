# LRR-009 — Equal-exposure curriculum order and relational retention

Woyengi Intelligence Labs · 6 November 2026

**Prospective protocol.** No LRR-009 scientific fit or trained-model TEST has run when this document is committed. Parent is draft PR #44, source head `0c6d9ecd1a5a145a9d1b6a47b428313d29f44f8d`. LRR-008's negative consistency result stays unchanged. This successor tests the proposed clean-first staging hypothesis, not another consistency penalty.

## Question and exact intervention

Does learning relational separation before exposure to perturbed filler embeddings improve hard indexed retrieval, compared with interleaving exactly the same clean and perturbed examples?

Use the unchanged 4,748-parameter LRR-005 hard runtime and LRR-008's training-only augmentation wrapper and two-sided answer-position objective. No added parameters, loss terms, role supervision, token vocabulary or inference masks. Gaussian filler embedding substitutions retain the existing generator-identified nuisance-position privilege. This is not autonomous discovery of irrelevant text.

For each fresh initialization seed **71,72,73,74,75**, generate a fixed bank of **500 minibatches**, each containing eight histories, three answer-supervised questions/history and two views/question (48 question views). A clean event uses duplicate clean views; an augmented event substitutes the second view's filler embeddings with the fixed recorded noise. Each arm consumes every bank minibatch **once clean and once augmented**, for **1,000 AdamW updates**, 48,000 question-view exposures, 4,000 unique histories, 500 clean events and 500 augmented events. The repeated bank is intentional: it gives an identical complete multiset of inputs and noise while isolating their order. Repeated exposures must not be called independent histories.

Three schedules, all with lr=.004, clip norm=2 and optimizer state carried continuously (no reset or unreported extension):

- **Interleaved (control):** clean(0), augmented(0), clean(1), augmented(1), ... clean(499), augmented(499).
- **Clean-first (primary):** clean(0..499), then augmented(0..499).
- **Augmentation-first (reverse control):** augmented(0..499), then clean(0..499).

The intervention changes temporal curriculum and optimizer history jointly; this study cannot separate those mechanisms. No result-driven selection or replacement of seeds, schedules, checkpoints or augmentation strengths.

## Data / partitions

Reuse the audited LRR-008 generator and latest-valid-record labels. TRAIN forbids ordered alias pairs {(0,3),(2,0),(1,2),(3,0),(1,3),(2,1)} in every question; DEV=(2,1); TEST=(1,3). New examples use a disjoint source namespace: call the parent `augmented_batch` using data seed `2000 + model_seed` and bank position. All three arms see the same bank and exact perturbation tensors. No changes to the grammar, four known hidden relation families, eight supplied code slots or literal entity copying.

DEV: 128 independent H=32 histories/model seed, namespace `130_000_000_000 + seed*1_000_000 + i`, regular and same-position untrained-in-vocabulary filler substitutions. Evaluate at updates 0/250/500/750/1000. No learning-rate, schedule, early stop or checkpoint changes based on DEV. Preserve all DEV results and the model/optimizer snapshots at 250/500/750/1000.

TEST: only after all 15 fits finish; 64 new histories per seed/H, namespace `150_000_000_000 + seed*1_000_000 + H*1000 + i`; H={7,32,128}; regular, familiar-filler permutation, untrained-in-vocabulary fillers (PRIMARY), new OOV filler surfaces, and additional familiar filler positions. These are fresh history draws from the already studied finite grammar, not a new external benchmark or unseen semantic family. No threshold or model selection on TEST.

Primary endpoint: H=32 untrained-in-vocabulary filler exact answer-token recall@1. Regular recall, long-history recall, code occupancy, overwrite/collision diagnostics and OOV transfer remain separately reported. A grammar-aware symbolic parser/union-find receives the same answer-supervised source histories as a privileged-grammar control. No capability-qualified attention/SSM language-model comparator is being trained here; attention replacement remains untested.

## Warm-up qualification and retention

At clean-first update 500, label each model **qualified** only if regular DEV recall >=90%. This is an observed prerequisite, not a selection rule: all five seeds continue the fixed schedule regardless of qualification, and every seed enters primary analysis. If few warm-ups qualify, report that the proposed qualified-start condition was not robustly achieved rather than declaring qualified-start adaptation disproven.

For each clean-first model, compare predictions on the identical regular DEV histories before augmentation (update 500) and after (update 1000). Report initially-correct count, lost-correct count, retained-correct fraction and new gains. An initially-correct denominator of zero is undefined, not 100% retained. Also report hard query relation-code occupancy and collisions; invariance alone is not correctness.

## Prospectively fixed gates

- G0 integrity: unchanged source hashes through fits; 4,748 parameters/model; equal initial states and exact input-event multiset per paired seed; 1,000 optimizer step counters; no TEST opened until all fits finish; causal original hard-runtime parity; all protected alias pair checks and latest-record labels valid.
- G1 capability: clean-first primary mean >=80%, >=70% in at least 4/5 seeds, and regular H32 TEST mean >=90%.
- G2 attributable schedule gain: clean-first minus interleaved primary mean >=10 percentage points and no seed regresses >5 points. Reverse order is a prespecified control, not an alternative primary winner.
- G3 qualified retention: >=4/5 clean-first warm-ups qualify; among qualified models, no more than 5% of initially correct regular DEV answers are lost after augmentation (pooled counts, with per-seed values visible). Insufficient qualified seeds means NOT ESTABLISHED, not silently omitted failures.

Report all gates separately and include per-seed uncertainty; five model seeds do not establish population-wide superiority. Historical LRR-008 scores are not matched comparators because total exposures differ.

## Execution and reporting

Bounded local CPU execution, one PyTorch thread, deterministic algorithms where supported. No paid GPU, CI training claim, merge or deployment. Freeze code, config and hashes before the first fit; preserve every final/intermediate model AND optimizer state, input/noise receipts, full raw predictions, fit traces, failures and separate audit. The same investigator can run a second audit program but must not call it an independent external review.

No abstention gate: known-absent/unknown predicates may be logged diagnostically, but selective answering is not the core trained capability. Preserve the user's distinction between retrieval correctness and explicitly taught rejection.

Related methodological context: Bengio et al. (ICML 2009), Curriculum Learning, DOI 10.1145/1553374.1553380. This is an order-effect hypothesis in the user's hard-address model, not a claim to invent curriculum learning.
