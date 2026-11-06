# LRR-008: nuisance-invariant causal relational addressing

Frozen protocol, 6 November 2026, before any LRR-008 scientific fit or TEST evaluation. This is a new controlled training experiment, not an amendment to LRR-007. Parent: 09becc095010e14002576a3582e8577aa54eb690. The original lrr005_register.py is preserved byte-for-byte (Git blob fb397a428ff483ce89ec14128fee9437487115a5).

## Question
Can explicit state-consistency training preserve hard entity/relation addresses under unseen irrelevant token representations, beyond what matched nuisance augmentation alone achieves?

The LRR-007 filler-only diagnostic motivates the question; its perfect familiar-grammar results are historical context, not a matched baseline for this experiment. Distributional invariance training has precedent (R-Drop, arXiv:2106.14448; VICReg, arXiv:2105.04906); no claim that consistency regularization itself is novel.

## Frozen arms and architecture
All three arms use the same 4,748-parameter CausalRoleRegister, initial weights and raw training examples per seed. Hard inference is the unchanged read-before-write role-state/dictionary routine. No learned attention matrix, query-to-all-history scoring or weighted token aggregation at inference. TRAIN retains dense answer-position compatibility scoring.

A — clean duplicated views: two identical copies of each source question; LRR-007 two-sided same-answer/different-answer objective.
B — nuisance augmentation: the same first view, plus a second view where only known training filler-token embeddings are replaced with independent Gaussian vectors (N(0,I) multiplied by independently sampled Uniform[0.5,1.5] scales). The token IDs, gold positions, values, entity/predicate surfaces and sequence lengths stay unchanged. Both views have the same answer supervision and two-sided relational objective. No consistency penalty.
C — augmentation + invariance: identical inputs/noise/initialization to B, plus 0.5 times mean valid-position squared discrepancy between paired pre-update entity distributions and relation distributions, plus mean squared gate discrepancy. All states are compared at aligned token positions; no oracle role label is an input to the model or inference routine.

IMPORTANT PRIVILEGE: the synthetic generator identifies which filler positions are safe to perturb. This is supervised invariance construction. It is NOT evidence that the network discovers which arbitrary natural-language words are irrelevant. The supplied mask is used ONLY to construct training embeddings and is absent at inference. The one-hot entity copy channel still uses the unchanged token IDs. No gate or role target is supplied.

## Counts and partitions
Fresh initialization seeds 61,62,63,64,65. Exactly 500 AdamW updates, lr=0.004, grad clip=2, one CPU thread, 48 question-view exposures/update: 8 source histories x 3 relation questions x 2 views. Thus each arm gets 24,000 question-view exposures and 4,000 source histories per seed. Equal exposure and parameter counts are NOT equal measured compute; report wall times.

TRAIN triples: anchor and positive use different aliases but the same latest answer; negative uses a different relation and different latest answer for the same literal entity. If duplicate entity/relation facts occur, the selected gold must be the latest record for that key, consistently with runtime replacement. Validate all three questions. Prior held-out alias pairs (0,3),(2,0),(1,2),(3,0) plus new TEST (1,3) and DEV (2,1) are excluded from every training query. All predicate lexemes are otherwise known. The new held-out pairs are fresh for this experiment, not globally unseen in all prior research.

Only nonconfirmatory seed 0 may be used for 2–20-update mechanical smoke checks; no hyperparameter search or capability tuning. TEST is opened only after code/protocol/seed configuration are hashed and frozen. Test histories have separate namespaces from TRAIN. Evaluation fixtures are identical across arms.

## Fixed evaluation
H=7,32,128 facts, 64 fresh histories per seed/H. Under fresh TEST alias pair (1,3), conditions:
1. regular familiar fillers;
2. same-position permutation of familiar fillers;
3. same-position fillers quickly/often/truly, in original vocabulary but unused in all new training data;
4. same-position new surface fillers novelx/novely/novelz, mapped to the shared UNK token;
5. additional known-filler positions (syntax shift), evaluated separately.

PRIMARY endpoint: hard recall@1 on condition 3 at H=32. Compare C minus B, not C against A alone. Secondary: condition 4, regular quality, per-seed failures, paired-answer agreement, query/gold key collisions and non-value writes. Known-but-absent and unknown-predicate false candidates are descriptive only; the router is not taught semantic abstention and no abstention gate is imposed.

Use a grammar-aware symbolic parser/union-find trained on the same answer-labeled triples; report its supplied grammar privilege. A capable Transformer/SSM language-model comparison is still absent: this run cannot establish attention replacement or a new model family.

## Precommitted decision rules
G0 integrity: source runtime hash matches; all protected alias pairs excluded; same source/target/noise streams between B/C; no perturbation at gold/entity/relation/punctuation positions; every new queried label is latest-valid; same parameter counts; default no-noise forward and source hard route agree; no future influence. Persist failed seeds.
G1 incremental invariance: C mean primary recall exceeds B by >=10 percentage points, with nonnegative difference in >=4/5 seeds and no regression >5pp.
G2 useful capability: C primary mean >=80%, >=70% in 4/5 seeds; regular mean >=90% and no >5pp mean loss versus A.
G3 broader transfer: report novel UNK filler results without a promotion threshold; ambiguity from shared UNK representation must be stated. Do not claim general lexical invariance from only an in-vocabulary success.

Always retain all three arms even if G1/G2 fail. Results are CPU implementation measurements in the active working container, not GitHub Actions or external replication unless separately executed there. Save source/config hashes, runtime versions, actual elapsed training, per-case predictions, final model AND optimizer state, score-recomputation audit, source bundle and all 15 checkpoints. No paid compute, merges, deployments or claimed independent agents.
