# LRR-006 — Decouple address retrieval from explicitly learned selective answering

**Preregistered November 6, 2026.** Scientific successor to frozen LRR-005, not an amendment to its original negative/no-answer gates.

## Question and methodological correction

LRR-005 models were trained on positive answer-position supervision only, despite exposing an unused binary null slot in the soft position loss. Demanding low false-answer rates on never-trained *unanswerable* examples is not a valid test of **trained abstention**. Those prior no-answer results remain useful **descriptive failure diagnostics**; the LRR-005 reported verdict and predeclared gates are not changed retroactively.

This experiment **freezes all 5 actual LRR-005 `register_local_credit` checkpoints**, trains a separate bounded energy head on examples explicitly labeled answerable/correct or invalid/unsupported, and tests two questions independently:

1. Has the frozen router learned correct **semantic address selection**? This is measured by raw gold-position recall with **no abstention adjustment**.
2. Given a single hard-indexed candidate and local structural evidence, can an explicitly trained energy head **selectively approve or abstain** at useful true-answer coverage and acceptably low false-answer rate? The energy head cannot repair a wrong token position and cannot replace the indexed retrieval function.

Only the second question uses the new selection criterion. Explicit training **does not mathematically guarantee** correct rejection of arbitrary unseen queries. Any conformal or finite-sample risk claim would require its own assumptions, calibration and uncertainty guarantees, and is not asserted here.

## Frozen data and architecture

- Load *exact* five LRR-005 saved checkpoints seeds **41–45**, artifact `13466926475`, ZIP SHA256 `801bf0f95e1379d1b783893c11b0d4032984997e7de26c153ac65f952e36d87f`, source scientific head `1d114fbc28c6b5b37084f3afbbd32a08676f7eb4`.
- Router is frozen: 4,748 params, causal single-convolution role-state address constructor, one hard indexed candidate per query. No full-history QK scoring or token-pair attention in **runtime retrieval**.
- New energy head: small 2-layer MLP of fixed structural/candidate-local features, output a scalar `s=-E_valid+E_null` where `E_null=0`. Train using **explicit** positives (correct candidate) and negatives (wrong candidate; known-predicate absent relation; unfamiliar predicate), with `softplus(m-y*s)`, `m=0.5` and `y=±1`. No invented absolute probability or universal abstention guarantee.
- Features may use the already-constructed **one** retrieved record's cached pre-write registers and its write gate, plus the final query state, local confidence/entropy, write-address collision count and candidate age. Do **not** compare query against every history token at inference. A full-context diagnostic implementation may precompute states once; do not characterize that as O(1) total work.
- The energy model is a **separate trained reliability head**. The original hard router, tokenizer, relation-count prior, non-causal answer-label source and model-quality limitations are unchanged.
- Control A: frozen index always returns available candidate (no trained abstention).
- Control B: single scalar structural-similarity threshold, chosen by the **same DEV-only** constrained calibration as the energy head.
- Control C: learned energy head with threshold 0; and DEV-calibrated energy threshold. Always-reject control establishes the trivial zero-FPR/zero-coverage floor.
- A proposed *inference-only* history-versus-question index separation (no question-token writes) and evaluator-only oracle value-only writes are **diagnostics**, not new learned capability or promotion controls. Initial frozen replay found they remove extra writes but do not change any of the five seeds' positive retrieval recall or known-absent false-answer rate.

## Partitions and counts

- Same known four relation families, 16 known predicate surface lexemes, six familiar filler tokens and ten literal entity names as the frozen original. All known lexemes may be observed during energy-head training; open-domain relation induction is NOT tested.
- Energy TRAIN: **256 independent positive histories/seed** on original train-restricted ordered predicate pairs, each with (a) one positive query and (b) two distinct negative queries: a known-predicate/known-entity **no-support** intervention that preserves hard distractors and an unfamiliar query predicate. The energy loss labels returned candidates correct/incorrect using the *training* oracle gold position only; no target labels at inference. This is 768 frozen-router scoring instances per seed.
- Energy/calibration DEV: **128 independent histories/seed** with frozen `(alias[3]→alias[0])` pairing; identical positive / known-absent / unknown-query groups. DEV selects score threshold maximizing **correct positive coverage**, subject to **empirical** known-absent false-answer rate ≤10%. Threshold selection rule is fixed and applied to both the energy head and scalar control; no TEST threshold tuning.
- Confirmatory TEST: **64 independent histories/seed/length/condition** with reserved `(alias[1]→alias[2])`; H∈{7,32,128}; conditions regular, known-filler reordered, and unseen filler surface words. For each condition, evaluate its supported query plus separate known-absent and unfamiliar-predicate negatives. Test seeds disjoint from TRAIN/DEV, and actual split verification runs before energy fits.
- Explicit energy training: 600 AdamW updates per seed with full precomputed frozen-router features, `lr=0.003`, batch up to 128, no class weighting beyond balanced 1:1 correct/incorrect sample construction. If no correct candidates exist in a frozen seed's 256 positive training histories, label the energy head untrainable and preserve failure, never cherry-pick or add source labels.
- Preserve every seed's raw labels, candidate-selection correctness, scores, thresholds, coverage and false-answer counts. No protected data, new original-router training or GPU allocation.

## Valid endpoints and gates (not the old abstention gate)

**Primary unchanged router endpoint:** `raw_exact_gold_recall@1` at H=32; identical for all selection policies. A changed value indicates instrumentation error.

**Energy-head selectable value:** on H=32 known-syntax TEST, report (i) accepted *correct* answers per total answerable queries, (ii) accepted wrong answers per total accepted, (iii) false answers per total unanswerable queries **separately for known-absent and unknown words**, (iv) accepted query coverage and abstentions, (v) empirical risk–coverage curve and DEV-calibrated threshold, and (vi) comparisons to the scalar DEV-calibrated control and always-reject.

Precommitted hypothesis: energy head has at least **10pp higher correct-answer coverage than the scalar threshold** at no more than **10% known-absent false-answer rate**, in **≥4/5 independently frozen seed checkpoints**. Failures and no-valid-coverage seeds count as FAIL; this hypothesis is distinct from the router's original recall gate.

**OOD:** independently report regular versus known-filler shift versus entirely unseen filler words; no transfer guarantee unless positive accepted-correct coverage and negative rates remain competitive under all three.

**Causal cost:** inspect number of candidate records used per question. Any energy feature requiring exhaustive query-to-history comparisons disqualifies the attention-free runtime boundary. Cached state reuse requires a verified prefill/update and query protocol; a one-off full-encoding helper does not establish speedup.

## Diagnostics, interpretation limits, next research step

A post-result independent frozen-checkpoint first-stage diagnostic (before trained energy-head execution) verified LRR-005 exact positive recall at H=32 as `[1,1,.078125,1,.71875]`. Suppressing writes from query tokens or privileging value-only writes **did not change retrieval accuracy or known-absent false-answer rate**; both were binding/identity problems. Seed 43 retained H=32 7.8125% accuracy and 34.375% known-absent FPR with oracle value-only writes. Seed 42 had 33 unnecessary non-value writes per history. These observations motivate separating **address consistency** from **answer selection**.

If explicit energy training improves risk/coverage, it validates only a *bounded learned selective-answer readout conditional on frozen LRR outputs*; it does not demonstrate learned semantic relations, identity generalization, non-attention training, memory lifecycle, an LLM attention replacement, or robustness to distribution shifts. A semantic-address successor must still resolve the seed43 collision and absent-fact failure without supplied relation classes or oracle writer hints.


## Pre-confirmation amendment: qualifying energy-training pairs (2026-11-06)

**Stage:** addendum made after TRAIN-only preflight of six generated histories, **before** any LRR-006 confirmatory TEST training/evaluation. Frozen prior LRR-005 test history was already known; no new LRR-006 TEST samples or thresholds were inspected. Original text is preserved.

TRAIN-only preflight found that correctly configured hard indices often return **no candidate** for known-absent and unknown-predicate questions, especially for high-recall seed 41. Such no-candidate rows abstain structurally and cannot provide any gradient for a one-candidate energy landscape. A binary energy learner trained only on natural hard-index outcomes would have positives without meaningful invalid *candidates* in some seeds.

**Additional explicit contrastive training pairs (training only):** for each training history, use its provided answer-index label to produce (a) the correct stored value token as a **privileged positive candidate**, and (b) another stored value token for the same entity but a **different** relation as a **counterfactual negative candidate**. These two positions are injected into the energy-head training scorer only; they do not modify the hard index, semantic code, or actual inference path. The candidate energy scorer gets the same structural features for both, with no direct correctness indicator. The resulting training dataset includes five kinds of rows per source history: naturally indexed positive query, known-absent query, unknown-predicate query, privileged positive candidate, and mismatched counterfactual candidate. For 256 source histories, this is 1,280 attempted scoring rows/seed; rows with no candidate are structural abstentions, not head-training examples.

**Important supervision classification:** oracle candidate positions and wrong-relation identities are supplied **only for constructing the energy supervision**. This makes the head a *supervised teacher-contrastive verifier*, not an end-to-end naturally learned energy landscape or a semantic parser. A gain therefore measures **selective reliability given one hard-address candidate**, not improved relation discovery and not comparison against a zero-privilege symbolic learner.

The remaining confirmation splits, seed set, fixed 600 head updates, DEV threshold selection, TEST metrics, comparator controls and no-superiority scientific boundary are unchanged. The amended protocol still admits negative results freely.
