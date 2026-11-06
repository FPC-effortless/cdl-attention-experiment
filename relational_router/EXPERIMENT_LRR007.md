# LRR-007 — Dual relational credit for causal hard addresses

**Frozen scientific protocol, 2026-11-06.** This experiment is a new learned *address-consistency* comparison, separate from LRR-006's optional energy-based answer-selection head. All models use the LRR-005 4,748-parameter role-state architecture with no Transformer self-attention at runtime. No frozen LRR-001–006 result is amended.

## Hypothesis

The failed LRR-005 seed demonstrates a gap between discriminating unrelated query roles and aligning **two distinct paraphrases pointing to the same answer**. With the *same trained model, optimizer, token histories and query exposure*, explicitly rewarding both equivalence and separation may reduce hard-address collisions and improve held-out-pair retrieval.

Training-time query relation distributions `r(q)` are soft. For a common-history triple `(q_a,q_+,q_-)`:
- `q_a` and `q_+` use different predicate words and must identify the **same gold value-token position**.
- `q_-` refers to a **different true relation and a different gold value position** for the same literal entity.
- Define `s(u,v) = sum_c r_c(u)r_c(v)` (categorical code agreement, not token-pair self-attention).
- Original outcome supervision: `L_ans` = the same causal answer-position loss as LRR-005, applied to all three questions.
- Neg-only control: `L_A = L_ans + 0.2 * [-log(1-s(q_a,q_-))]`.
- **Primary** dual-credit intervention: `L_B = L_A + 0.2 * [-log(s(q_a,q_+))]`.
- Secondary energy-margin treatment: `L_C = L_ans + 0.2 * softplus(0.25 + s(q_a,q_-)-s(q_a,q_+))`.

No canonical relation-category IDs, entity-span tags or value write masks are passed into `forward()` or `route()`. The **training data generator uses known answer positions** to construct same-answer and different-answer supervision; this is supervised relational learning, not spontaneous semantic discovery. Like LRR-005, `L_ans` computes dense training-only position similarities. **Hard inference** uses a single dictionary/posting address and no query-to-history pair scoring.

## Strict source / train–dev–test split

- Use the original LRR-005 unsegmented-but-templated text grammar (literal entity names, optional fillers, exact punctuation, opaque values, four known relation families and eight preallocated relation codes).
- All three training questions share exactly the same fact sequence and entity. A positive synonym `q_+` differs lexically from `q_a`. The negative has different target value and family. Before every update validate **each of the three ordered (fact predicate, question predicate) pairs**.
- Forbid both current reserved pairs `(0,3)` and `(2,0)`, **and** the prior LRR-005 held-out pairs `(1,2)` and `(3,0)`, during this experiment's TRAIN in every family. The excluded pairs' lexical words may still occur elsewhere in TRAIN; the challenge measures **combinatorial relation transfer**, not unseen word semantics.
- DEV uses held-out ordered pair `(2,0)`. Confirmatory TEST uses fresh `(0,3)`, never used for model selection in this experiment. The old excluded pairs may be reported only as secondary challenges.
- Fresh training seeds **51,52,53,54,55**, 500 AdamW updates × 48 labeled questions (16 common-history triples) per seed per arm, lr=0.004, grad clip 2, PyTorch one CPU thread. Each arm starts from the same seed-matched random initialization and consumes **the same triple stream**. No additional exposures in a treatment. Full training logs and exact SHA/checkpoint identities required.
- Confirmatory TEST: H={7,32,128} facts, 64 fresh documents per seed × H × condition. Conditions: `regular`, `shift` (more known filler words), and `novel_fillers` (different lexemes from those in training). TEST source seed formula must be disjoint from TRAIN and DEV, fixed before execution. No TEST-driven tuning or retries for a scientific claim.

## Comparators and diagnostics

1. A (neg-only) is the **matched experiment-internal baseline**. Historical LRR-005 results differ in query exposure and must **not** be used as a matched causal comparator.
2. B (dual-credit) is the primary evaluated intervention; C (margin-ranking) is a preregistered exploratory secondary treatment, **not eligible** for the primary claim even if better.
3. Zero-trainable-parameter grammar-aware symbolic parser + union-find trained on the **same answer-labeled triples**. Its privilege must be stated.
4. For each hard query, separately track correct token position, no matching index entry, wrong retrieved position, query/gold-address disagreement, overwrite on the correct address, value-write admission, non-value writes, distinct hard addresses, and hard-code occupancy.
5. Evaluate known-predicate no-support and unknown-word questions as **descriptive diagnostics only**, because this model has not been separately trained to abstain. LRR-006 is the trained selective-answer lane. No additional abstention promotion gate.
6. Sanity controls: reverse token/nonce names, duplicate fact writes and unseen syntactic variants require their own distinct assessment. The main structural shift and filler OOD conditions are included; no open-domain NLP is inferred.

## Precommitted decision gates

- **G0 integrity:** All 3 × every training triple alias pairs respect held-outs, positive questions have identical gold index but different predicate lexemes, negative has different gold index, no gold labels are supplied to runtime, all reads causal, baseline and intervention have identical seed/data/model counts, and whole report preserves failed seeds.
- **G1 primary capability:** At H=32 TEST regular B achieves ≥0.85 mean recall@1 and ≥0.80 in **4/5** seeds.
- **G2 paired attributable improvement:** B mean recall@1 exceeds A by ≥0.10 absolute and no individual seed regresses by more than 0.05. With only five model seeds the uncertainty and per-seed dependence must be reported; no population-level superiority claim from an isolated mean.
- **G3 syntax:** B mean H=32 known-filler shift recall@1 ≥0.75. Novel-filler OOD recall is reported without a pass threshold; it remains an adversarial falsifier.
- **G4 nontrivial semantics:** For a wider claim (not G1/G2), B must at least **match** the grammar-aware symbolic result on controlled tasks and a separately trained, capability-qualified attention reference on natural text. No such reference is qualified by this experiment, so general attention replacement remains UNPROVEN regardless of G1/G2.
- Training FLOPs, GPU gather kernels, parameter-matched Transformer language modeling and full end-to-end speed are **not** measured here. Query-time hash lookup O(1) expected does **not** imply full sequence O(1).

## Authority and handling

Run bounded CPU-only GitHub Actions. Preserve all negative results, all 15 checkpoints, exact evaluated commit, independent score audit and reporting. Draft PR only; **no merge, promotion, paid GPU, model deployment or external announcement** without separate independent review and the appropriate gates.
