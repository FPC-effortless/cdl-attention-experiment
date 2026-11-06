# LRR-005 — Causal role-register and paired counterfactual credit (2026-11-06)

**Status:** Protocol frozen before new confirmatory TEST. Local development-only seeds 0,1,2 explored on a different ordered synonym pair; their outputs are NOT confirmation. No attention replacement or neural superiority claim is authorized.

## Hypothesis and distinction from LRR-004

LRR-004 collapses to the immediate predecessor relation offset and fails adverb perturbations. Replace fixed relative-token offsets with **two causal registers**: an entity surface-token register and a relation-code register. At position t, an independent token-local classifier predicts the entity-copy gate, relation-update gate, and write gate. Value-token writes use **S_(t-1)** *before* any update with x_t; a query at '?' likewise reads the register **before** the question-mark token. Runtime writes the value's **token position** under (literal_entity_surface, hard_relation_code) and returns the newest matched posting without token-pair QK scoring or attention-weighted mixing.

This implementation is **inspired by** the PRS role-state work and PSN's write/read separation; it does not import those complete architectures. It evaluates a **CASM-X20U-inspired** local counterfactual outcome loss, not the full X20U graph-conditioned local estimator.

## Training arms and strict comparison

- A: role-register baseline, identical dataset and initialization, ordinary answer-position contrastive loss.
- B: the same role-register and training documents/updates, plus `0.2 × -log(1 - cosine_categorical(q_rel_A, q_rel_B))` for two distinct relation queries against the **same history and entity**. Distinctness is defined by different independently labeled target *answer positions*, never canonical relation-family labels supplied to the model. Each question additionally receives identical answer-position supervision.
- Matched model: 32-wide token embedding, one causal convolution and layer norm, 3 independent sigmoid role/update/write gates, **eight** learned relation codes and one scalar null score; count all parameters (expected ~4,748).
- Both arms: 500 AdamW updates, batch 32 questions, learning rate 0.004, grad clip 2.0, one CPU thread. Compare seeds **41,42,43,44,45**, no replacement based on outcome; 16,000 labeled question exposures per seed/arm.
- Training uses provided answer **token position** derived from the answer string (not scalar reward only). It supplies no entity-span labels, role tags, canonical family IDs or full pairwise alias-equivalence matrix.
- **Mandatory anti-leakage:** exclude ordered predicate alias pairs (1→2) and (3→0) for each of four relation families in *all positive training questions, including the counterfactual second question*. Dev used (3→0) only; confirmatory TEST uses (1→2) only. Run a validator against every actually generated training example and fail before training if either reserved pair appears.

## Dataset and evaluation

Restricted natural token strings: clauses `entity [optional filler] predicate [optional filler] opaque_value .` and questions `what does entity [optional filler] predicate ?`; no fact tags or entity IDs, but the vocabulary, question-ending marker and relation code capacity are supplied priors. Exact surface entity strings are used as index keys, **not learned entity coreference**. Each document has same-entity/wrong-relation and wrong-entity/same-relation distractors, plus additional random facts.

Confirmatory TEST uses 64 new examples per arm/seed/length for lengths **7, 32, 128** facts, generated only after parameters are frozen. All arms evaluate identical TEST documents.

Separate endpoints:
1. Primary: **H=32, known filler words, reserved ordered alias pairing**, exact gold value token position recall@1, five per-seed scores, paired difference, and correctly abstained fraction.
2. Structural shift: same heldout aliases but **two consecutive fillers** at selected positions, using known trained filler words. Do not conflate this with unknown-word generalization.
3. Lexical OOD: fillers not used in training plus nontraining combinations. Show separate failure rates.
4. Known relation but absent matching fact for an existing entity (target fact subject changed; other same-entity facts remain), exact no-answer false-positive rate.
5. Unseen relation surface token, false-answer rate; do not quietly add a vocabulary whitelist.
6. Symbolic grammar-aware parser + union-find, trained on the **identical answer-labeled training stream**. It is a strong privileged-grammar reference, not a claim of learned raw parsing.

## Gates and limits

- Integrity: 0 heldout pairs appear in either training arm, no gold position/role labels enter route(), no future-token index, single hard address lookup, and *read-before-write* equivalence for chunked versus monolithic execution on a given model.
- Paired-credit capability: at least **4/5** B seeds achieve H=32 recall@1 >=0.80; B mean H=32 >=0.85, paired mean gain >=0.10, and no seed regresses >0.05.
- Structural transfer: B H=32 shifted-syntax recall >=0.70 mean. Fail if shifted syntax stays fragile even if normal syntax saturates.
- Governance: B known-absent false-answer rate <=0.10; B unknown-relation false-answer rate <=0.10 **without** a lexical guard. These are independent gates, not automatic requirements for publication of a negative result.
- Strong reference: record exact symbolic correctness and family-code usage, do not claim neural superiority unless B outperforms appropriate controls on truly unseen semantic structure; strong attention/SSM language modeling is outside this small diagnostic.

**Non-claims:** no coreference, unknown semantic family discovery, inferred code cardinality, open-domain language, end-to-end attention replacement, GPU or full-model speed. Expected hard lookup O(1) alone does not make ingestion, soft training loss or full text parsing constant-time.

## Development failure disclosure and preregistration rationale

Early same-document paired-credit experiments accidentally permitted the withheld ordered alias pairs in the second query; **those observations are contaminated** and excluded. After repairing all training pair paths, development seeds 0–2 showed that read-before-write prevents the 'value token overwrites the relation address' shortcut, but accuracy remained variable under unseen fillers. These observations motivated the fixed λ=0.2, read-before-write contract; no confirmatory seeds 41–45 or TEST pair (1→2) were inspected while selecting them.

**Decision:** Any successful controlled outcome is a bounded role-state/address-training result only, and requires independent code/data and checkpoint review before model integration.