# PNDS Research Results Ledger

Retrospective mapping of existing CDL Attention and CASM experiments into PNDS-URP v0.1. Original claims remain bounded by their source documents.

Evidence: E0 idea; E1 implemented; E2 smoke; E3 controlled; E4 reproduced; E5 cross-condition; E6 cross-domain.

## PNDS-CDL-ADDRESS-001 — Stage A conditional description length retrieval
- Status: SUPPORTED
- Source: main @ c31554413301e3c9d3e6b3f8c8c6be572a74a748
- Historical head: 2ee0fac8886b60c0fd9737d7ec09fedc1770ebe9
- Date: 2026-08-29
- Model: SmolLM2-135M. 120 paired cases per candidate-count condition; 6/12/24 candidates.
- Baseline: gzip conditional description length.
- Result: Top-1 at 6 candidates 0.8750 CDL vs 0.6333 gzip; 12: 0.8250 vs 0.5750; 24: 0.7583 vs 0.4833. CDL-selected answer NLL was lower at all three counts.
- Observed: conditional LM description length remained materially better than gzip as distractors increased.
- PNDS relation: strong evidence for Addressability/relevance routing R_t.
- Limitation: synthetic relational language; teacher scoring is expensive.
- Evidence: E5. Decision: SUPPORTED.
- Next: natural/paraphrased tests and cheap-router distillation.

## PNDS-CDL-ADDRESS-002 — Joint attention control
- Status: SUPPORTED
- Date: 2026-08-29
- Historical head: 6320e9f6c3a603e8cc8eacf75a2fc0da791fd5cc
- Controls: gzip and joint-context native attention mass; 120 cases.
- Result: CDL Top-1 0.8750, MRR 0.9361, selected-answer NLL 5.2170; all-layer attention Top-1 0.2667, MRR 0.5422, NLL 7.9693.
- Observed: raw attention mass was a weak explicit relevance score on this benchmark.
- PNDS relation: supports separating routing/relevance objectives from the attention computation mechanism.
- Limitation: does not establish CDL as an attention replacement.
- Evidence: E3. Decision: SUPPORTED.
- Next: explicit relevance routing before attention.

## PNDS-CDL-ADDRESS-003 — Stage B CDL distillation to Q/K router
- Status: INCONCLUSIVE
- Date: 2026-08-29
- Run: GitHub Actions 33231945184; 800 train / 240 held-out; 64-dim bilinear Q/K student.
- Baseline: direct-label Q/K student.
- Result: direct-label Top-1 77.92%, MRR 0.8854, answer NLL 5.5113; CDL-distilled 73.75%, MRR 0.8635, answer NLL 5.7686; teacher agreement 72.92% vs 67.08% for direct-label.
- Paired test p≈0.275.
- Observed: distillation transferred some teacher geometry but did not demonstrate a task benefit.
- PNDS relation: critical bottleneck: a strong relevance teacher does not automatically yield a cheap effective router.
- Evidence: E4. Decision: INCONCLUSIVE.
- Next: temperature sweep, CE+KL, rank/margin distillation, richer scorer, larger independent test.

## PNDS-CDL-ADDRESS-004 — Candidate-count robustness
- Status: SUPPORTED
- Parent: PNDS-CDL-ADDRESS-001
- Conditions: 6,12,24 candidates.
- Result: CDL Top-1 0.8750 -> 0.8250 -> 0.7583; gzip 0.6333 -> 0.5750 -> 0.4833.
- PNDS relation: supports relevance routing under increasing distractor load.
- Evidence: E5. Decision: SUPPORTED.
- Next: larger candidate sets plus matched compute measurement.

## PNDS-CDL-COST-001 — Runtime cost of relevance scoring
- Status: BLOCKED
- Parent: PNDS-CDL-ADDRESS-001
- Observed benchmark times: 128.44 s at 6 candidates, 114.22 s at 12, 194.82 s at 24. These are end-to-end runs with diagnostic overhead, not clean serving measurements.
- PNDS relation: exposes the systems bottleneck: routing quality must be achieved without a full LM pass over every candidate.
- Evidence: E3. Decision: BLOCKED.
- Next: isolate router inference cost and compare full-context, teacher and distilled Q/K routing on matched hardware.

## PNDS-CDL-ADDRESS-005 — Relevance failure clustering
- Status: SUPPORTED
- Parent: PNDS-CDL-ADDRESS-001
- Observed: 15/120 cases failed; failures clustered around same-entity wrong-relation distractors, especially head-of-government queries.
- PNDS relation: negative evidence against CDL as a sufficient semantic verifier. Structural/relation-aware routing and verification remain necessary.
- Evidence: E3. Decision: SUPPORTED.
- Next: relation-conflict, paraphrase and structurally matched distractors.

## PNDS-CDL-EXEC-001 — CASM Phase 1 structural execution interface
- Status: PARTIALLY_SUPPORTED
- Architecture: single-pass Boolean DAG; factorized node-wise query/key compatibility; runtime values excluded from routing.
- Controls/gates: numerical viability, gate intervention, task gradient, structural-input routing, true-edge intervention, held-out NOT->XOR routing, topology integrity. Static Mask vs CASM-S uses identical episode splits; copy-mask is an oracle preflight.
- Observed: the harness establishes a clean test of whether relevance-selected structure can control executable computation; the later L2 reconstruction is recorded separately because its held-out routing gate was not solved.
- PNDS relation: bridge from R_t to C_t and executable action selection.
- Evidence: E3. Decision: PARTIALLY_SUPPORTED.
- Next: finish pre-registered held-out routing controls without recurrence or auxiliary losses.

## PNDS-CDL-EXEC-002 — Phase 1.5A FactorizedRouterL2 reconstruction
- Status: INCONCLUSIVE
- Baseline/control: bilinear FactorizedRouter; Static Mask.
- Result: L2 tau=1 achieved 0/6 held-out successes in each dimension, 18/18 total; exact held-out accuracy 0.7857; wrong-gate basin 0.93-0.97; training loss reached about 0.0012; 145 tests passed.
- Observed: optimization loss fell while the intended held-out structural relation remained unsolved.
- PNDS relation: negative evidence that low training loss is sufficient for executable routing. Verification must inspect held-out structural behavior.
- Evidence: E4. Decision: INCONCLUSIVE.
- Next: continue the pre-registered reconstruction carefully; no contrastive loss, recurrence or dual masks before substrate is settled.

## CDL-to-PNDS synthesis

| Capability | Evidence | Status |
|---|---|---|
| Persistence | CDL provides memory/addressing mechanisms, but not long-lived state by itself | not established here |
| Addressability | Stage A and candidate scaling | strong bounded support |
| Executability | CASM structural gating | partial |
| Verification | failure clustering and held-out routing controls | required, unresolved |

Unified hypothesis under test:

CDL relevance -> cheap routing -> structural execution -> verification -> persistent state update.

The main unresolved transition is from a powerful relevance teacher to a cheap, causally correct executable router.

## PNDS-SETUP-001
- Status: SUPPORTED
- Purpose: universal protocol installation.

## PNDS-CDL-GATE-001 — Causal routed execution and verified persistent update
- Status: PRE-REGISTERED / STAGE-0 IMPLEMENTED; NO SCIENTIFIC RESULT YET
- Contract: docs/pnds/PNDS_GATE_001.md
- Implementation: casm_v01/pnds_gate_001/
- Question: can a cheap relevance router select a persistent structure, execute it causally, verify the outcome, and update state without answer/gold-structure leakage?
- Stage 0: explicit router-input allowlist; rejection of target/answer/outcome/true-edge/gold-structure/correct-action fields; verified-only state commit rule; causal intervention metric.
- Stage 1-6: oracle/static/random baselines -> cheap learned router -> causal intervention -> verifier -> verified state update -> 3-seed replication and scaling.
- Current evidence: implementation only. The preflight is a methodological control, not evidence that the PNDS hypothesis is true.
- Evidence: E1. Decision: DO NOT PROMOTE.
- Next: run the GATE-001 preflight in CI, then implement the outcome-driven router before adding recurrence or auxiliary losses.
