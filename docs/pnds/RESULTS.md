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

## Mathematical-audit correction — 2026-09-24

The new PNDS mathematical specification changes the interpretation of the existing CDL/CASM results:

- Stage A CDL relevance remains bounded evidence for candidate ranking, not evidence of sublinear serving cost because the teacher evaluates every candidate.
- Stage B distillation remains INCONCLUSIVE; a cheap router must be evaluated on downstream execution and cost, not only teacher agreement.
- CASM Phase 1/1.5A results remain routing/execution substrate evidence. The L2 held-out result (0/18) is a failure boundary and must not be obscured by low optimization loss.
- Path verification, registration, adaptation regularization, state-to-computation intervention, and the Context-Scaling Proof have no completed PNDS result yet.
- The first Stage-2 synthetic environment was corrected after detecting an independent-gold generation flaw. The pre-correction version is methodologically invalid and not an evidence record.

### PNDS claim-status matrix

| Claim | Current status |
|---|---|
| CDL can rank relevant candidates on the bounded Stage-A task | SUPPORTED / bounded |
| CDL itself provides sublinear routing cost | NOT ESTABLISHED |
| Cheap Q/K router distilled from CDL solves the routing problem | INCONCLUSIVE |
| CASM can execute selected structural computation under bounded controls | PARTIALLY_SUPPORTED |
| Held-out structural routing is solved by the current L2 | FALSIFIED for the tested condition / 0 of 18 |
| Path-level verification improves over final-only verification | NOT_RUN |
| Verified-only persistent update improves state integrity | NOT_RUN |
| PNDS context-scaling proof | NOT_RUN |

### R13/R14 provenance

No R13/R14 identifiers were located in the current CDL/TAC Transformer repository search used for this audit. Do not assign PNDS IDs to those claims until a source artifact, branch, or commit is located. External references may be attached later as provenance without altering the existing ledger entries.

## PNDS-CDL-GATE-001 STAGE 3B — **INVALIDATED**
- Status: **INVALIDATED** — preserved under protocol §8, not erased
- Parent: PNDS-CDL-GATE-001
- Source: `pnds` @ `dd8f63c`
- Retracted by: `STAGE_3B_AMENDMENT_3.md` @ `f989430`
- Original claim: a learned, outcome-trained router must read key-indexed
  persistent state to route.
- Original result: `learned` 0.1500 vs `learned_no_state` 0.1333 (chance 0.125);
  10 seeds gave 4/10 positive margins, mean −0.011, binomial p = 0.581.
  Criterion F1 triggered; the stage was reported falsified.
- Flaw: `PersistentStateRouter._features` never read `state.key`. All 70 key
  blocks were computed for every candidate, making the feature map a function of
  `(candidate, target)` alone. Under the exact-`T` constraint the analytically
  ideal weights score `2·C(7,3)·T − 70·T = 0` for every candidate — a constant,
  verified in 200/200 episodes. The intended relation was **identically
  unrepresentable**, so the learned arm could not have beaten chance under any
  training signal.
- Why the flaw was invisible: every representability check in the registered
  design was an *arm*. `true_key` reads the relation directly and never passes
  through the learned feature map, so it reported 1.0000 while the basis sat in
  its own null space. A perfect oracle concealed a broken basis for four
  commits.
- Corrected protocol: gate the feature blocks on `state.key` so only the
  state's key block is non-zero; add a mechanism-level representability gate
  (protocol §34, v0.4).
- Rerun identifier: **PNDS-CDL-GATE-001 STAGE 3B-r2** (below).
- Evidence: E0. Decision: **INVALIDATED**, not FALSIFIED. Do not cite the F1
  trigger, the credit-assignment diagnosis, or the probe results as findings.

### Stage 3b invalidation chain — preserved in full

Per the user directive and protocol §14 ("do not retroactively rewrite old
experiment records to match later findings"), the historical chain is retained
rather than collapsed:

1. `3a03225` — pre-registration, criteria F1–F5.
2. `528ad89` — Amendment 1: the registered environment itself was defective (a
   `t`-only router scored 0.6965); corrected by the exact-`T` constraint.
3. `2a53338` — property tests for the corrected environment.
4. `dd8f63c` — F1 negative result, now INVALIDATED.
5. `d545e62` — Amendment 2: registered the learner-bottleneck probes before
   running them.
6. `b8b873b` — Probe 1 (k = 2, 28 keys), now INVALIDATED on the same basis.
7. Probe 2 (dense gradient) ran, and its **anomaly is what exposed the defect**:
   a dense gradient failed where fixed-key REINFORCE had succeeded, which
   contradicted the credit-assignment hypothesis and forced the
   representability audit that found the missing `state.key` gate.
8. `f989430` — proof that the feature map had collapsed the intended relation
   into its null space; complete retraction; corrected implementation; corrected
   F1/F2/F3/F4 results.
9. `32a0026` — Probe 2 retained as a retracted record, unretracted as a document.

That the negative result was reproduced, investigated, and then falsified by a
mechanism-level audit is itself methodological evidence, and is the reason §34
exists.

## PNDS-CDL-GATE-001 STAGE 3B-r2 — learned key-indexed routing over persistent state
- Status: **SUPPORTED**
- Parent: PNDS-CDL-GATE-001 (rerun of the INVALIDATED Stage 3b)
- Source: `pnds` @ `f989430`; record `docs/pnds/STAGE_3B_RESULT_CORRECTED.md`
- Protocol: PNDS-URP **v0.4** (representability gate §34 applied from the start)
- Hypothesis: a learned, outcome-trained router acquires a routing rule that
  requires reading the **key-indexed** persistent state, and that rule is
  destroyed by intervening on the state.
- Configuration: `dim = 8`, `k = 4` (70 keys), `T = 6`, 8 candidates, chance
  0.125, 250 train streams × 8 episodes, 100–300 held-out test streams.
- Representability gate: **PASS** — analytically ideal weights through the actual
  feature map separate gold from the best distractor in 300/300 episodes,
  minimum margin 4.0. Verified to FAIL on the committed map of `dd8f63c`.
- Baseline/control: `learned_no_state` (state withheld, matched budget);
  `t_static`; `static`; `anti_static`; `random`; `true_key` oracle; `wrong_key`
  and `corrupt_key` interventions.
- Seeds: 10 independent training seeds, identical schedule to the invalidated
  run, so the comparison isolates the feature-map fix.
- Result: `learned` **0.7867** vs `learned_no_state` **0.1333**. Ten seeds give
  **10/10** positive margins, mean **+0.624**, minimum +0.580, exact one-sided
  sign test p = **0.001**. F2 intervention effect **0.8333**. F3 within 0.02 of
  chance. F4 key-reset collapses accuracy to exactly **0.1250**. Dose-response
  monotone: 0.8667 → 0.3500 → 0.1000 → 0.0000 at target corruption 0/0.25/0.5/1.
  At 1000 and 4000 streams the learned arm reaches **1.0000**.
- Observed: the router reads `S_t = (t, K)` through a key-gated feature map and
  routes by it; replacing or corrupting the state destroys the ability at
  inference time, so the behaviour is carried by the state read and not by the
  weights.
- PNDS relation: first PNDS-GATE-001 evidence for C1 persistence + C2
  addressability jointly, under causal intervention.
- Limitations — the scope is deliberately narrow. This demonstrates
  **persistent state + state-indexed routing + learned acquisition + causal
  intervention**. It does NOT demonstrate: long-horizon memory; multi-step
  persistent computation; state composition; verifier-guided repair; continual
  learning; generalization to unseen state structures; scaling beyond the tested
  key space; or the full PNDS decision loop. The state is read, never written;
  there is no `S_{t+1}` update and no temporal separation between writing and
  reading. Stage 3b is a clean substrate result, not evidence for the PNDS
  architecture as a whole.
- Evidence: **E4** (reproduced across 10 independent seeds with pre-registered
  criteria and a passing representability gate).
- Decision: **SUPPORTED**, narrow scope as stated.
- Next: **Stage 4** — write state → temporal separation → retrieve/use state,
  with the §34 gate registered before implementation.

### Stage 3b reusable testing contract

Extracted from the corrected run as the template for Stage 4 and beyond:

```
environment invariants
        ↓
no-shortcut controls (static / anti_static / t_static at chance)
        ↓
actual-map representability gate   ← protocol §34, new
        ↓
training acquisition (F1, majority of seeds)
        ↓
held-out evaluation
        ↓
state intervention (F2, do(K ← K'))
        ↓
no-state / static controls (F3, F4)
```

Each layer is a precondition for interpreting the next. The representability
gate is the one that was missing, and it is the reason the first Stage 3b
attempt was invalidated rather than merely corrected.
