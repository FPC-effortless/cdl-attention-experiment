# PNDS-GATE-001 — Causal Routed Execution, Verification, and Persistent Update

**Protocol:** PNDS-URP v0.1  
**Status:** STAGE-0/1 COMPLETE IN CI; STAGE-2 SMOKE GREEN (NOT EVIDENCE)  
**Purpose:** Directly test the unresolved PNDS gap without introducing recurrence, answer leakage, gold structural labels, or a learned verifier before the routing substrate is measurable.

## Research question

Can a cheap relevance router select the correct persistent structure, execute it causally, verify its outcome, and update persistent state without receiving the answer, true edge set, or other gold structural metadata?

## Canonical loop

`S_t -> R_t -> C_t -> A_t -> O_t -> V_t -> S_{t+1}`

- **S_t:** persistent store of candidate structures and provenance.
- **R_t:** cheap query-conditioned router.
- **C_t:** selected executable structure.
- **A_t:** intervention/action emitted by the selected structure.
- **O_t:** environment outcome, exposed only after the action.
- **V_t:** verifier comparing predicted/expected outcome with observed outcome.
- **S_{t+1}:** state update performed only under the pre-registered commit rule.

## Core causal requirement

The environment must contain hidden structure that determines which action succeeds. The model receives an observation and query, but the router must not receive:

- true edge sets;
- target answer;
- target structure ID;
- oracle masks;
- post-action outcome;
- metadata that uniquely encodes the correct action.

The environment interface must support `do(action)` so action changes can be measured independently of observation correlation.

## Experimental arms

| Arm | Persistent state | Router | Structure | Verifier | Update |
|---|---|---|---|---|---|
| A Oracle | yes | oracle | gold | yes | yes |
| B Static retrieval | yes | fixed similarity | selected | yes | yes |
| C Cheap learned router | yes | Q/K or bilinear | selected | yes | yes |
| D Teacher-distilled router | yes | distilled CDL signal | selected | yes | yes |
| E Full PNDS | yes | learned | selected | learned/thresholded | verified only |
| F Shuffled control | yes | same | shuffled state | same | same |

A is a capacity ceiling, not evidence for learned routing.

## Required controls

1. **Query shuffle:** hold state fixed, replace query.
2. **State shuffle:** hold query fixed, permute candidate structures.
3. **Structure corruption:** alter the selected structure while preserving surface statistics.
4. **Action intervention:** replace the chosen action with a matched alternative.
5. **Outcome holdout:** verifier sees outcomes only after action.
6. **Reset:** remove persistent state.
7. **Cold-start:** query before any useful structure has been stored.
8. **No-answer channel:** assert programmatically that target/outcome fields never enter router inputs.
9. **Gold-metadata audit:** router input tensors must be schema-validated against an allowlist.
10. **Update ablation:** compare verified update against unconditional update.

## Primary metrics

### Routing
- top-1 correct structure
- MRR
- selected-vs-random action success
- router latency / FLOPs proxy
- candidate count

### Causal execution
- `P(success | selected)`
- `P(success | intervened action)`
- action intervention effect
- structure corruption effect

### Verification
- verifier precision
- verifier recall
- false commit rate
- false reject rate
- calibration / Brier score where probabilistic

### Persistence
- carry success after reset boundary
- update gain
- verified-update vs unconditional-update delta
- stale-state failure rate

### Efficiency
- router compute
- full-context compute
- candidate count
- active structure fraction

## Success gate

A run can be promoted only if all are true:

1. Learned router beats random and fixed-similarity controls on held-out structures.
2. Selected actions show a positive causal intervention effect.
3. Shuffling persistent state removes or materially reduces the advantage.
4. Router inputs contain no target/outcome/gold-structure fields.
5. Verified-only update outperforms unconditional update on held-out episodes.
6. False commits remain below the pre-registered threshold.
7. Results reproduce across at least 3 seeds.
8. No component receives the hidden true structure during inference.

A single positive task-accuracy result is insufficient.

## Falsification conditions

The PNDS hypothesis is weakened if:

- router accuracy improves but action intervention has no effect;
- state shuffling does not materially affect performance;
- unconditional updates match verified updates;
- the verifier can only succeed when given gold structural information;
- distilled routing quality collapses under paraphrase or structure-preserving distractors;
- compute cost approaches full candidate evaluation.

## Stage order

**Stage 0:** interface and leakage audit.  
**Stage 1:** static/oracle/random routing baselines.  
**Stage 2:** cheap learned router.  
**Stage 3:** causal action intervention.  
**Stage 4:** verifier.  
**Stage 5:** verified persistent update.  
**Stage 6:** 3-seed replication and candidate-count scaling.

Do not add recurrence, reinforcement learning, contrastive routing loss, or dual masks until Stage 2 passes its held-out routing gate.

## Expected interpretation

- **Routing succeeds, execution fails:** relevance is not equivalent to executable structure.
- **Execution succeeds, verification fails:** action correctness is not reliably identifiable from outcomes.
- **Verification succeeds, persistence fails:** state representation/update is the bottleneck.
- **All stages pass:** first controlled evidence for the complete PNDS loop.
- **Any stage fails:** retain the failure as a boundary condition and isolate the failing component.

## External evidence policy

Related literature establishes that explicit non-parametric memory and learned retrieval can improve knowledge-intensive prediction, but does not establish this complete causal loop. RAG, differentiable memory, and neural-module approaches therefore remain comparative design references rather than PNDS evidence.

## Artifact requirements

Every run must record:

`repo -> branch -> commit -> PNDS ID -> seed -> config -> artifact -> metrics -> interpretation -> decision`

No result is promoted without this provenance chain.


## Stage 1 implementation

The first executable harness is `casm_v01/pnds_gate_001/stage1_baselines.py`. It compares oracle, fixed token-overlap retrieval, and random routing on a hidden synthetic action environment. The router receives only query and candidate state fields; the gold index is retained inside the environment for scoring and is never passed to a routing function. The harness also measures matched action intervention effects. Stage 1 is a baseline/capacity characterization only and cannot establish the PNDS hypothesis.

## Stage 0+1+2 — first CI-produced artifacts

**Provenance:** `cdl-attention-experiment` -> `pnds` -> `b2b3268` (package-marker fix `1ea5856` + CI-trigger fix) -> workflow `PNDS-GATE-001 Preflight` -> run `36008057693` (green, 1m38s) -> artifacts `pnds-gate-001-stage1`, `pnds-gate-001-stage2-smoke`.

### Stage 1 baseline artifact (30 episodes, 8 candidates)

| Arm | success | top-1 | MRR | intervention |
|---|---|---|---|---|
| oracle | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| static_overlap | 0.1333 | 0.1333 | 0.1333 | -0.0333 |
| random | 0.0667 | 0.0667 | 0.0667 | -0.1667 |

Interpretation: the environment is solvable (oracle = 1.0) and the chance floor is at ~1/8 (0.125), with `random` slightly below from finite-sample noise. `static_overlap` sits at chance, confirming that opaque candidate keys carry no lexical shortcut to the gold candidate. This is a **capacity/chance characterization**, not PNDS evidence. Success gate items 1 and 2 are not addressed by Stage 1.

### Stage 2 smoke artifact (200 train / 50 held-out test, 8 candidates, dim 8, noise 0.10)

- `test_success_mean` = **0.4600** (chance = 0.1250)
- `test_intervention_effect_mean` = **0.4400**
- training success = 0.5450, training intervention = 0.4800

**Status: NOT EVIDENCE.** This is a smoke run (200 training episodes) whose purpose is to prove the pipeline executes and produces an artifact, not to evaluate the router. It is deliberately far below the 2000/300 configuration used for evaluation.

Two controls were run against this artifact before any interpretation:

1. **Position-bias control.** The router selects across all 8 candidate positions with chi-square 4.40 against uniform (df=7, critical value 14.07 at p<0.05), so selection is not explained by candidate position. Per-position success rates range 0.20-0.71 with no position near the oracle ceiling, indicating the router tracks candidate content rather than a positional artifact.
2. **Intervention consistency.** `intervention_effect == 1` implied `success == 1` in 50/50 rows, and `intervention_effect == -1` implied `success == 0` in 50/50 rows (0 violations). `P(intervention=1)` equals `P(success)` exactly, as expected for the matched-alternative definition; the metric is internally consistent and is not independently measuring anything beyond the success indicator at this stage.

### Known limitations blocking promotion

- **Single seed.** Success gate item 7 requires at least 3 seeds; this run is seed 0 only.
- **No held-out routing gate.** Success gate item 1 (learned router beats random and fixed-similarity on held-out structures) is not yet evaluated, because no learned-router arm has been compared against `static_overlap` and `random` under the same Stage-1 protocol.
- **Intervention is not yet causal evidence.** The current `intervention_effect` is a within-episode re-scoring against `(selected+1) % n`, not a `do()` intervention on a persisted structure. It agrees exactly with the success indicator and therefore cannot yet support success-gate item 2.
- **Persistence untested.** No reset/shuffle/corruption arm has been run; success-gate items 3 and 5 are unaddressed.
- **No verifier.** Stages 4-5 are unimplemented; `verify_state_update` exists as an interface only.

### Decision

Stage 2 remains at the smoke level. The next scientifically meaningful step is a **held-out routing comparison** of the learned router against the Stage-1 `static_overlap` and `random` arms at the evaluation configuration (2000 train / 300 test), multi-seed, with the position-bias and intervention-consistency controls applied to every artifact. No architectural complexity is added until that comparison passes.
