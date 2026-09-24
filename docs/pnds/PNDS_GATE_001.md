# PNDS-GATE-001 — Causal Routed Execution, Verification, and Persistent Update

**Protocol:** PNDS-URP v0.1  
**Status:** STAGE 2b HELD-OUT ROUTING COMPLETE (PARTIAL PASS, MIXED RESULT); STAGES 3-5 OPEN  
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

## Stage 2b — held-out routing comparison (2000 train / 300 held-out test, 5 training seeds)

**Provenance:** `cdl-attention-experiment` -> `pnds` -> `<this commit>` -> `casm_v01/pnds_gate_001/stage2b_heldout.py` + `run_stage2b.py` + `test_stage2b_heldout.py`. Runs executed locally (stdlib-only, sub-second) and reproducible via `python -m casm_v01.pnds_gate_001.run_stage2b --train-episodes 2000 --test-episodes 300 --candidates <N> --seeds 5`. A CI step is added by this commit so the artifact is produced on the runner as well.

### Design correction made before running

The Stage-1 and Stage-2 environments are **not interchangeable**. Stage-1 `make_episode` samples `gold_index` uniformly at random and uses opaque string candidates, so the gold candidate is uncorrelated with any observable feature; a router *cannot* beat chance there. Stage-2 `make_episode` derives `gold_index` from a latent bit-concept that the query and the gold candidate's descriptor both reveal noisily. Comparing a learned router against Stage-1 controls on Stage-1 episodes would therefore be meaningless.

Stage 2b fixes this by evaluating **all four arms on identical Stage-2 episodes**, drawn from the same `make_episode` used for Stage-2 training, with train seeds disjoint from test seeds. The arms differ only in the selector; the environment is shared.

### Result: 5 independent training seeds, identical held-out test episodes

| Candidates | learned | static | random | oracle | Δ(learned−static) | Δ(learned−random) |
|---|---|---|---|---|---|---|
| 8 | 0.7107 | 0.7133 | 0.1067 | 1.0000 | **−0.0027 ± 0.0092** | +0.6040 |
| 16 | 0.5520 | 0.5100 | 0.0600 | 1.0000 | **+0.0420 ± 0.0156** | +0.4920 |
| 32 | 0.4340 | 0.4200 | 0.0267 | 1.0000 | **+0.0140 ± 0.0043** | +0.4073 |

± values are the standard deviation of the per-training-seed delta (n=5). Because `static` is deterministic, the delta variance is attributable to learned-router training variance alone.

### What passed

- **Learned beats random decisively, at every candidate count** (success gate item 1, random half). The margin is +0.60 / +0.49 / +0.41 as distractor count rises, and random collapses toward chance as expected. This is robust across all 5 seeds.
- **Content dependence is confirmed.** The learned router is not exploiting position (see the Stage-2 position-bias control) and its converged weight vector is uniformly positive across all 8 bit dimensions (1.84–2.55, bias 3.53), i.e. it independently rediscovered "agreeing bits predict relevance" from outcome feedback alone, with no gold supervision.
- **Multi-seed replication works** (success gate item 7 partially met for this sub-claim): 5 independent training seeds, with per-seed deltas recorded.
- **Held-out discipline holds**: train and test seeds are disjoint, verified by a unit test; the router is frozen for evaluation; all arms see identical episodes, verified by a unit test.

### What did not pass

- **Learned does not reliably beat the fixed bit-agreement control** (success gate item 1, static half). At 8 candidates the delta is negative (−0.0027, t = −0.64 across seeds, straddling zero). At 16 it is positive on all 5 seeds (+0.0420, t = 6.03 across training seeds). At 32 it shrinks again (+0.0140). The advantage is therefore **regime-dependent and small**, not a uniform win.
- The environment's observable signal is nearly exhausted by fixed bit-agreement. At 8 candidates the static scorer alone reaches 0.7133 against an oracle ceiling of 1.0, so roughly 71% of the available headroom is captured without any learning. The learned router converges to a weighted version of the same feature and gains little.
- **No persistence, intervention, or verification claim is supported** by this stage. Success-gate items 2, 3, and 5 remain unaddressed.

### Interpretation

The honest reading is: **content-dependent routing generalizes and clearly beats chance, but in this environment it does not meaningfully beat a cheap fixed similarity scorer.** The learned router extracts the full available signal; that signal is largely already available for free from bit agreement.

This is a **boundary condition, not a failure of PNDS**: it says the synthetic Stage-2 environment is too easy for the similarity baseline to be discriminating. The useful consequence is that the next experiment must make the relevance signal *non-trivially* structured — e.g. relevance determined by a latent relation among query, candidate, and context rather than by direct feature agreement — rather than adding architectural complexity.

### Decision

- **Promote the Δ(learned−random) result** as the first held-out, multi-seed, no-gold-supervision evidence that outcome-trained routing generalizes. Evidence level E2 (controlled, multi-seed, held-out, but synthetic environment and no causal-intervention component).
- **Do not promote the Δ(learned−static) result.** It is zero or negative at 8 candidates, positive only at 16, and small everywhere. This does not satisfy success-gate item 1 against the fixed-similarity control.
- **Do not add persistence, recurrence, a verifier, or contrastive losses.** Per the pre-registered stage order and the asymmetric promotion rule, the response to a partial pass is to diagnose the environment, not to expand the architecture.
- **Next experiment: make relevance relationally structured**, so that fixed feature-agreement is no longer a near-optimal policy, then re-run the identical four-arm comparison. Only if learned then beats static across regimes does the gate advance to causal intervention (Stage 3).

## Stage 2c — relational relevance (2000 train / 300 held-out test, 5 training seeds)

**Provenance:** `cdl-attention-experiment` -> `pnds` -> commit `42f9814` -> CI run `36043770779` (green, `pnds-gate-001-stage2c` artifact) -> `casm_v01/pnds_gate_001/stage2c_relational.py` + `run_stage2c.py` + `test_stage2c_relational.py`. Runs executed locally (stdlib-only) and reproduced on the GitHub Actions runner by the workflow step added in this commit (`pnds-gate-001.yml`, Stage 2c 8/16/32-candidate steps). All 15 runner-produced per-seed values match the local run bit-for-bit.

### What changed relative to Stage 2b

Exactly one thing, as pre-registered: **the source of relevance.**

    Stage 2b:  R(q, c) ~ total feature agreement(q, c)
    Stage 2c:  R(q, c, x) = agreement(q, c) restricted to the positions x marks

The context is no longer inert. Its 1 bits mark `dim // 4` positions; the gold candidate matches the query on every marked position and deliberately *anti-matches* the query on the unmarked positions. Relevance is therefore a **conditional** rule: which candidate features matter is decided per episode by the context. Total query-agreement is actively anti-correlated with relevance, because the query-deceptive distractors agree with the query on the unmarked positions and fail it on the marked ones.

Everything else is identical to 2b: the same four arms, the same 2000/300 train/test split, the same 5 independent training seeds, the same held-out test episodes, the same metrics.

### Two earlier Stage 2c designs were rejected before running

Both were caught by representability checks, not by tuning:

1. **Indexed-lookup relation** (`descriptor[context[0]] == context[1]`). Static collapsed to chance (0.0933 vs 0.0967) and the relation was unique in 91% of episodes, but the learned router reached only 0.1933 against chance 0.125. Rejected: an indexed lookup is not representable by a linear feature basis, so the run would have measured model-class expressiveness, not learnability from outcomes.
2. **Three-way positional conjunction** (`descriptor[pos_a] == context[pos_a] AND descriptor[pos_b] == query[pos_b]`, with `pos_a`/`pos_b` drawn per episode). Static collapsed to 0.0467 and learned reached 0.1750, which looked like the intended result. It was not. Three diagnostics killed it:
   - In **284/300 episodes all 8 candidates** satisfied the *existential* version of the relation.
   - The best observable surrogate (`argmax` of `min(qa, xa)`) reached only **0.22**, and `argmax` of the product only **0.1567**.
   - A **supervised** least-squares linear fit over the router's 33-feature basis topped out at **0.30** (chance 0.125), rank 32/33.

   The relation was keyed on per-episode random positions that no router-visible feature encodes, so relevance was not a function of the router's inputs. The router at 0.175 was already near its achievable ceiling. This is the same class of flaw as design 1, relocated rather than fixed.

**Lesson recorded:** for a Stage 2c relation to test learning-from-outcomes rather than model class, the relation must be (a) computable from router-visible inputs alone and (b) expressible as a vector in the hypothesis class. The third design satisfies both, and this is now enforced by a unit test (`test_relation_is_observable_from_router_inputs`) so a future design cannot silently regress it.

### Result: 5 independent training seeds, identical held-out test episodes

| Candidates | learned | static | random | oracle | Δ(learned−static) | Δ(learned−random) |
|---|---|---|---|---|---|---|
| 8 | 0.8853 | 0.0000 | 0.1167 | 1.0000 | **+0.8853 ± 0.0077 (t = 258.0)** | +0.7687 ± 0.0077 |
| 16 | 0.7687 | 0.0000 | 0.0900 | 1.0000 | **+0.7687 ± 0.0126 (t = 136.4)** | +0.6787 ± 0.0126 |
| 32 | 0.5027 | 0.0000 | 0.0267 | 1.0000 | **+0.5027 ± 0.0401 (t = 28.0)** | +0.4760 ± 0.0401 |

± is the standard deviation of the per-training-seed delta (n=5); t is the paired t-statistic across seeds. Because `static` is deterministic and sees the identical episodes, all delta variance is learned-router training variance. Every per-seed delta is positive at every candidate count (25/25).

### Environment invariants, verified by unit test

- `test_gold_is_the_unique_satisfier` — gold is the **only** candidate satisfying the relation, in 400/400 sampled episodes. Without this the task would be ambiguous and the oracle arm capped below 1.0; an earlier draft leaked this (113/400 episodes had >1 satisfier) and was fixed by forcing a guaranteed violation on every random-mode distractor.
- `test_relation_is_observable_from_router_inputs` — the relation is decidable from query, context, and candidate descriptor alone. No hidden per-episode index is involved.
- `test_static_rule_cannot_express_the_relation` — the unchanged 2b static scorer is at 0.0000, i.e. the environment actually changed the source of relevance rather than only re-labelling it.
- The 2b methodology tests are mirrored exactly: identical episodes across arms, no gold supervision of the learned arm, disjoint train/test seeds, static has no trainable state, seed-level determinism, distinct multi-seed training seeds.

### What passed

- **Learned beats the fixed-similarity control decisively, at every candidate count, on all 5 seeds** (success gate item 1, static half). This is the result Stage 2b failed to produce: there the delta was −0.0027 at 8 candidates and never exceeded +0.042. Here the smallest delta is +0.50 and the weakest t-statistic is 28.0. The learned router recovers a rule that the fixed scorer provably cannot express.
- **Learned beats random decisively** (success gate item 1, random half): +0.77 / +0.68 / +0.48, all t > 26.
- **The learned rule is the intended one.** The router's context-gated weights converge uniformly positive (1.29–1.47 at 8 candidates) while the *total*-agreement weights converge uniformly **negative** (−1.25 to −1.72) and the bias is strongly positive (+4.00). That is a signed signature of exactly the intended rule, learned from outcome feedback alone with no gold supervision: reward agreement on context-marked positions, penalise agreement on unmarked ones, and carry the offset. It did not rediscover the 2b similarity rule; it inverted it.
  
  A separate relation-following oracle arm — which knows the relation and selects the unique satisfier — scores **1.0000** on the same held-out episodes, matching the `oracle` arm, because the environment is now unambiguous (gold is the unique satisfier in 400/400 sampled episodes). The learned arm reaches 0.8853, i.e. it recovers most but not all of the available headroom, and the shortfall is the honest measure of what outcome training leaves on the table. This oracle is a *ceiling reference*, not a competitor: the learned arm is not compared against it as a win, and its score should not be read as the learned arm matching the relation.
- **Held-out discipline holds**: train/test seeds disjoint, router frozen at evaluation, all four arms on identical episodes — each asserted by a unit test.

### What did not pass / is not claimed

- **No persistence, recurrence, verifier, or causal-intervention component.** Success-gate items 2, 3, and 5 remain unaddressed. Stage 2c is deliberately still a bandit, not a persistent decision loop.
- **Learned does not reach the oracle ceiling** (0.8853 vs 1.0 at 8 candidates, 0.5027 at 32). The relation-following oracle is 1.0000 by construction, so the whole 0.115 / 0.497 gap is candidate-count-dependent shortfall in outcome training as distractor density rises, not ambiguity in the task.
- The environment remains **synthetic and low-dimensional** (dim=8, bit descriptors). This is evidence level E2: controlled, multi-seed, held-out, no-gold-supervision, but synthetic and non-causal.

### Interpretation

Stage 2b's boundary condition was that the environment's relevance signal was nearly exhausted by fixed feature-agreement. Stage 2c removes that confound by construction: the static scorer scores **0.0000**, so any learned success above chance is attributable to relational learning and not to residual similarity. The router learns the context-gated rule from outcome feedback alone, and the learned-static gap is large, sign-consistent, and statistically overwhelming in all three candidate regimes.

### Decision

- **Promote the Δ(learned−static) and Δ(learned−random) results for Stage 2c** as the first held-out, multi-seed, no-gold-supervision evidence that outcome-trained routing can learn a *relational* relevance rule that a fixed similarity scorer provably cannot express. Evidence level E2.
- **Do not promote anything to E3.** No causal-intervention, persistence, or verification claim is supported.
- **Do not add persistence, recurrence, a verifier, or contrastive losses.** Per the asymmetric promotion rule, the response to a decisive pass is to advance to the next pre-registered diagnostic — the causal intervention stage (Stage 3) — not to expand the architecture. Stage 3 must apply the position-bias and intervention-consistency controls used in Stage 2/2b/2c to the new artifacts.
