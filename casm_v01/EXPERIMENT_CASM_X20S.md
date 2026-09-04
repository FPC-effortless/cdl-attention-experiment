# CASM-X20S — storage-onset scheduling for discrete state construction

## Status and purpose

CASM-X20S is preregistered before implementation or execution.

Frozen X20R result base:

`a4b8ee98dd300dc51e4398c84020a2e90c2cccc6`

Frozen authorized X20R scientific executable head:

`300edab0ef4abcf130db4ce4957c00035da246de`

X20R is valid but its graph-conditioned binary-forward straight-through treatment is an optimization failure: seen competence is 0/6, every seed collapses to zero hard records, and strong/partial unseen extension are ineligible. The positive control passes 6/6.

A post-hoc diagnostic of the already-completed X20R artifacts motivates this successor without modifying X20R: on all six fresh X20R seeds, the straight-through treatment is already at the empty-state answer-loss ceiling and zero storage by step 500, while the matched soft-X20 comparator has already reduced answer loss below 0.001 by step 500. The successor therefore tests storage-pressure timing rather than another estimator or architecture.

This diagnostic is used only to preregister one fixed schedule family. There is no seed replacement, schedule sweep, adaptive stopping, or tuning after execution begins.

## Scientific question

> Does immediate storage pressure create the absorbing all-absent optimum in hard-forward state construction, and can the same straight-through constructor become robust if answer learning establishes an executable path before the unchanged final storage pressure is introduced?

## Frozen substrate

Inherit unchanged from X20/X20R:

- 8 supplied candidate entities;
- version-correct temporal dependency graph;
- shared reverse-topological `ProgramStateConstructor`;
- deterministic nonlearned candidate identity codes;
- local-equivariant gated executor;
- final-answer-only task supervision;
- no live-mask/cardinality/intermediate/hidden/role/address labels;
- train live cardinalities `{2,3,4}` only;
- unseen live cardinalities `{5,6}`;
- train depth 12;
- evaluation depths 12/24/48/96;
- batch size 128;
- AdamW, 12,000 steps, cosine LR `2e-3 -> 2e-4`, weight decay `1e-4`, gradient clip `1.0`;
- hard existence decision `g_soft >= 0.5`;
- exact binary-forward, identity-through-soft-backward straight-through estimator from X20R;
- no top-k, cardinality repair, matching, rerouting, hidden oracle, entropy loss, polarization loss, gate-label loss, or live-mask auxiliary loss.

The X20 generator, constructor, executor, and hard/soft evaluation semantics are frozen unchanged. X20S may add only a training-time scalar schedule multiplying the existing storage term.

## Environment provenance

The scientific workflow must pin and record:

- Python `3.11.16`;
- PyTorch CPU `2.14.0+cpu`;
- the exact executable git head;
- harness version;
- train/eval seeds;
- package/runtime version strings in every result JSON.

Changing the runtime after any scientific seed has run invalidates cross-seed comparison.

## Straight-through gate

Exactly as frozen in X20R:

`g_soft = sigmoid(logit)`

`g_hard = 1[g_soft >= 0.5]`

The training forward value is exactly `g_hard`; the backward derivative to the soft gate input is identity-through-soft.

Evaluation always uses the raw learned `g_soft` and the same frozen hard threshold. The schedule is never used during evaluation.

## Frozen schedule rationale

The completed X20R artifacts show a repeatable timing separation:

- every straight-through seed has collapsed to zero storage and the no-state answer-loss ceiling by step 500;
- every matched soft comparator has answer loss below 0.001 by step 500.

X20S therefore fixes a conservative 1,000-step answer-only warmup, followed by either immediate full pressure or a 1,000-step linear ramp. These values are fixed now and are not tuned during the run.

## Frozen regimes

All graph-conditioned learned regimes must begin from bit-identical parameters for each seed and receive identical training batches in identical order.

1. `canonical_live_mask`
   - unchanged positive control;
   - no learned existence decision.

2. `immediate_st`
   - exact X20R graph-conditioned straight-through treatment;
   - `lambda(step) = 0.05` for all 12,000 steps;
   - required replication control for the all-absent failure.

3. `no_storage_st`
   - same graph-conditioned straight-through model;
   - `lambda(step) = 0` for all 12,000 steps;
   - diagnostic for whether answer-only hard-forward learning can establish an executable path without pressure;
   - not a valid efficient-state solution unless it independently satisfies the frozen state-selection thresholds.

4. `delayed_abrupt_st`
   - same graph-conditioned straight-through model;
   - steps 1..1000: `lambda = 0`;
   - steps 1001..12000: `lambda = 0.05`.

5. `delayed_ramp_st`
   - same graph-conditioned straight-through model;
   - steps 1..1000: `lambda = 0`;
   - steps 1001..2000: `lambda = 0.05 * (step - 1000) / 1000`;
   - steps 2001..12000: `lambda = 0.05`.

6. `delayed_ramp_structure_blind`
   - identical to `delayed_ramp_st` except dependency/version connectivity propagation is removed exactly as in X20/X20R;
   - same schedule, optimizer, seeds, batches, parameter count, and hard estimator.

No other learned treatment is permitted in the preregistered scientific run.

## Objective

For learned regimes:

`L = L_answer(execute(g_ST)) + lambda(step) * mean(g_ST)`

No additional loss term is permitted.

`lambda(step)` is deterministic from the regime name and global step only. It may not depend on loss, accuracy, gate statistics, cardinality, seed, or evaluation results.

## Fresh seeds

Use six fresh paired seeds, fixed before implementation:

- train `20261191`, eval `20261271`;
- train `20261192`, eval `20261272`;
- train `20261193`, eval `20261273`;
- train `20261194`, eval `20261274`;
- train `20261195`, eval `20261275`;
- train `20261196`, eval `20261276`.

No seed replacement or selective scientific rerun is permitted.

## Integrity requirements before scientific training

Tests/workflow must establish all of the following before any scientific job is authorized:

1. branch ancestry contains frozen X20R result `a4b8ee98dd300dc51e4398c84020a2e90c2cccc6`;
2. this preregistration commit predates implementation;
3. the frozen X20 data/model/runner substrate files remain unchanged from the X20R frozen result unless the X20S runner imports them without modification;
4. `immediate_st` is numerically identical to X20R `straight_through_instantiation` when weights, batches, and optimizer state match;
5. all graph-conditioned learned regimes begin bit-identically for each seed;
6. all learned graph regimes have equal parameter/trainable-parameter counts;
7. `delayed_ramp_structure_blind` differs only in graph connectivity propagation;
8. straight-through forward values are exactly binary `{0,1}`;
9. evaluation hard threshold remains raw `g_soft >= 0.5`;
10. schedule values are exactly:
    - immediate: 0.05 at steps 1, 500, 1000, 1001, 2000, 12000;
    - no-storage: 0 throughout;
    - delayed abrupt: 0 through step 1000, 0.05 from 1001 onward;
    - delayed ramp: 0 through 1000, 0.00005 at 1001, 0.025 at 1500, 0.05 at 2000 and thereafter;
11. storage forward value is exactly hard record fraction multiplied only by the deterministic schedule coefficient;
12. answer gradients and storage gradients reach the constructor through the same X20R straight-through path;
13. no learned regime receives the live mask, active cardinality, target answer, hidden trajectory, causal label, or evaluation-only metadata as constructor input;
14. no per-candidate learned gate/ID table is introduced;
15. non-finite gates/losses/states abort;
16. canonical positive-control executor remains exact under the inherited X20 regression suite;
17. result JSON records exact git SHA, Python version, PyTorch version, harness version, seeds, optimizer contract, schedule contract, training history, and evaluation results.

## Evaluation

Use the same X20R evaluation matrix:

- live cardinalities `n=2,3,4,5,6`;
- suites `iid_depth_12`, `composition_depth_24`, `stress_depth_48`, `stress_depth_96`;
- `eval_n=256` per cell;
- both hard and raw-soft execution metrics;
- hard existence precision/recall/F1;
- mean absolute hard record-count error;
- mean raw gate on live candidates and distractors;
- no matching or count repair.

## Frozen classification

### Positive-control validity

`canonical_live_mask` must satisfy every seed and `n=2..6`:

- hard answer-final >=99% every suite;
- hard deep step-state exactness >=95%;
- hard deep live-register accuracy >=99%.

Failure invalidates X20S.

### Seen competence

A learned treatment is extension-eligible only if every 6/6 seed at trained live `n=2,3,4` satisfies:

- hard and raw-soft answer-final >=99% every suite;
- hard and raw-soft deep step-state exactness >=95%;
- hard and raw-soft live-register accuracy >=99%;
- hard existence precision >=0.95;
- hard existence recall >=0.95;
- mean absolute hard record-count error <=0.25;
- mean raw soft gate on live candidates >=0.90;
- mean raw soft gate on distractors <=0.10.

Any hard IID/base answer-final <80% on a seen cardinality is an optimization failure for that seed.

### Strong state-instantiation extension

A strong PASS for a learned treatment requires every 6/6 seed at unseen live `n=5,6` to satisfy the same capability and state-selection thresholds. No averaging rescues a failed seed/cell.

### Partial extension

If strong fails but every unseen capability cell has hard/raw-soft answer >=90%, deep step/live-register >=80%, and hard existence F1 >=0.80, classify partial and report the exact failed thresholds.

### Storage-onset causal interpretation

A **delayed-onset repair PASS** requires:

1. `delayed_abrupt_st` or `delayed_ramp_st` passes 6/6 seen competence and 6/6 strong unseen extension;
2. `immediate_st` fails the same every-seed robustness criterion or reproduces the X20R all-absent optimization collapse on at least one seed; and
3. the successful scheduled treatment reaches the final frozen `lambda=0.05` for at least the last 10,000 steps (`delayed_abrupt_st`) or last 10,000 steps including ramp endpoint and post-ramp (`delayed_ramp_st` reaches full lambda at step 2000 and holds it through 12000).

Interpretation:

- if both delayed treatments pass strongly, conclude delayed onset is sufficient in this controlled setup; do not claim the linear ramp is necessary;
- if only `delayed_ramp_st` passes strongly, conclude gradual onset after answer warmup is required relative to the preregistered abrupt comparator in this setup;
- if delayed treatments fail seen competence, the storage-timing hypothesis fails and the frontier remains on discrete construction optimization;
- if `immediate_st` unexpectedly passes 6/6 as well, report robust state construction but do not attribute the result causally to delayed storage onset.

### No-storage diagnostic

`no_storage_st` is not evidence for efficient state construction unless it independently satisfies the full state-selection thresholds. If it executes correctly but over-instantiates, report that answer-only hard-forward training can learn execution but not selective state existence.

### Structure ablation

If `delayed_ramp_st` is seen-competent 6/6 while `delayed_ramp_structure_blind` is not, report that dependency/version connectivity is required for robust scheduled discrete selection under this controlled benchmark. Do not make an unseen graph-effect claim unless both are seen-competent.

## Successor boundary

Only a valid 6/6 strong PASS from a graph-conditioned discrete learned treatment authorizes the next model-development experiment on **reuse/merge of repeated or aliased observations into working-state identity**.

If no graph-conditioned X20S treatment earns the strong criterion, do not advance to reuse/merge, lifecycle, persistence, controller/program induction, or verifier-guided repair. Freeze the exact optimization boundary instead.

## Claim boundary

Even a strong X20S result establishes only learned discrete instantiation/ignore of supplied candidate records from a supplied program graph under answer-only supervision plus deterministic storage pressure.

It does not establish entity discovery from raw language/perception, semantic ontology induction, alias resolution, reuse/merge, deletion/lifecycle, cross-episode persistence, controller/program induction, or general reasoning.
