# CASM-X9R — Cardinality-valid executor control

## Status and purpose

This experiment is a **validity repair**, not a rescue or reinterpretation of CASM-X9.

CASM-X9 remains formally invalid because its frozen `canonical_functional` positive control missed the preregistered thresholds. X9R asks only whether a slot/cardinality-invariant executor can satisfy that control contract while still training exclusively on cardinalities 2/3/4.

No result from X9R changes X9's classification.

## Question

> Can the contextual transition be learned under fixed-answer-only supervision in an explicit workspace whose transition computation is invariant to absolute internal slot identity, such that deterministic canonical binding transfers from trained cardinalities 2/3/4 to unseen cardinalities 5/6 through depth 96?

A positive X9R result validates an executor architecture for a later binding-generalization experiment. It does not establish learned binding or ontology discovery.

## Frozen diagnosis being tested

X9's original positive control used canonical `e -> slot e` binding with an executor containing absolute slot embeddings and a flattened eight-slot state projection. Training on `n<=4` therefore never presented slots 4/5 as active canonical variable slots, while unseen `n=5/6` simultaneously introduced more active variables and new active absolute slot positions.

X9R removes that absolute-slot dependence from the transition computation.

## Data and supervision

Use the same variable-cardinality contextual world as X9.

Training:

- cardinalities `n ∈ {2,3,4}` only;
- deterministic schedule `2,3,4,2,3,4,...` by optimizer step;
- depth 8;
- batch size 128;
- 10,000 optimizer steps;
- AdamW learning rate `2e-3`;
- weight decay `1e-4`;
- fixed final external register `0` is the only target in the training loss.

No teacher forcing, intermediate-state targets, hidden-register targets, semantic labels or binding labels.

Evaluation:

- cardinalities `n=2,3,4,5,6` separately;
- fresh IID depth 8;
- held-out composition depths 12 and 24;
- stress depths 48 and 96;
- `eval_n=256` per `(n,depth)` suite.

## Binding

Both X9R regimes use deterministic collision-free canonical binding:

`external e -> internal slot e` for every `n<=6`.

Binding is supplied, not learned. There is no binding generator in the decisive validity test.

The eight-slot categorical workspace and X8/X9 probability transport/update semantics are retained.

## Regimes

### `x9_absolute_slot_control`

Replication/diagnostic control using the X9 absolute-slot executor:

- learned slot embeddings;
- flattened eight-slot state encoding;
- destination slot representation;
- canonical `e -> slot e` binding.

This regime is expected to reproduce the X9 validity weakness and is not required to pass.

### `local_equivariant_control`

Decisive repaired positive control.

Its transition network may consume only:

- the opaque command/family embedding;
- the binding-gathered categorical/expected value at source `a`;
- the binding-gathered categorical/expected value at source `b`;
- the binding-gathered categorical/expected value at `dst`.

It must **not** consume:

- an absolute internal slot embedding;
- a flattened or position-sensitive eight-slot state vector;
- absolute destination-slot identity;
- active cardinality `n` as a transition feature;
- external variable index `e` as a transition feature;
- target states or semantic labels.

The same transition parameters are reused at every program step, every variable and every cardinality. State mutation still occurs through the supplied binding and explicit workspace.

This architecture is intentionally local because the benchmark's state-dependent semantic choice is a function of the current values at `a`, `b`, and `dst`; those values were already supplied to the X9 transition path. X9R removes absolute-slot nuisance dependence rather than adding target information.

## Seeds

Three independent seeds:

- train `20261031`, eval `20261111`;
- train `20261032`, eval `20261112`;
- train `20261033`, eval `20261113`.

No seed may be dropped or replaced based on outcome.

## Integrity requirements

Before training, tests must establish:

1. `local_equivariant_control` contains no learned slot-identity embedding;
2. its transition input is invariant to a simultaneous permutation of internal slot columns and supplied binding columns;
3. no flattened eight-slot tensor enters its transition MLP;
4. it cannot access active cardinality `n` except through tensor shapes used for binding transport;
5. it cannot access external variable indices except through binding-based gather/update;
6. fixed-answer loss is invariant to hidden/intermediate targets and semantic labels;
7. changing final register-0 target can change the loss;
8. gradients reach the transition/value/command parameters;
9. canonical bindings for every `n=2..6` are row-stochastic, collision-free and probability-valid;
10. adversarial slot permutations produce numerically equivalent external rollouts (within `1e-5` soft probability tolerance before argmax);
11. variable-cardinality rollouts have correct shapes through depth 96;
12. training schedule is exactly `2,3,4` repeated;
13. training aborts on non-finite or negative categorical loss.

## Frozen interpretation

### Validity PASS

X9R validates `local_equivariant_control` for a successor binding experiment only if **every seed and every cardinality `n=2..6`** satisfies:

- hard answer-final accuracy >=99% on every suite;
- hard step-state exactness >=95% at depths 24, 48 and 96;
- hard hidden-register accuracy >=99% at depths 24, 48 and 96.

No averaging across seeds, cardinalities or depths may rescue a failed cell.

### Near-pass

If every cell has:

- answer-final >=98%;
- deep step-state exact >=95%;
- deep hidden-register >=98%;

but the strict validity PASS fails, classify X9R as **near-pass but still insufficient for a decisive binding experiment**. Do not proceed to binding claims without another validity repair.

### FAIL

If any seed/cardinality has:

- IID depth-8 answer-final <90%; or
- deep hidden-register <90%;

classify the local executor as not cardinality-valid for this benchmark.

### Diagnostic comparison

Report `x9_absolute_slot_control` with the same metrics. If the local-equivariant control passes while the absolute-slot control reproduces X9-like degradation, that supports the diagnosis that absolute slot-position dependence was a material X9 validity confound.

If both pass, X9's original control failure was likely optimization/seed variance rather than a necessary architectural defect.

If both fail, do not modify binding generation next; repair the executor/data test first.

## Successor boundary

Only after strict X9R validity PASS should the next experiment test binding generalization. That successor should use the validated local-equivariant executor and compare:

- X9-style independent descriptor-to-row generation; versus
- a coordinated/permutation-equivariant set binding generator.

The purpose will be to determine whether coordination among variable descriptors, rather than executor slot exposure, is required to allocate new variables to new slots.
