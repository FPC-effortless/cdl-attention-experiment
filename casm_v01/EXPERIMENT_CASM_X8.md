# CASM-X8 — Does discrete variable topology emerge without an injective binding prior?

## Motivation

CASM-X7 established that fixed-answer-only supervision can select four distinct useful slots from eight candidates and preserve exact hidden execution through depth 96 **when the architecture supplies an injective 4→8 assignment family**.

X8 removes that all-different constraint.

The treatment receives the same four external registers and eight candidate internal slots, but each external register independently learns a row-normalized distribution over the eight slots. Multiple registers may therefore share a slot, and no architectural rule forces four distinct selected locations.

The experiment asks whether answer-only learning independently sharpens this unconstrained binding into a discrete one-variable-per-slot decomposition.

## Question

> When the model is not told that external variables must occupy distinct internal slots, does fixed-answer-only supervision still discover a sharp, collision-free discrete binding that supports hidden executable state and depth extrapolation?

A positive result would reduce the supplied topology prior beyond X7. A negative result would locate the current boundary: one-to-one variable topology would still need to be supplied even though the specific slot identities can be learned.

## External task contract

The external world is unchanged from X7:

- four external registers;
- values `0..15`;
- contextual command semantics;
- explicit recurrent state;
- train depth 8;
- only the final value of external register `0` is supervised;
- external registers `1–3` are never target labels;
- no intermediate-state targets;
- no teacher forcing;
- no semantic-operator labels.

Evaluation remains IID depth 8, held-out composition depths 12 and 24, and stress depths 48 and 96.

## Candidate internal state

The model has eight candidate internal slots and the same seventeenth categorical `EMPTY` symbol used by X7.

Let `B[e,s]` be a 4×8 binding matrix. Every external-register row is normalized:

`sum_s B[e,s] = 1`.

Unlike X7, the treatment places **no constraint on column occupancy**:

`c[s] = sum_e B[e,s]`

may be less than, equal to, or greater than one.

To keep every internal slot a valid categorical distribution even when several external registers overlap, define

`d[s] = max(1, c[s])`.

Initial internal state is

`p_s = sum_e (B[e,s] / d[s]) onehot(value_e) + (1 - c[s] / d[s]) onehot(EMPTY)`.

Therefore every `p_s` sums to one:

- if `c[s] <= 1`, unused capacity is explicit `EMPTY` mass;
- if `c[s] > 1`, the slot is a normalized mixture of the colliding register values with no `EMPTY` mass.

External decoding remains

`q_e = sum_s B[e,s] p_s`.

Because each row of `B` sums to one, every decoded external register is also a valid categorical distribution.

Destination updates use the same row binding:

`p'_s = (1 - B[dst,s]) p_s + B[dst,s] new_value`.

This remains categorical for every slot because `B[dst,s]` lies in `[0,1]`.

All external-register-specific lookup, command argument representation, destination representation, state update and external decoding must pass through the same binding. No direct external-register embedding is permitted.

## Regimes

Four parameter-matched regimes start from identical transition parameters and binding-score initialization.

### `canonical_sparse`

Positive implementation control. Fixed mapping register `e` → slot `e` for `e=0..3`; slots 4–7 are surplus.

### `learned_injective`

X7 topology control. The same trainable 4×8 scores are projected through the exact convex mixture over all `8P4 = 1680` injective assignments.

This verifies that any X8 failure is not caused by the shared transport/generalized-state implementation.

### `learned_dense`

**Decisive X8 treatment.**

Each external-register row is independently normalized:

`B[e,:] = softmax(logits[e,:] / temperature)`.

There is:

- no column-capacity constraint;
- no injectivity constraint;
- no matching or all-different loss;
- no sparsity penalty;
- no entropy penalty;
- no collision penalty;
- no identity prior;
- no assignment labels.

Rows may converge to the same slot.

### `diffuse_dense`

Fixed uniform `1/8` binding for every row. Diagnostic negative control.

## Hard evaluation without topology repair

The decisive `learned_dense` hard binding is formed independently per row:

`slot_e = argmax_s B[e,s]`.

Each row becomes one-hot at its own argmax.

**No collision repair, Hungarian matching, injective projection, permutation projection or best-assignment replacement is allowed for the decisive dense evaluation.**

If two registers choose the same slot, the hard representation retains that collision. This prevents evaluation from reintroducing the topology prior that X8 is intended to remove.

For diagnostics only, the soft matrix may also be scored against the best possible injective assignment, but that score must never alter predictions.

`learned_injective` retains its X7 best-injective discrete projection because it is a control whose architecture explicitly supplies that family.

## Budget

Preregistered before X8 implementation/execution:

- 8,000 optimization steps;
- batch size 128;
- train depth 8;
- AdamW learning rate `2e-3`;
- weight decay `1e-4`;
- binding temperature `1.0`;
- three independent seeds:
  - train `20261011`, eval `20261091`
  - train `20261012`, eval `20261092`
  - train `20261013`, eval `20261093`
- evaluation sample count 384 per suite;
- evaluation suites:
  - IID depth 8;
  - held-out composition depth 12;
  - held-out composition depth 24;
  - stress depth 48;
  - stress depth 96.

No seed may be replaced after observing its outcome.

## Integrity requirements

Before training, tests must establish:

1. `learned_dense` initializes near uniform (`~1/8` row maximum);
2. every dense binding row sums to one within `1e-6`;
3. dense column occupancy is allowed to exceed one under adversarial colliding logits;
4. generalized internal-state construction remains normalized when column occupancy exceeds one;
5. decoded external distributions remain normalized under collisions;
6. destination updates remain normalized under collisions;
7. the dense hard projection is independent row argmax and deliberately preserves collisions;
8. no injective repair is applied to dense predictions;
9. the `learned_injective` control still satisfies X7 row/column/total-mass contracts;
10. fixed-answer categorical loss is finite and non-negative;
11. loss ignores intermediate targets, hidden final targets and semantic labels;
12. gradients reach both dense binding scores and transition parameters;
13. no direct external-register embedding bypass exists;
14. all four regimes are parameter matched;
15. all four regimes produce valid depth-96 rollout shapes.

The training runner must abort immediately on any non-finite or negative categorical loss.

## Preregistered interpretation

### Control prerequisites

On **every seed**, both `canonical_sparse` and `learned_injective` must achieve:

- answer-final accuracy ≥99% on every suite;
- full step-state exactness ≥95% at depths 24, 48 and 96;
- hidden-register accuracy ≥99% at depths 24, 48 and 96.

Additionally, `learned_injective` must satisfy:

- mean row maximum ≥0.90;
- best injective-assignment score ≥3.60 / 4.00;
- four distinct projected slots.

If either control fails, no strong X8 topology conclusion is made.

### Strong emergent discrete topology

If both controls pass, `learned_dense` supports the strong X8 result only if **every seed** satisfies all of the following without collision repair:

- hard row-argmax answer-final accuracy ≥99% on every suite;
- hard row-argmax full step-state exactness ≥95% at depths 24, 48 and 96;
- hard row-argmax hidden-register accuracy ≥99% at depths 24, 48 and 96;
- soft-binding answer-final accuracy ≥99% on every suite;
- soft-binding step-state exactness ≥95% at depths 24, 48 and 96;
- soft-binding hidden-register accuracy ≥99% at depths 24, 48 and 96;
- mean row maximum ≥0.90;
- independent row argmax uses four distinct candidate slots;
- collision count is zero;
- diagnostic best-injective score ≥3.60 / 4.00.

No cross-seed averaging may rescue a failed seed.

### Distributed executable topology

If the soft dense binding clears the capability thresholds but hard row-argmax execution, row sharpness or unique-slot criteria fail, X8 supports **distributed/non-discrete executable state**, not spontaneous discrete variable decomposition.

### Colliding discrete topology

If rows become sharp but independent argmax contains collisions, the result shows that answer-only supervision did not recover one-variable-per-slot topology without the injective prior, even if soft execution remains strong.

### Hard discrete execution without soft stability

If hard row-argmax execution passes but soft execution fails, no strong topology claim is made; this would indicate a projection-dependent solution rather than a stable learned transport.

### Answer-only shortcut

If answer-final accuracy reaches ≥95% but step-state exactness or hidden-register accuracy falls below 80% at depth 24 or deeper, the model found an answer strategy without recovering hidden executable state.

### Dense optimization failure

If both controls pass but `learned_dense` IID answer-final accuracy remains below 80%, the unconstrained topology was not optimized successfully under this budget. This does not establish impossibility.

### Diffuse control

The fixed diffuse condition is diagnostic. Strong diffuse performance would weaken any interpretation that resolved variable identity matters in this benchmark.

## Claim boundary

Even a strong X8 result would not be autonomous ontology discovery.

Still supplied are:

- four external variables;
- eight candidate internal slots;
- categorical value domain and `EMPTY` symbol;
- row-normalized register-to-slot transport;
- command, argument and destination identities;
- explicit recurrent state;
- shared local transition architecture.

A later experiment would need to relax external-variable cardinality, candidate-slot existence, or the row-normalized binding interface itself.