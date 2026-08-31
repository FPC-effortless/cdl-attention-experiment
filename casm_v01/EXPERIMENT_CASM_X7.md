# CASM-X7 — Surplus-slot sparse binding from a fixed answer channel

## Motivation

CASM-X6R established that, inside a supplied four-variable / four-slot explicit-state ontology, fixed-answer-only supervision can learn a sharp one-to-one external-register ↔ internal-slot binding and preserve essentially exact hidden execution through depth 96.

X7 weakens one remaining structural prior: the model is no longer told that there are exactly four useful internal slots.

It receives **eight candidate internal slots for a four-variable world** and must learn which distinct subset of four slots carries the external computational state.

This is not full ontology discovery. The experiment still supplies that there are four external variables and that each external variable should bind injectively to one internal candidate slot.

## Question

> Can fixed-answer-only supervision identify a sparse injective 4→8 register-to-slot assignment and preserve hidden executable state when half of the candidate internal slots are surplus?

Internal slot labels remain gauge symmetric. Success is therefore defined by functional execution plus a sharp injective assignment, not correspondence to canonical slot numbers.

## State contract

The external world is unchanged from X6R:

- four external registers;
- values `0..15`;
- contextual command semantics;
- explicit recurrent state;
- train depth 8;
- final external register `0` is the only target label;
- external registers `1–3` are never target labels;
- no intermediate-state supervision;
- no teacher forcing;
- no semantic-operator labels.

The internal state now has eight candidate slots.

A seventeenth categorical symbol, `EMPTY`, is used only to make surplus-slot probability mass explicit and normalized. Under a hard injective assignment, four selected slots contain the four register values and four surplus slots remain `EMPTY`.

The transition predicts only world values `0..15`; it never writes `EMPTY` as an operator result.

## Binding family

The learned binding is a 4×8 matrix.

X7 supplies the **injective-assignment prior** but not the assignment itself:

1. enumerate all ordered injective mappings of four external registers into eight internal slots (`8P4 = 1680` assignments);
2. score each assignment from a trainable 4×8 score matrix;
3. softmax the 1680 assignment scores;
4. form the binding as their convex weighted sum.

Therefore:

- every external-register row sums exactly to 1;
- every candidate-slot column sum is between 0 and 1;
- total assignment mass is exactly 4;
- no two external registers share a slot under the projected discrete assignment.

This is differentiable and probability preserving. The model receives no assignment labels, canonical-slot prior, sparsity penalty, entropy penalty, identity target or permutation target.

## Probability transport with surplus slots

Let `B[e,s]` be external-register→candidate-slot assignment mass and `o[s] = sum_e B[e,s]` be slot occupancy.

Initial internal categorical state is:

`p_s = sum_e B[e,s] onehot(value_e) + (1 - o[s]) onehot(EMPTY)`.

Because column occupancy is at most 1, every internal slot is a valid categorical distribution.

External decoding for register `e` is:

`q_e = sum_s B[e,s] p_s`.

Because every row of `B` sums to 1, each decoded register remains a valid categorical distribution.

All external-register-specific lookup, destination update and positional representation passes through the same binding.

## Regimes

Three parameter-matched regimes start from identical transition parameters.

### `canonical_sparse`

Positive control. Fixed injective mapping:

- register 0 → slot 0
- register 1 → slot 1
- register 2 → slot 2
- register 3 → slot 3

Slots 4–7 are surplus.

### `learned_sparse`

Trainable near-uniform 4×8 assignment scores projected through the exact 1680-assignment convex mixture.

### `diffuse_surplus`

Negative control. Every register spreads uniformly over all eight slots (`1/8` per slot). Each slot therefore has occupancy `1/2`; the remaining `1/2` categorical mass is `EMPTY`.

This deliberately prevents resolved variable identity.

## Budget

Preregistered before implementation/execution:

- 8,000 optimization steps;
- batch size 128;
- train depth 8;
- AdamW learning rate `2e-3`;
- weight decay `1e-4`;
- assignment temperature `1.0`;
- three independent seeds:
  - train `20261001`, eval `20261081`
  - train `20261002`, eval `20261082`
  - train `20261003`, eval `20261083`
- evaluation suites:
  - IID depth 8;
  - held-out composition depth 12;
  - held-out composition depth 24;
  - stress depth 48;
  - stress depth 96.

The larger 8,000-step budget is fixed in advance because the assignment search space grows from 24 permutations in X6R to 1680 injective assignments in X7.

## Integrity requirements

Before training, tests must establish:

1. learned assignment initializes near uniform (`~1/8` row maximum);
2. every learned row sums to 1 within `1e-6`;
3. every learned column occupancy lies in `[0,1]` within tolerance;
4. total assignment mass is 4;
5. these constraints hold under adversarially sharp/conflicting score matrices;
6. the projected assignment contains four unique slot IDs;
7. initial internal slot distributions sum to 1, including surplus `EMPTY` mass;
8. decoded external distributions sum to 1;
9. the probability contract survives optimizer updates;
10. fixed-answer categorical loss is finite and non-negative;
11. loss ignores intermediate targets, hidden final targets and semantic labels;
12. gradients reach both assignment scores and transition parameters;
13. no direct external-register embedding bypass exists;
14. all three regimes produce valid depth-96 rollout shapes.

The training runner must abort immediately on any non-finite or negative categorical loss.

## Preregistered interpretation

### Positive-control prerequisite

On **every seed**, `canonical_sparse` must achieve:

- answer-final accuracy ≥99% on every suite;
- full step-state exactness ≥95% at depths 24, 48 and 96;
- hidden-register accuracy ≥99% at depths 24, 48 and 96.

If this fails, no X7 assignment claim is made.

### Strong sparse injective binding

If the positive control passes, `learned_sparse` supports the strong result only if **every seed** satisfies:

- answer-final accuracy ≥99% on every suite;
- full step-state exactness ≥95% at depths 24, 48 and 96;
- hidden-register accuracy ≥99% at depths 24, 48 and 96;
- mean row maximum ≥0.90;
- best injective-assignment score ≥3.60 / 4.00;
- projected assignment uses four distinct candidate slots.

No cross-seed averaging may rescue a failed seed.

### Functional execution without sparse selection

If execution thresholds pass but row sharpness or assignment score fails, the result supports distributed surplus-slot computation rather than sparse variable selection.

### Answer-only shortcut

If answer-final accuracy reaches ≥95% but step-state exactness or hidden-register accuracy falls below 80% at depth 24 or deeper, the model found an answer strategy without recovering the hidden executable state.

### Assignment optimization failure

If `canonical_sparse` passes but `learned_sparse` IID answer-final accuracy remains below 80%, the setup failed to optimize the larger assignment problem. This does not establish impossibility.

### Diffuse control

The diffuse condition is diagnostic rather than a formal prerequisite. Strong performance from `diffuse_surplus` would weaken the claim that resolving external-variable identity to distinct internal slots is necessary in this benchmark.

## Claim boundary

Even a strong X7 result would establish **surplus-slot selection within a supplied injective binding family**, not autonomous ontology discovery.

Still supplied are:

- four external variables;
- the existence of candidate slots;
- the maximum candidate count of eight;
- categorical world value domain;
- `EMPTY` as an architectural surplus-slot symbol;
- the injective-assignment family;
- command, argument and destination identities;
- explicit recurrent state;
- shared local transition architecture.

A later experiment would need to relax the injective binding family or external-variable cardinality itself.