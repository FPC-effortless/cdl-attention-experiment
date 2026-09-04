# CASM-X18 — Global-coordinate preference representation

## Status and purpose

CASM-X18 is preregistered before implementation or execution.

X17 established that both 8-round and 64-round projected dual-price allocators are fully competent on trained `n=2,3,4` across all three new seeds, yet both fail strong and partial unseen `n=5,6` generalization. Applying 64 inference rounds to the already-trained 8-round weights also fails. The remaining bottleneck therefore moves from dual convergence horizon toward the learned variable→slot preference representation.

The current external-variable descriptor changes its coordinate frame with active cardinality `n`: it includes `e/(n-1)`, `n/6`, and Fourier phases scaled by `e/n`. A variable at external index `e` is therefore represented differently as `n` changes. CASM-X18 tests whether that cardinality-relative frame shift is the missing factor.

This experiment still supplies active cardinality, external-variable indices, a fixed eight-slot workspace, deterministic descriptors, the local executor, and the dual allocation law. Success would not constitute ontology discovery or variable discovery.

## Question

> Is a fixed global external-coordinate frame sufficient to turn the validated answer-supervised preference scorer plus 8-round dual-price allocator into a cardinality-generalizing allocation system, when the cardinality-relative descriptor fails?

## Frozen regimes

Three regimes receive identical task batches and optimizer schedules:

1. `canonical_functional` — deterministic positive control.
2. `relative_descriptor` — exact X16/X17 learned scorer and exact legacy cardinality-relative external descriptor.
3. `global_descriptor` — parameter-identical learned scorer and allocator, differing only in the deterministic external descriptor defined below.

`relative_descriptor` and `global_descriptor` must begin bit-identically and have exactly equal total/trainable learned parameter counts.

## Frozen relative descriptor control

For external index `e` and active cardinality `n`, retain the exact existing 9-dimensional descriptor:

- `e / max(n-1,1)`;
- `n / 6`;
- `sin(pi * e / n)`;
- `cos(pi * e / n)`;
- `sin(2*pi * e / n)`;
- `cos(2*pi * e / n)`;
- the same three deterministic binary bits of `e` in `{-1,+1}`.

The implementation must test exact equality against the existing canonical `variable_descriptor` function.

## Frozen global descriptor treatment

Use the same external index `e` and the already-supplied fixed workspace size of eight candidate slots, but place every variable in one global coordinate frame independent of active cardinality `n`.

The 9-dimensional descriptor is:

- `e / 7`;
- `8 / 8 = 1` as the fixed normalized workspace-size coordinate;
- `sin(pi * e / 8)`;
- `cos(pi * e / 8)`;
- `sin(2*pi * e / 8)`;
- `cos(2*pi * e / 8)`;
- the same three deterministic binary bits of `e` in `{-1,+1}`.

For a fixed external index `e`, this descriptor must be bit-identical for every active cardinality in which `e` exists. It contains no target value, hidden state, correct slot assignment, future command, semantic operator label, or learned external-ID embedding.

This is a supplied global-coordinate prior. It uses the known maximum candidate workspace size of eight and known external index; it does not infer either quantity.

## Frozen scorer / allocator / executor

Retain X16's learned independent relational variable↔slot scorer unchanged except for which external descriptor function is supplied.

Retain X16's exact 8-round projected dual-price allocator:

- `lambda^0 = zeros(8)`;
- for exactly 8 rounds:
  - `P = softmax(L - lambda)`;
  - `c = sum_rows(P)`;
  - `lambda = relu(lambda + (c - 1))`;
- final binding `B = softmax(L - lambda^8)`.

Freeze `eta = 1.0` and unit soft capacity.

Retain the X9R2 slot-identity-invariant local executor unchanged.

No hard injectivity, Hungarian/matching, Sinkhorn, sorting assignment, hard mask, collision repair, target assignment, binding labels, or learned price network is permitted.

## Frozen objective

Retain the X13 final-binding structural objective unchanged:

- normalized row entropy/spread coefficient `1.0`;
- saturation-resistant pairwise collision barrier coefficient `1.0` with epsilon `1e-3`;
- final-register-0 answer loss.

No descriptor-specific regularizer or unseen-cardinality loss is added.

## Frozen data / optimization

Retain X17 exactly:

- task-supervised train cardinalities `n ∈ {2,3,4}` only;
- deterministic `2,3,4` repeated schedule;
- train depth `8`;
- batch size `128`;
- `10,000` optimizer steps;
- fixed final external register `0` is the only task target;
- no teacher forcing;
- no intermediate/hidden targets;
- no semantic operator labels;
- eight candidate slots;
- AdamW, weight decay `1e-4`;
- cosine LR `2e-3 -> 2e-4`, no warmup;
- global gradient clipping `1.0`.

There is **no train-time model forward, binding diagnostic, descriptor comparison, structural objective, or loss call at `n=5,6`**. Unseen evaluation begins only after all 10,000 optimizer steps.

Evaluation remains separate at `n=2..6` and depths `8,12,24,48,96`, `eval_n=256`, for both final soft binding and unrepaired independent-row-argmax hard binding.

## New seeds

- train `20261131`, eval `20261211`;
- train `20261132`, eval `20261212`;
- train `20261133`, eval `20261213`.

No seed may be replaced, omitted, or selectively rerun based on outcome.

## Integrity requirements

Before training, tests must establish:

1. inherited X9R2 executor slot-permutation equivalence;
2. exact X16 8-round dual dynamics and `eta=1.0`;
3. `relative_descriptor` exactly matches the existing canonical `variable_descriptor` for all allowed `e,n`;
4. `global_descriptor` has exactly nine finite coordinates;
5. for each fixed `e`, `global_descriptor(e,n)` is exactly invariant to `n` wherever `e<n`;
6. the global descriptor uses only `e`, the fixed workspace size eight, and deterministic arithmetic;
7. global descriptors for external indices `0..5` are distinct;
8. the slot descriptor is unchanged from X10-X17;
9. the learned scorer architecture is identical between relative/global treatments;
10. both learned treatments start bit-identically and have equal parameter counts;
11. changing only the descriptor frame can change learned logits, but no other treatment path differs;
12. no learned external-ID, cardinality-ID, or slot-ID embedding/table is introduced;
13. complete binding remains equivariant to external-row permutation when descriptor rows are permuted consistently;
14. complete binding remains equivariant to candidate-slot permutation when slot descriptors/logit columns are permuted consistently;
15. exact symmetric logits remain symmetric under the dual allocator and hard collisions remain unrepaired;
16. primal rows remain valid categorical distributions and prices remain finite/nonnegative;
17. structural gradients reach the shared scorer and executor for both learned regimes;
18. changing hidden/intermediate targets or semantic labels while preserving final register-0 target leaves all losses unchanged;
19. changing final register-0 target can change answer/total loss;
20. all regimes receive identical batches, cardinality schedule, optimizer, LR schedule, and gradient clipping;
21. training schedule is exactly `2,3,4` repeated;
22. no train-time forward or diagnostic at `n=5,6` occurs;
23. hard evaluation is independent row argmax with no collision repair;
24. training aborts on non-finite/negative answer loss, total loss, binding values, or prices.

## Frozen classification

### Positive-control prerequisite

`canonical_functional` must satisfy every seed and every `n=2..6`:

- hard answer-final >=99% on every suite;
- hard deep step-state exactness >=95% at depths 24/48/96;
- hard deep hidden-register accuracy >=99% at depths 24/48/96.

Failure invalidates X18.

### Seen-cardinality competence

A learned regime is unseen-eligible only if every seed at `n=2,3,4` has:

- hard and soft answer-final >=99% on every suite;
- hard and soft deep step-state exactness >=95%;
- hard and soft deep hidden-register accuracy >=99%;
- exactly `n` unique hard-argmax slots;
- zero hard collisions;
- mean final soft row maximum >=0.90.

IID depth-8 answer-final <80% on any seen cardinality is optimization failure.

### Strong unseen-cardinality generalization

A learned regime passes strongly only if every seed at both `n=5,6` has:

- hard and soft answer-final >=99% on every suite;
- hard and soft deep step-state exactness >=95%;
- hard and soft deep hidden-register accuracy >=99%;
- exactly `n` unique hard-argmax slots;
- zero collisions;
- mean final soft row maximum >=0.90.

No averaging rescues a failed cell.

### Partial unseen generalization

If strong fails but every unseen hard and soft cell has answer-final >=90% and deep step-state/hidden-register >=80%, classify partial and report the exact boundary.

### Descriptor-frame effect

A causal descriptor-frame extrapolation comparison is eligible only if **both `relative_descriptor` and `global_descriptor` satisfy seen competence on every seed**.

If eligible:

- if global passes strong unseen and relative does not, support the fixed global coordinate frame as the decisive treatment under this supplied workspace;
- if both pass strongly, the global frame is not necessary under the shared scorer/allocator;
- if both fail unseen, global coordinates are insufficient and the next experiment should move to a procedural/recursive role generator;
- if global is partial and relative fails partial, report only partial improvement;
- if either treatment fails seen competence, classify the paired extrapolation comparison as ineligible and diagnose optimization before making a descriptor-frame claim.

## Claim boundary

Even a strong global-descriptor PASS would show only that a supplied, cardinality-invariant coordinate system enables extrapolative allocation in this controlled benchmark. It would not show discovery of variables, discovery of cardinality, discovery of slots, spontaneous ontology formation, or creation of a workspace.

## Successor boundary

Only a strong every-seed unseen PASS authorizes attempting to remove supplied active cardinality or external-variable identity. If both descriptor frames fail unseen while remaining seen-competent, the next preregistered step should test a shared recursive/procedural external-role generator rather than another allocator or penalty modification.
