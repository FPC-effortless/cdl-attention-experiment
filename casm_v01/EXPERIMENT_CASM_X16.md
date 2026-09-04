# CASM-X16 — Persistent primal-dual capacity allocation

## Status and purpose

CASM-X16 is preregistered before implementation or execution.

X15 showed that a parameter-free instantaneous remaining-capacity response materially improves allocation robustness over a matched neutral scorer, but is not sufficient for robust cardinality extrapolation. Two of three X15 capacity-conserving seeds were exactly competent on trained `n=2,3,4`, while one failed at trained `n=4`. No seed passed the frozen unseen `n=5,6` criteria. The strongest X15 seed reached a correct 5/5 hard allocation at unseen `n=5` but did not preserve the required soft-binding competence and then collided at `n=6`.

The next missing ingredient may be persistent global constraint state rather than another instantaneous occupancy transform. X16 therefore tests a fixed entropic primal-dual allocator with nonnegative per-slot prices carried across refinement rounds.

This is a supplied resource-conservation prior. Success would not constitute spontaneous ontology discovery.

## Question

> Is persistent global dual-price state sufficient to turn a learned answer-supervised variable→slot preference function into a robust cardinality-generalizing allocation procedure when instantaneous capacity feedback is insufficient?

## Frozen learned preference model

Retain the X10/X13 independent relational variable↔slot scorer unchanged.

- deterministic external-variable descriptors unchanged;
- deterministic candidate-slot descriptors unchanged;
- no learned external-ID, active-cardinality-ID, or slot-ID embedding/table;
- one shared learned scorer produces `base_logits ∈ R^{n×8}`;
- exactly eight candidate slots.

The positive-control executor remains the validated X9R2 slot-identity-invariant local executor.

## Frozen primal-dual allocator

For `n` active variables and eight slots, let learned base logits be `L ∈ R^{n×8}`.

Initialize a global nonnegative slot-price vector:

`lambda^0 = zeros(8)`.

Freeze **8 dual rounds** and dual step size **eta = 1.0**.

At each round `t = 0..7`:

1. primal binding:

   `P^t = softmax(L - lambda^t[None, :], dim=slot)`

2. total slot occupancy:

   `c^t_s = sum_i P^t_{i,s}`

3. projected dual update for unit slot capacity:

   `lambda^{t+1}_s = relu(lambda^t_s + eta * (c^t_s - 1.0))`

The final binding is:

`B = softmax(L - lambda^8[None, :], dim=slot)`.

The same global price for a slot is visible to every variable through the subtraction from its slot logit. Prices are generated only from the model's current soft occupancy. They are not learned parameters and receive no oracle signal.

### Why this is stronger than X15

X15 responds myopically to current other-row occupancy and does not carry global constraint state between rounds. X16 carries a persistent projected dual variable for each slot. Repeated overload raises a slot's price across rounds; under-utilization can reduce an existing price through the projected update.

### Important non-guarantee

This is **not** an injective projection or matching algorithm. If all rows have exactly identical logits and all slots are symmetric, the price dynamics remain symmetric and cannot invent a row-specific assignment. Hard collisions must remain observable in that exact-symmetry falsifier.

## Regimes

Three regimes receive identical task batches and optimizer schedules:

1. `canonical_functional` — deterministic positive control.
2. `dual_neutral` — the learned independent relational scorer with all slot prices fixed to zero; equivalent to a one-shot binding after repeated neutral rounds.
3. `dual_priced` — parameter-identical scorer with the frozen 8-round projected dual-price dynamics above.

`dual_neutral` and `dual_priced` must begin bit-identically, have exactly equal learned parameter counts, and receive identical batches. The only treatment difference is the parameter-free price dynamics.

## Frozen structural objective

Retain the X13 saturation-resistant structural objective on the **final binding**:

`R_spread(B) = mean_i H(B_i) / log(8)`

with `epsilon = 1e-3`:

`R_barrier(B) = mean_{i<j} -log(1 - (1-epsilon) * dot(B_i,B_j))`.

All learned regimes use:

`L_total = L_answer + R_spread + R_barrier`

with coefficients `1.0` and `1.0`.

The dual prices are not separately supervised and no capacity penalty is added to the loss. This isolates persistent price-state computation from another loss-term intervention.

## Frozen data / executor / optimization

Retain X13/X15 exactly:

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
- X9R2 slot-identity-invariant local executor;
- AdamW, weight decay `1e-4`;
- cosine LR `2e-3 -> 2e-4`, no warmup;
- global gradient clipping `1.0`.

There is **no train-time forward or loss call at `n=5,6`**. Evaluation of `n=5,6` starts only after the 10,000 optimization steps are complete.

Evaluation remains separate for `n=2..6` at depths `8,12,24,48,96`, `eval_n=256`, using both final soft binding and unrepaired independent-row-argmax hard binding.

## New seeds

- train `20261111`, eval `20261191`;
- train `20261112`, eval `20261192`;
- train `20261113`, eval `20261193`.

No seed may be replaced, omitted, or rerun selectively based on outcome.

## Integrity requirements

Before training, tests must establish:

1. inherited X9R2 slot-permutation equivalence;
2. unchanged X13 external/slot descriptors and independent relational scorer;
3. no learned external-ID, cardinality-ID or slot-ID embedding/table;
4. `dual_neutral` and `dual_priced` learned parameters are bit-identical before optimization;
5. `dual_neutral` and `dual_priced` have exactly equal total/trainable learned parameter counts;
6. `lambda^0` is exactly zero;
7. exactly eight dual rounds are executed;
8. `eta` is exactly `1.0`;
9. primal rows are valid categorical distributions every round;
10. each occupancy equals the sum of current primal probabilities across rows;
11. each dual update exactly matches `relu(lambda + occupancy - 1)`;
12. every price remains finite and nonnegative;
13. an overloaded slot raises its price when not already offset by projection;
14. an underloaded slot can reduce a positive price but never below zero;
15. `dual_neutral` keeps prices exactly zero and returns the same binding as direct softmax of base logits;
16. the complete priced allocator is equivariant to arbitrary external-row permutation;
17. the complete priced allocator is equivariant to candidate-slot permutation when slot descriptors/logit columns are permuted consistently;
18. changing one row's logits can affect another row's final binding in `dual_priced` through shared prices;
19. the same intervention cannot affect another row in `dual_neutral`;
20. exact all-row/all-slot symmetric logits remain symmetric under priced dynamics and hard argmax collisions remain observable;
21. no hard injectivity, Hungarian/matching, Sinkhorn, sorting assignment, hard mask, collision repair, target assignment, binding labels, teacher forcing, or hidden-state targets are used;
22. X13 row-spread/barrier formulas and coefficients are unchanged;
23. gradients from total loss reach the learned scorer and executor through the differentiable price dynamics;
24. changing hidden/intermediate targets or semantic labels while preserving final register-0 target leaves all losses unchanged;
25. changing final register-0 target can change answer/total loss;
26. all regimes receive identical batches and optimizer schedule;
27. training schedule is exactly `2,3,4` repeated;
28. no train-time model forward, structural objective, price diagnostic, or binding diagnostic is evaluated at `n=5,6`;
29. training aborts on non-finite/negative answer loss, non-finite total loss, non-finite binding, or non-finite/negative dual price.

## Frozen interpretation

### Positive-control prerequisite

`canonical_functional` must satisfy every seed and every `n=2..6`:

- hard answer-final >=99% on every suite;
- hard deep step-state exactness >=95% at depths 24/48/96;
- hard deep hidden-register accuracy >=99% at depths 24/48/96.

Failure invalidates X16.

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

If strong fails but every unseen cell has answer-final >=90% and deep step-state/hidden-register >=80%, classify partial and report the boundary.

### Dual-price effect

A causal dual-price extrapolation claim is eligible only if **both `dual_neutral` and `dual_priced` satisfy seen competence on every seed**.

If eligible:

- if `dual_priced` passes strong unseen on every seed and `dual_neutral` does not, persistent dual-price state is supported as the decisive treatment under this allocator;
- if both pass strongly, dual prices are not necessary under the shared scorer;
- if both fail unseen, eight-round unit-capacity dual pricing is insufficient;
- if `dual_priced` is partial and `dual_neutral` fails partial, report only a partial improvement;
- if neutral fails seen competence, the causal extrapolation comparison is ineligible even if priced succeeds;
- if priced fails seen competence, classify priced optimization failure rather than evidence against persistent dual state in general.

### Treatment robustness relative to X15

Independently of causal extrapolation eligibility, report whether `dual_priced` improves every-seed **seen competence count** relative to X15's 2/3 capacity-conserving result. This is descriptive across different seed sets and must not be presented as a paired causal comparison.

## Successor boundary

Only a strong every-seed unseen PASS authorizes removing supplied active cardinality or external-variable identity. Otherwise research remains inside allocation learning.
