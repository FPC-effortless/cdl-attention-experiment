# CASM-X15 — Soft capacity-conservation allocation

## Status and purpose

CASM-X15 is preregistered before implementation or execution.

X14 showed that exposing generated slot occupancy to a learned 8-round refiner is capable but not robust: two seeds solve trained `n=2,3,4`, one seed collapses at seen `n=3,4`, and only one seed generalizes exactly to unseen `n=5` before failing at `n=6`.

X15 tests a narrower hypothesis: the missing ingredient may be a stable **resource-response law**, not occupancy information itself. Instead of learning how to respond to occupancy, X15 supplies a fixed differentiable remaining-capacity rule while leaving the variable→slot preferences learned from answer-only supervision.

## Question

> Is an explicit soft unit-capacity response law sufficient to make answer-trained relational binding robustly collision-free and executable at unseen cardinalities, without hard matching or collision repair?

Success is evidence for a supplied resource-conservation prior, not spontaneous ontology/topology discovery.

## Frozen binding computation

Use the unchanged X10/X13 independent relational base scorer to produce base logits `L^0 ∈ R^{n×8}` from the deterministic external-variable and slot descriptors.

Initialize:

`P^0 = softmax(L^0, dim=slot)`.

Freeze `epsilon = 1e-3`, damping `alpha = 0.5`, and **8 capacity rounds**.

For each round `t` and focal row `i`, define other-row occupancy:

`O^t_{i,s} = sum_{j != i} P^t_{j,s}`.

Define remaining soft capacity:

`A^t_{i,s} = clamp(1 - O^t_{i,s}, min=epsilon, max=1)`.

Define the capacity-aware proposal:

`Q^t_i = softmax(L^0_i + log(A^t_i), dim=slot)`.

Damped update:

`P^{t+1}_i = (1-alpha) P^t_i + alpha Q^t_i`.

Because both terms are row-normalized, each updated row remains normalized. This update does **not** guarantee column capacity or injectivity; collisions can remain and hard evaluation must expose them.

The final binding is `P^8`.

## Regimes

Three regimes receive identical task batches and optimizer schedules:

1. `canonical_functional` — deterministic positive control.
2. `capacity_neutral` — the same learned independent relational base scorer with capacity disabled (`A=1`), so the final binding is the ordinary one-shot row softmax.
3. `capacity_conserving` — parameter-identical base scorer with the frozen 8-round remaining-capacity update above.

`capacity_neutral` and `capacity_conserving` begin bit-identically, have exactly equal trainable/total parameter counts, and differ only by the parameter-free capacity transform.

All learned regimes retain the X13 structural objective:

`L = L_answer + R_spread + R_barrier`

where:

`R_spread(B) = mean_i H(B_i) / log(8)`

and, with barrier epsilon `1e-3`:

`R_barrier(B) = mean_{i<j} -log(1 - 0.999 * dot(B_i,B_j))`.

No hard injectivity, matching/Hungarian, Sinkhorn projection, sorting assignment, hard masking, collision repair, binding labels, teacher forcing, intermediate-state supervision or hidden-state targets are used.

## Frozen data / executor / optimization

Retain X14/X13/X9R2:

- train cardinalities `n ∈ {2,3,4}` only;
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

There is no train-time forward or loss at `n=5,6`.

Evaluation remains separate at `n=2..6`, depths `8,12,24,48,96`, `eval_n=256`, using both soft binding and independent-row-argmax hard binding with collisions unrepaired.

## New seeds

- train `20261101`, eval `20261181`;
- train `20261102`, eval `20261182`;
- train `20261103`, eval `20261183`.

No seed may be replaced or omitted based on outcome.

## Integrity requirements

Before training, tests must establish:

1. inherited X9R2 slot-permutation equivalence;
2. unchanged X10/X13 external and slot descriptor definitions;
3. no learned external-ID, cardinality-ID or slot-ID table/embedding;
4. neutral/conserving models are bit-identical before optimization;
5. equal total/trainable parameter counts;
6. exactly 8 capacity rounds, `epsilon=1e-3`, `alpha=0.5`;
7. generated `O` equals sum of other rows and excludes the focal row;
8. `A=1` when other-row occupancy is zero;
9. `A=epsilon` when other-row occupancy is at least one;
10. the conserving update preserves row normalization and nonnegative probabilities;
11. the neutral transform leaves ordinary base row-softmax unchanged exactly within tolerance;
12. another row's probability can affect a focal row only in conserving mode;
13. complete conserving binding is external-row permutation-equivariant;
14. complete conserving binding is slot-column permutation-equivariant when slot descriptors are permuted consistently;
15. structural gradients reach the learned base scorer through the conserving transform;
16. structural computation reads only generated binding/descriptor state;
17. hidden/intermediate target and semantic-label changes preserving final register-0 target leave losses unchanged;
18. changing final register-0 target can change answer/total loss;
19. no matching, hard injectivity, Sinkhorn, sorting assignment, hard masking or collision repair exists;
20. hard collisions remain observable;
21. all regimes receive identical batches and optimizer schedule;
22. training cardinality schedule is exactly `2,3,4` repeated;
23. no train-time forward/loss occurs at `n=5,6`;
24. training aborts on non-finite/negative answer or non-finite total loss.

## Frozen interpretation

### Positive control prerequisite

`canonical_functional` must satisfy every seed and every `n=2..6`:

- hard answer-final >=99% on every suite;
- hard deep step-state exactness >=95% at depths 24/48/96;
- hard deep hidden-register accuracy >=99% at depths 24/48/96.

Failure invalidates X15.

### Seen competence

A learned regime is unseen-eligible only if every seed at `n=2,3,4` has:

- hard and soft answer-final >=99% on every suite;
- hard and soft deep step-state exactness >=95%;
- hard and soft deep hidden-register accuracy >=99%;
- exactly `n` unique hard-argmax slots;
- zero collisions;
- mean row maximum >=0.90.

IID depth-8 answer-final <80% on any seen cardinality is optimization failure.

### Strong unseen generalization

A learned regime passes strongly only if every seed at both `n=5,6` has:

- hard and soft answer-final >=99% on every suite;
- hard and soft deep step-state exactness >=95%;
- hard and soft deep hidden-register accuracy >=99%;
- exactly `n` unique hard-argmax slots;
- zero collisions;
- mean row maximum >=0.90.

No averaging rescues a failed cell.

### Partial unseen generalization

If strong fails but every unseen cell has answer-final >=90% and deep step-state/hidden-register >=80%, classify partial.

### Capacity-law effect

- If `capacity_conserving` is seen-competent on every seed and passes strong unseen while `capacity_neutral` does not, support the narrow claim that the supplied remaining-capacity law enables robust allocation under this representation.
- If both pass strongly, the capacity law is unnecessary.
- If conserving is seen-competent but fails unseen, the supplied capacity law is insufficient for extrapolation.
- If conserving fails seen competence, classify capacity-treatment optimization failure rather than evidence that resource constraints are generally useless.
- If neutral fails seen competence while conserving passes strongly, support a broader robustness+generalization treatment effect, but do not describe the difference as pure extrapolation because the comparator is not seen-qualified.

## Successor boundary

Only a strong every-seed unseen PASS authorizes the next experiment to remove supplied active cardinality or external-variable identity. Otherwise work remains inside allocation learning.