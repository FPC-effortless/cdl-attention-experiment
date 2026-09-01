# CASM-X13 — Saturation-resistant collision barrier

## Status and purpose

CASM-X13 is preregistered before implementation or execution.

X12 established that the entropy-plus-quadratic-capacity scarcity objective is not a robust allocation mechanism. The canonical executor is valid, but both X12 scarcity regimes hit the preregistered seen-cardinality optimization-failure boundary on every seed. Several failures end in sharply colliding rows with positive capacity cost, despite distinct one-hot allocations being the zero-cost structural optimum.

X13 tests one specific mechanistic diagnosis: the quadratic probability-space overflow correction may become ineffective after row-softmax saturation. X13 changes only the collision-cost shape. It does not change the executor, relational generators, descriptors, supervision, cardinalities, optimizer schedule, row-compactness term, or hard evaluation.

## Question

> Does replacing X12's quadratic slot-overflow correction with a saturation-resistant logarithmic collision barrier robustly prevent seen-cardinality binding collapse and enable unseen-cardinality allocation under answer-only supervision?

If it succeeds, is scorer-level cross-variable coordination still required?

## Frozen structural objectives

For row-stochastic binding `B ∈ R^{n×8}`, retain X12 normalized row spread:

`R_spread(B) = mean_i H(B_i) / log(8)`.

### X12 quadratic comparator

Retain exactly:

`R_capacity(B) = (1/n) * sum_s relu(sum_i B[i,s] - 1)^2`.

Quadratic comparator loss:

`L_quadratic = L_answer + R_spread + R_capacity`.

### X13 collision barrier

For each external-variable pair `i<j`, define overlap:

`o_ij = sum_s B[i,s] * B[j,s]`.

Freeze `epsilon = 1e-3` and define:

`R_barrier(B) = mean_{i<j} -log(1 - (1-epsilon) * o_ij)`.

Barrier loss:

`L_barrier = L_answer + R_spread + R_barrier`.

All structural coefficients are frozen at `1.0`.

Properties:

- distinct one-hot rows: pair overlap `0`, barrier contribution exactly `0`;
- uniform rows: pair overlap `1/8`, barrier about `0.1334`;
- identical one-hot rows: pair overlap `1`, barrier `-log(1e-3) ≈ 6.9078`;
- the barrier remains finite but its derivative with respect to overlap grows as overlap approaches `1`.

The barrier reads only model-generated binding probabilities. It receives no correct assignment, target state, semantic label, answer value, matching solution or oracle topology.

This is an explicit soft anti-collision prior. Success is not spontaneous ontology discovery.

## Mechanistic gradient falsifier

Before training, construct two identical near-saturated rows from logits `[12,0,0,0,0,0,0,0]` and compute the gradient norm of the collision correction with respect to those logits.

The X13 barrier gradient norm must be at least `100x` the X12 quadratic-capacity gradient norm under this exact construction. If this integrity condition does not hold, X13 does not actually test the proposed saturation-resistance mechanism and training must not run.

This gradient comparison excludes `L_answer` and `R_spread`; it compares only `R_barrier` against `R_capacity` on the same binding.

## Regimes

Five regimes receive identical task batches and optimizer schedules:

1. `canonical_functional` — deterministic positive control.
2. `relational_independent_quadratic` — X12 independent relational scorer with `L_quadratic`.
3. `relational_coordinated_quadratic` — X12 coordinated scorer with `L_quadratic`.
4. `relational_independent_barrier` — parameter-identical independent scorer with `L_barrier`.
5. `relational_coordinated_barrier` — parameter-identical coordinated scorer with `L_barrier`.

For each architecture, quadratic and barrier variants begin bit-identically and receive identical batches. Independent-barrier and coordinated-barrier generators also begin from bit-identical parameter states, preserving the coordination comparison.

No regime uses hard injectivity, Hungarian/matching, Sinkhorn projection, collision repair, binding labels, intermediate-state supervision or hidden-state targets.

## Frozen data / executor / optimization

Exactly retain X12/X9R2:

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

Evaluation remains separate for `n=2..6` at depths `8,12,24,48,96`, `eval_n=256`, using both soft binding and unrepaired independent-row-argmax hard binding.

## New seeds

- train `20261081`, eval `20261161`;
- train `20261082`, eval `20261162`;
- train `20261083`, eval `20261163`.

No seed may be replaced or omitted based on outcome.

## Integrity requirements

Before training, tests must establish:

1. inherited X9R2 slot-permutation equivalence;
2. inherited X10 descriptor and relational permutation-equivariance contracts;
3. no learned external-ID or slot-ID embedding/table;
4. quadratic/barrier paired models are bit-identical before optimization;
5. independent/coordinated barrier parameter counts match;
6. `R_barrier=0` for distinct one-hot rows within tolerance;
7. `R_barrier≈-log(1e-3)` for identical one-hot rows;
8. uniform-row barrier matches the frozen formula at overlap `1/8`;
9. barrier is invariant to row permutation and slot-column permutation;
10. barrier gradients reach binding-generator parameters;
11. the exact near-saturated-logit stress test gives barrier collision-gradient norm at least `100x` quadratic-capacity gradient norm;
12. `R_spread` remains exactly the X12 definition;
13. structural terms read only generated binding;
14. changing hidden/intermediate targets or semantic labels while preserving final register-0 target leaves structural and total losses unchanged;
15. changing final register-0 target can change answer/total loss;
16. no hard injectivity, matching, Sinkhorn, collision repair or target assignment is used;
17. hard collisions remain observable under independent row argmax;
18. all regimes receive identical batches and optimizer schedule;
19. training cardinality schedule is exactly `2,3,4` repeated;
20. training aborts on non-finite/negative answer loss or non-finite total loss.

## Frozen interpretation

### Positive-control prerequisite

`canonical_functional` must satisfy every seed and every `n=2..6`:

- hard answer-final >=99% on every suite;
- hard deep step-state exactness >=95% at depths 24/48/96;
- hard deep hidden-register accuracy >=99% at depths 24/48/96.

Failure invalidates X13.

### Seen-cardinality competence

A learned regime is unseen-eligible only if every seed at `n=2,3,4` has:

- hard and soft answer-final >=99% on every suite;
- hard and soft deep step-state exactness >=95%;
- hard and soft deep hidden-register accuracy >=99%;
- exactly `n` unique hard-argmax slots;
- zero collisions;
- mean row maximum >=0.90.

IID depth-8 answer-final <80% on any seen cardinality is optimization failure.

### Strong unseen-cardinality generalization

A learned regime passes strongly only if every seed at both `n=5,6` has:

- hard and soft answer-final >=99% on every suite;
- hard and soft deep step-state exactness >=95%;
- hard and soft deep hidden-register accuracy >=99%;
- exactly `n` unique hard-argmax slots;
- zero collisions;
- mean row maximum >=0.90.

No averaging rescues a failed cell.

### Partial unseen generalization

If strong fails but every unseen cell has answer-final >=90% and deep step-state/hidden-register >=80%, classify partial and report the boundary.

### Saturation-resistance effect

For an architecture, the barrier mechanism is supported as a seen-optimization improvement only if:

- its barrier regime satisfies seen competence on every seed; and
- the paired quadratic regime fails seen competence on at least one same seed.

The stronger mechanistic interpretation additionally requires the preregistered `>=100x` near-collision gradient ratio to pass integrity.

If both paired regimes fail seen competence, the barrier is insufficient despite its stronger near-collision gradient.

If both satisfy seen competence, X13 cannot claim the barrier is necessary for seen optimization.

### Extrapolation effect

Barrier is supported as an extrapolation improvement only if it is seen-competent and achieves a strictly better frozen unseen classification than the paired quadratic comparator without sacrificing seen competence.

### Coordination under barrier

- if independent barrier passes strongly, scorer-level cross-variable coordination is unnecessary under this barrier prior;
- if independent barrier is seen-competent but fails unseen while coordinated barrier passes strongly, scorer-level coordination is required under the tested barrier representation;
- if both pass strongly, coordination is unnecessary;
- if both are seen-competent but fail unseen, the limitation remains extrapolation despite saturation-resistant anti-collision pressure;
- if either barrier regime fails seen competence, no coordination claim is eligible.

## Successor boundary

Only a strong unseen-cardinality PASS authorizes removing supplied active cardinality or external-variable identity. Otherwise work remains inside binding-allocation learning.