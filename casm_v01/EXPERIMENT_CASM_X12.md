# CASM-X12 — Soft scarcity-economy binding

## Status and purpose

CASM-X12 is preregistered before implementation or execution.

X11 showed that pairwise binding-overlap competition is not robust: it sometimes repairs relational seen-cardinality optimization, but neither independent nor coordinated competitive binding is competent on every seed, and no learned regime generalizes strongly to unseen `n=5,6`.

X12 retains the validated X9R2 local executor, X10 relational generators, answer-only supervision, mixed-cardinality data, optimizer schedule, and unrepaired hard evaluation. It changes only the soft structural objective.

## Question

> Is a decomposed scarcity objective—one cost for spreading a variable across many slots and another for exceeding per-slot capacity—sufficient to produce robust collision-free relational allocation and unseen-cardinality generalization under answer-only supervision? If it succeeds, is cross-variable coordination still required?

## Frozen structural prior

For row-stochastic binding `B ∈ R^{n×8}`:

### Row spread cost

`R_spread(B) = mean_i H(B_i) / log(8)`

where `H(p) = -sum_s p_s log p_s`.

- uniform row: `R_spread = 1`;
- one-hot row: contribution `0`.

### Slot capacity overflow

Let `c_s = sum_i B[i,s]`.

`R_capacity(B) = (1/n) * sum_s relu(c_s - 1)^2`

- occupancy <=1 incurs no capacity cost;
- overloaded slots incur quadratic cost;
- distinct one-hot allocation has `R_capacity = 0`;
- all `n` rows on one slot gives `(n-1)^2/n`.

### Scarcity loss

`L_scarcity = L_answer + λ_spread R_spread + λ_capacity R_capacity`

with frozen:

- `λ_spread = 1.0`;
- `λ_capacity = 1.0`.

The scarcity terms read only the model-generated binding. They receive no target binding, hidden state, semantic label, final answer value, matching solution, or oracle topology.

This is an explicit soft resource prior. Success would not constitute spontaneous ontology discovery.

## Regimes

Five regimes train on identical batches and optimizer schedules:

1. `canonical_functional` — deterministic positive control.
2. `relational_independent_overlap` — exact X11 independent overlap objective (`λ_overlap=1.0`) on new seeds.
3. `relational_coordinated_overlap` — exact X11 coordinated overlap objective on new seeds.
4. `relational_independent_scarcity` — parameter-identical independent relational scorer with scarcity objective only.
5. `relational_coordinated_scarcity` — parameter-identical coordinated relational scorer with scarcity objective only.

For each architecture, overlap and scarcity variants begin bit-identically and receive identical task batches. Independent-scarcity and coordinated-scarcity relational generators also begin from bit-identical parameter states, preserving the coordination comparison.

No regime uses hard injectivity, Hungarian/matching, Sinkhorn projection, collision repair, binding labels, or intermediate-state supervision.

## Frozen data / executor / optimization

Exactly retain X11/X9R2:

- train cardinalities `n=2,3,4` only;
- deterministic `2,3,4` repeated schedule;
- train depth 8;
- batch size 128;
- 10,000 optimizer steps;
- fixed final external register 0 is the only task target;
- no teacher forcing;
- no intermediate/hidden targets;
- no semantic operator labels;
- eight candidate slots;
- X9R2 slot-identity-invariant local executor;
- AdamW, weight decay `1e-4`;
- cosine LR `2e-3 -> 2e-4`, no warmup;
- global grad clipping `1.0`.

Evaluation remains separate for `n=2..6` at depths `8,12,24,48,96`, `eval_n=256`, with both soft binding and unrepaired independent-row-argmax hard binding.

## New seeds

- train `20261071`, eval `20261151`;
- train `20261072`, eval `20261152`;
- train `20261073`, eval `20261153`.

No seed may be replaced or omitted based on outcome.

## Integrity requirements

Before training, tests must establish:

1. inherited X9R2 slot-permutation equivalence;
2. inherited X10 descriptor and relational permutation-equivariance contracts;
3. no learned external-ID or slot-ID embedding/table;
4. overlap/scarcity paired models are bit-identical before optimization;
5. independent/coordinated scarcity parameter counts match;
6. `R_spread=1` for uniform rows within tolerance;
7. `R_spread=0` for one-hot rows within tolerance;
8. `R_capacity=0` for distinct one-hot rows;
9. `R_capacity=(n-1)^2/n` when all `n` rows choose one slot;
10. both scarcity terms are invariant to row permutation and slot-column permutation;
11. scarcity gradients reach binding-generator parameters for a non-stationary near-uniform binding;
12. scarcity terms read only generated binding;
13. target-leakage invariance from X11 remains intact;
14. changing final register-0 target can change answer/total loss;
15. no hard injectivity, matching, Sinkhorn, collision repair or target assignment is used;
16. hard collisions remain observable under independent row argmax;
17. all regimes receive identical batches and optimizer schedule;
18. training schedule is exactly `2,3,4` repeated;
19. training aborts on non-finite/negative answer loss or non-finite total loss.

## Frozen interpretation

### Positive-control prerequisite

`canonical_functional` must satisfy every seed and every `n=2..6`:

- hard answer-final >=99% on every suite;
- hard deep step-state exactness >=95% at depths 24/48/96;
- hard deep hidden-register accuracy >=99% at depths 24/48/96.

Failure invalidates X12.

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

### Scarcity-vs-overlap effect

For an architecture, the scarcity objective is supported as a robustness improvement only if:

- scarcity satisfies seen competence on every seed; and
- paired overlap fails seen competence on at least one same seed.

It is supported as an extrapolation improvement only if scarcity also strictly improves the preregistered unseen classification over paired overlap without sacrificing seen competence.

### Coordination under scarcity

- if independent scarcity passes strongly, scorer-level cross-variable coordination is unnecessary under this resource objective;
- if independent scarcity is seen-competent but fails unseen while coordinated scarcity passes strongly, scorer-level coordination is required under the tested scarcity representation;
- if both pass strongly, coordination is unnecessary;
- if both are seen-competent but fail unseen, the remaining limitation is extrapolation despite soft scarcity;
- if either scarcity regime fails seen competence, no coordination claim is eligible.

## Successor boundary

Only a strong unseen-cardinality PASS authorizes removing supplied active cardinality or external-variable identity. Otherwise work remains inside binding-allocation learning.
