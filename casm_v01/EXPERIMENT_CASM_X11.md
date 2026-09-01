# CASM-X11 — Soft resource-competitive relational binding

## Status and purpose

CASM-X11 is the preregistered successor to CASM-X10.

X10 established two distinct facts on the validated X9R2 executor:

1. direct independent descriptor-to-fixed-slot classification is exactly competent on trained cardinalities `n=2,3,4` but reproducibly reuses occupied slots for unseen variables at `n=5,6`;
2. the relational binding generators often collapse multiple external variables onto one or two slots even on trained cardinalities, making the X10 coordination comparison ineligible for unseen-cardinality interpretation.

X11 therefore changes only the allocation-learning signal around the X10 relational generators. It does not change the executor, descriptors, supervision, cardinalities, hard evaluation, optimizer schedule, or model parameterization.

## Question

> Is a weak differentiable resource-competition signal sufficient to prevent relational binding collapse and produce an extensible collision-free allocation rule under answer-only supervision, and if so is cross-variable coordination still required?

## Claim boundary

X11 still supplies:

- active cardinality `n`;
- external variable identities `e=0..n-1`;
- deterministic external descriptors `d(e,n)`;
- eight candidate internal slots;
- deterministic slot descriptors `q(s)`;
- command aliases and source/destination indices;
- categorical value ontology plus `EMPTY`;
- the validated local recurrent transition interface.

X11 does not test cardinality inference, variable discovery from raw observations, ontology discovery, or state-interface discovery.

## Frozen data, supervision and executor

Exactly retain X10/X9R2:

- train cardinalities `n ∈ {2,3,4}` only;
- deterministic optimizer-step schedule `2,3,4` repeated;
- train depth 8;
- batch size 128;
- 10,000 optimizer steps;
- fixed final external register 0 is the only task target entering answer loss;
- no teacher forcing;
- no intermediate-state targets;
- no hidden-register targets;
- no semantic operator labels;
- no binding labels;
- no matching or collision repair at evaluation;
- hard evaluation uses independent row argmax;
- X9R2 local slot-identity-invariant executor.

Optimizer for every regime:

- AdamW;
- X9R2 cosine LR schedule from `2e-3` to `2e-4` over 10,000 steps, no warmup;
- weight decay `1e-4`;
- global gradient clipping `1.0`.

Evaluation remains separate for `n=2,3,4,5,6` at IID depth 8, composition depths 12/24, and stress depths 48/96 with `eval_n=256`.

## Soft resource-competition treatment

For a soft row-stochastic binding matrix `B ∈ R^{n×8}`, define the mean pairwise binding overlap:

`R_overlap(B) = (2 / (n*(n-1))) * sum_{i<j} sum_s B[i,s] * B[j,s]`

for `n>=2`.

Properties:

- uniform independent rows give overlap `1/8 = 0.125`;
- two identical one-hot rows give overlap `1` for that pair;
- two distinct one-hot rows give overlap `0`;
- the term is differentiable and requires no target assignment, binding label, matching algorithm, or discrete projection.

Competitive regimes optimize:

`L_total = L_answer + λ * R_overlap(B)`

with frozen `λ = 1.0`.

The overlap term is computed only from the current model-generated binding for the current training cardinality. It reads no target states, semantic labels, answer value, hidden states, or oracle binding.

This is an explicit soft structural prior favoring non-overlapping use of limited slot capacity. It is **not** evidence of spontaneous topology discovery if it succeeds.

## Regimes

Five regimes train on identical batches.

### `canonical_functional`

Deterministic collision-free positive control with the validated local executor. No overlap term is needed because the binding is fixed.

### `relational_independent_no_competition`

Exact X10 `relational_independent` model and answer-only loss. Diagnostic replication on new seeds.

### `relational_coordinated_no_competition`

Exact X10 `relational_coordinated` model and answer-only loss. Diagnostic replication on new seeds.

### `relational_independent_competitive`

Parameter-identical clone of `relational_independent_no_competition`, initialized bit-identically per seed and trained on identical batches, with only `λ * R_overlap` added to its loss.

### `relational_coordinated_competitive`

Parameter-identical clone of `relational_coordinated_no_competition`, initialized bit-identically per seed and trained on identical batches, with only `λ * R_overlap` added to its loss.

The independent competitive and coordinated competitive models must also begin from bit-identical relational-generator parameters, preserving the X10 coordination isolation.

## New independent seeds

- train `20261061`, eval `20261141`;
- train `20261062`, eval `20261142`;
- train `20261063`, eval `20261143`.

No seed may be replaced or dropped based on result.

## Metrics

Retain every X10 hard/soft execution and binding metric.

Additionally record per train step/checkpoint and regime:

- answer loss;
- overlap penalty before weighting;
- weighted overlap contribution;
- total loss;
- gradient norm;
- LR.

Record final `R_overlap(B)` separately for every learned regime/cardinality.

## Integrity requirements

Before training, tests must establish:

1. X9R2 slot-permutation equivalence remains intact;
2. X10 deterministic external/slot descriptor contracts remain intact;
3. no learned external-ID or slot-ID embedding/table is introduced;
4. X10 relational row/slot permutation equivariance remains intact;
5. no-comp and competitive clones are bit-identical before optimization;
6. independent and coordinated competitive relational-generator parameter counts match;
7. `R_overlap` equals `0.125` for uniform rows;
8. `R_overlap` equals `0` for distinct one-hot rows;
9. `R_overlap` equals `1` for two identical one-hot rows;
10. the overlap term is invariant to a simultaneous permutation of binding columns;
11. the overlap term is invariant to external row ordering;
12. overlap gradients reach binding-generator parameters;
13. overlap calculation reads only the generated binding matrix;
14. changing hidden/intermediate targets or semantic labels while preserving final register-0 target leaves answer loss and total competitive loss unchanged;
15. changing final register-0 target can change answer and total loss;
16. no matching, Sinkhorn/Hungarian projection, hard injectivity or collision repair is used in training or evaluation;
17. hard collisions are preserved under independent row argmax;
18. every regime receives identical task batches and optimizer schedule;
19. training cardinality schedule is exactly `2,3,4` repeated;
20. training aborts on non-finite/negative answer loss or non-finite total loss.

## Frozen interpretation

### Positive-control prerequisite

`canonical_functional` must satisfy on every seed and every `n=2..6`:

- hard answer-final >=99% on every suite;
- hard deep step-state exactness >=95% at depths 24/48/96;
- hard deep hidden-register accuracy >=99% at depths 24/48/96.

Failure invalidates X11.

### Seen-cardinality competence

A learned regime is eligible for unseen interpretation only if every seed at `n=2,3,4` satisfies:

- hard and soft answer-final >=99% on every suite;
- hard and soft deep step-state exactness >=95%;
- hard and soft deep hidden-register accuracy >=99%;
- independent hard argmax uses exactly `n` unique slots;
- zero hard collisions;
- mean row maximum >=0.90.

IID depth-8 answer-final below 80% on any seen cardinality is optimization failure.

### Strong unseen-cardinality generalization

A learned regime passes strongly only if every seed at both `n=5` and `n=6` satisfies:

- hard and soft answer-final >=99% on every suite;
- hard and soft deep step-state exactness >=95%;
- hard and soft deep hidden-register accuracy >=99%;
- exactly `n` unique hard-argmax slots;
- zero hard collisions;
- mean row maximum >=0.90.

No averaging rescues a failed cell.

### Partial unseen generalization

If strong fails but every unseen cell has answer-final >=90% and deep step-state/hidden-register >=80%, classify as partial and report the boundary.

### Competition effect

For a relational architecture, soft resource competition is supported as an anti-collapse mechanism only if its competitive version satisfies seen-cardinality competence on every seed while the paired no-competition version fails that prerequisite on at least one of the same seeds.

If both paired versions satisfy seen competence, X11 cannot claim competition is necessary for optimization stability.

If both fail, the overlap prior is insufficient.

### Coordination effect after competition

If `relational_independent_competitive` passes strong unseen generalization, cross-variable coordination is unnecessary under this benchmark.

If independent competitive is seen-competent but fails unseen topology/execution while `relational_coordinated_competitive` passes strongly, cross-variable coordination is required under the tested soft-competition representation.

If both competitive regimes pass strongly, coordination is unnecessary.

If both are seen-competent but fail unseen strongly, the remaining limitation is extrapolation of the binding representation despite anti-collapse pressure.

If hard topology collides while soft execution remains strong, classify as distributed binding rather than discrete allocation.

## Successor boundary

Only a strong unseen-cardinality PASS should authorize removing supplied active cardinality or supplied external-variable identity. Otherwise the next experiment remains inside binding-allocation learning.
