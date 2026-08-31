# CASM-X10 — Coordinated relational binding generalization

## Status and purpose

CASM-X10 is the first learned-binding successor authorized by the strict CASM-X9R2 executor-validity PASS.

X9 showed a reproducible diagnostic pattern: an independent descriptor-to-eight-logit binding generator was exact and collision-free on trained cardinalities `n=2,3,4`, but produced collisions on unseen `n=5,6`. X9 could not formally decide the binding claim because its original absolute-slot positive control was invalid.

X9R2 has now validated a local, slot-identity-invariant executor/training substrate on three new seeds with 100% execution through depth 96 for every `n=2..6` evaluation cell.

X10 therefore returns to binding generalization while retaining that validated executor.

## Question

> When external variable identities and active cardinality are supplied, what inductive structure is required for a learned binding rule to allocate unseen variables to distinct internal slots and support hidden execution at unseen cardinalities?

X10 separates three hypotheses:

1. direct descriptor-to-fixed-slot logits are sufficient;
2. factorized relational scoring between external-variable descriptors and deterministic candidate-slot descriptors is sufficient even without cross-variable interaction;
3. cross-variable coordination is required in addition to relational slot scoring.

## Claim boundary

X10 still supplies:

- active cardinality `n`;
- external variable indices `e=0..n-1`;
- deterministic external descriptors derived only from `(e,n)`;
- eight candidate internal slots;
- deterministic candidate-slot descriptors;
- command aliases and source/destination indices;
- categorical value ontology and `EMPTY`;
- the explicit local recurrent state-transition architecture validated by X9R2.

A positive result does **not** show inference of cardinality, discovery of variable identity, ontology discovery or discovery of the recurrent state interface.

## Data and supervision

Exactly retain the X9/X9R2 variable-cardinality contextual world.

Training:

- cardinalities `n ∈ {2,3,4}` only;
- deterministic schedule `2,3,4` repeated by optimizer step;
- train depth 8;
- batch size 128;
- 10,000 optimizer steps;
- fixed final external register `0` is the only target entering the loss;
- no teacher forcing;
- no intermediate-state targets;
- no hidden-register targets;
- no semantic operator labels;
- no binding labels;
- no collision penalty;
- no injectivity constraint;
- no entropy/sparsity regularizer;
- no matching or collision repair.

Optimizer for every regime:

- AdamW;
- cosine learning-rate schedule frozen from X9R2: `2e-3` at step 1 to `2e-4` at step 10,000 with no warmup;
- weight decay `1e-4`;
- global gradient clipping `1.0`.

Evaluation:

- `n=2,3,4,5,6` separately;
- IID depth 8;
- held-out composition depths 12 and 24;
- stress depths 48 and 96;
- `eval_n=256` per `(n,depth)` suite;
- report both soft-binding execution and unrepaired independent-row-argmax hard-binding execution.

## Validated executor

All regimes use the X9R2 `LocalEquivariantTransitionModel` transition path:

- no learned absolute slot embedding;
- no flattened slot-position-sensitive workspace input;
- no cardinality feature in the transition;
- no external-variable-ID feature in the transition;
- transition consumes only opaque command embedding and binding-gathered values at `a`, `b`, `dst`;
- the same learned transition parameters are reused across all variables/cardinalities.

The X9R/X9R2 slot-permutation equivalence test remains mandatory.

## External-variable descriptor

Use the exact X9 descriptor `d(e,n)`:

- `e/(n-1)`;
- `n/6`;
- `sin(pi*e/n)`, `cos(pi*e/n)`;
- `sin(2*pi*e/n)`, `cos(2*pi*e/n)`;
- three deterministic binary bits of `e` represented as `{-1,+1}`.

No learned external-ID embedding or per-variable parameter table is allowed.

## Candidate-slot descriptor

For each candidate slot `s ∈ {0,...,7}`, use a deterministic descriptor `q(s)` containing:

- normalized slot position `s/7`;
- `sin(pi*s/8)`, `cos(pi*s/8)`;
- `sin(2*pi*s/8)`, `cos(2*pi*s/8)`;
- three deterministic binary bits of `s` represented as `{-1,+1}`.

No learned slot-ID embedding or free slot embedding table is allowed in either relational binding generator.

These descriptors are binding-side information only; they do not enter the validated transition network.

## Regimes

### `canonical_functional`

Positive control. Deterministic collision-free binding `external e -> slot e` for every `n<=6` using the validated local executor.

This regime must pass the executor-validity threshold before any learned-binding conclusion is made.

### `x9_direct_independent`

Diagnostic replication of X9's binding parameterization on the validated executor.

A shared MLP independently maps each external descriptor directly to eight fixed-column logits:

`d(e,n) -> R^8`.

Rows are independently softmax-normalized. There is no cross-variable interaction, slot descriptor, injectivity prior or collision repair.

### `relational_independent`

A factorized external-slot scorer with no cross-variable interaction.

Architecture:

1. shared external encoder maps each `d(e,n)` to a latent token;
2. shared deterministic-slot encoder maps each `q(s)` to a latent token;
3. a multi-head-attention block is applied to each external token as a **length-one sequence**, so it cannot observe any other external variable;
4. a shared pair scorer maps each resulting external token and candidate-slot token to a scalar score;
5. each external row is independently softmax-normalized over slots.

There is no matching, injective constraint, collision penalty or repair.

### `relational_coordinated`

The decisive coordinated treatment has the **same trainable architecture and parameter count** as `relational_independent`.

The only difference is that the same multi-head-attention block receives all active external tokens together as one set/sequence before pairwise slot scoring. There are no positional embeddings; therefore this interaction is permutation-equivariant to the presentation order of external descriptor rows.

After the interaction, the exact same shared slot encoder and pair scorer produce row logits.

Thus `relational_independent` vs `relational_coordinated` isolates cross-variable interaction while holding slot factorization, parameter count and initialization family constant.

## Binding probability and hard evaluation

For all learned regimes:

- every external row softmax-normalizes independently;
- use X8/X9 capacity-normalized transport for dense/colliding rows;
- hard evaluation uses independent row argmax;
- collisions are preserved exactly;
- no Hungarian matching, Sinkhorn projection, all-different constraint or collision repair is permitted.

## Initialization and pairing

On every seed:

- all regimes begin from an identical cloned local-executor state;
- `relational_independent` and `relational_coordinated` begin from bit-identical binding-generator parameters;
- every regime receives the identical training batch at every optimizer step;
- learned binding outputs must be near-random/near-uniform before training.

## Seeds

Three new independent seeds:

- train `20261051`, eval `20261131`;
- train `20261052`, eval `20261132`;
- train `20261053`, eval `20261133`.

No seed may be dropped or replaced based on result.

## Metrics

For every seed/cardinality/suite/regime record:

- final full-state exactness;
- full step-state exactness;
- per-register accuracy;
- answer-register final accuracy;
- answer-register step accuracy;
- hidden-register accuracy;
- hidden-final exactness.

For every learned binding/cardinality record:

- mean row maximum;
- mean row entropy;
- independent argmax assignment;
- unique selected slot count;
- collision count;
- max/min column occupancy;
- total binding mass.

## Integrity requirements

Before training, tests must establish:

1. the local executor retains X9R/X9R2 slot-permutation equivalence;
2. no learned external-variable ID embedding exists;
3. relational generators contain no learned slot-ID embedding or free slot table;
4. external descriptors read only `(e,n)`;
5. slot descriptors read only `s`;
6. coordinated-generator output is permutation-equivariant under a permutation of external descriptor rows;
7. relational-generator output columns permute equivariantly when candidate-slot descriptor rows are permuted;
8. changing one external descriptor cannot change any other row in `relational_independent` before row softmax;
9. changing one external descriptor is allowed to change other rows in `relational_coordinated`;
10. `relational_independent` and `relational_coordinated` have identical parameter counts;
11. their initial trainable states are bit-identical before the coordination-mode difference is applied;
12. every generated row sums to 1 within `1e-6`;
13. total binding mass equals `n`;
14. near-random initialization has mean row max below `0.18` for every `n=2..6`;
15. capacity-normalized internal probabilities remain normalized under normal and adversarial collisions;
16. hard evaluation preserves independent-row collisions without repair;
17. fixed-answer loss ignores hidden/intermediate targets and semantic labels;
18. changing final register-0 target can change loss;
19. gradients reach both binding generator and executor in every learned regime;
20. all regimes receive identical batches and optimizer schedule;
21. training cardinality schedule is exactly `2,3,4` repeated;
22. evaluation suites and thresholds match X9/X9R2;
23. training aborts on non-finite or negative categorical loss.

## Frozen interpretation

### Positive-control prerequisite

On every seed and every `n=2..6`, `canonical_functional` must achieve:

- hard answer-final >=99% on every suite;
- hard deep step-state exactness >=95% at depths 24/48/96;
- hard deep hidden-register accuracy >=99% at depths 24/48/96.

If this fails, X10 is invalid and no learned-binding conclusion is made.

### Seen-cardinality prerequisite

For a learned regime to be eligible for unseen-cardinality interpretation, every seed on `n=2,3,4` must satisfy:

- hard and soft answer-final >=99% on every suite;
- hard and soft deep step-state exactness >=95%;
- hard and soft deep hidden-register accuracy >=99%;
- hard independent argmax uses exactly `n` unique slots;
- hard collision count is zero;
- mean row maximum >=0.90.

If a learned regime has IID depth-8 answer-final below 80% on a seen cardinality, classify that regime as optimization failure rather than evidence about extrapolation.

### Strong unseen-cardinality binding generalization

A learned regime passes strong unseen generalization only if every seed, for both `n=5` and `n=6`, satisfies:

- hard and soft answer-final >=99% on every suite;
- hard and soft deep step-state exactness >=95%;
- hard and soft deep hidden-register accuracy >=99%;
- hard independent argmax uses exactly `n` unique slots;
- hard collision count is zero;
- mean row maximum >=0.90.

No averaging across seeds/cardinalities/depths may rescue a failed cell.

### Partial unseen generalization

If the strong criterion fails but every unseen-cardinality seed has answer-final >=90% on every suite and deep step-state exactness/hidden-register accuracy >=80%, classify the regime as partial unseen generalization and report the degradation boundary.

### Mechanistic interpretation

If `x9_direct_independent` fails unseen topology while `relational_independent` passes strongly, conclude that **factorized relational external-to-slot scoring is sufficient** and cross-variable coordination is not required by this benchmark.

If `relational_independent` fails unseen topology/execution but `relational_coordinated` passes strongly, conclude that **cross-variable coordination is required under the tested factorized representation**.

If both relational regimes pass strongly, coordination is unnecessary under this benchmark; relational slot factorization is sufficient.

If both relational regimes fail strongly, X10 does not solve binding cardinality extrapolation and the next experiment must change the binding representation/objective rather than the validated executor.

If hard topology collides while soft execution remains strong, classify the result as distributed binding rather than discrete collision-free generalization.

## Successor boundary

Only a strong unseen-cardinality PASS should justify removing supplied cardinality or supplied external-variable identity in the next experiment.
