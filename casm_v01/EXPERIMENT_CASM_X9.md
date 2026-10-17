# CASM-X9 — Variable-cardinality binding generalization

## Motivation

CASM-X8 removed the injective/all-different binding prior. With four known external variables and eight candidate slots, independently normalized binding rows learned from one fixed answer channel spontaneously differentiated into four collision-free slots on every seed and preserved exact hidden execution through depth 96.

However, X8 still gives the model **one free learned binding-logit row for each already-known external variable**. External-variable cardinality is fixed at four during both training and evaluation.

X9 removes that parameterization.

## Question

> Can one shared descriptor→binding generator, trained only on worlds with 2, 3 and 4 active external variables, produce executable variable-to-slot bindings for unseen 5- and 6-variable worlds under fixed-answer-only supervision?

This is a test of **cardinality extrapolation and shared binding-rule generalization**, not inference of cardinality itself.

## Supplied information and claim boundary

For every program, X9 explicitly supplies:

- the active external-variable count `n`;
- external variable indices `e = 0 .. n-1`;
- deterministic non-learned descriptors derived from `(e,n)`;
- command family, source indices `a,b`, and destination index `dst`;
- eight candidate internal slots;
- the categorical value domain and `EMPTY` symbol;
- an explicit recurrent state-transition architecture.

Therefore a positive result does **not** show that the model discovers how many variables exist or derives variable identities from raw observations.

It tests whether a learned binding rule can extrapolate across supplied variable identities and cardinalities without a per-variable parameter table.

## External world

Generalize the contextual X2–X8 world from fixed four registers to variable `n` registers.

For a world with cardinality `n`:

- register values are in `0..15`;
- initial values are sampled independently;
- `a`, `b`, and `dst` are sampled from `0..n-1`;
- command families retain the same four opaque aliases and state-dependent semantic pairs;
- the context bit remains a deterministic function of the current values at `a`, `b`, and `dst`;
- family-order composition holdouts retain the same train/IID vs composition rule.

Training cardinalities:

`n ∈ {2,3,4}`.

Unseen evaluation cardinalities:

`n ∈ {5,6}`.

The answer target remains final external register `0` only.

No other final register, intermediate state, or semantic operator label enters the training loss.

## Deterministic external-variable descriptor

The shared binding generator receives a deterministic descriptor `d(e,n)`. There must be **no learned external-variable ID embedding and no free parameter table indexed by `e`**.

The descriptor is fixed before training and contains only bounded numerical functions of the supplied external index and cardinality. The implementation should use:

- normalized position `e / (n - 1)`;
- normalized cardinality `n / 6`;
- `sin(pi * e / n)` and `cos(pi * e / n)`;
- `sin(2*pi * e / n)` and `cos(2*pi * e / n)`;
- three deterministic binary bits of `e`, represented as `{-1,+1}`.

Training uses `n>=2`, so the normalized position denominator is always defined.

No descriptor component may use target states, semantic labels, future commands or evaluation-only information.

## Shared binding generator

A single MLP `g_theta` maps every descriptor independently to eight binding logits:

`g_theta(d(e,n)) -> R^8`.

The same generator parameters are reused for every `e` and every `n`.

The learned treatment has no parameter whose first dimension is the maximum number of external variables and no embedding table keyed by external index.

Binding rows are independently softmax-normalized exactly as in X8. There is:

- no injective constraint;
- no column-capacity constraint;
- no collision penalty;
- no sparsity penalty;
- no entropy penalty;
- no identity target;
- no matching supervision.

Hard evaluation uses independent row argmax without collision repair.

## Probability transport

Use the X8 capacity-normalized transport so colliding dense rows remain probabilistically valid.

For binding `B[e,s]`:

- each external-variable row sums to 1;
- column occupancy is `c_s = sum_e B[e,s]`;
- `d_s = max(1,c_s)`;
- transport weight is `T[e,s] = B[e,s] / d_s`;
- internal slot state is the transported external categorical mass plus `EMPTY` mass `1-c_s/d_s`;
- external decoding uses the original row-stochastic binding.

The probability contract must hold for all `n=2..6`, including adversarial all-rows-to-one-slot bindings.

## Regimes

### `canonical_functional`

Positive control. Deterministic binding `e -> slot e` for all `n<=6`.

This uses the same transition architecture and proves that the variable-cardinality task and transition implementation are solvable on unseen cardinalities.

### `shared_generator_dense`

Decisive treatment. One shared descriptor-to-eight-logit generator produces every binding row for all cardinalities.

### `diffuse_dense`

Negative control. Every active external variable uses a uniform `1/8` row.

No treatment receives hidden-state supervision.

## Parameter accounting

The treatments should share the same transition architecture and total trainable parameter budget wherever possible.

For fair accounting, the canonical/diffuse controls may retain unused generator parameters initialized identically but must not receive binding-specific gradients. Parameter counts and trainable counts must be reported explicitly.

No treatment may have a learned external-variable embedding or free row-specific binding parameter table.

## Training distribution

Preregistered training budget:

- 10,000 optimizer steps;
- batch size 128;
- train depth 8;
- AdamW learning rate `2e-3`;
- weight decay `1e-4`;
- binding temperature `1.0`;
- cardinalities cycle deterministically as `2,3,4,2,3,4,...` by optimizer step, so exposure is balanced exactly up to the final partial cycle;
- all regimes receive the identical batch on each step.

The larger budget relative to X8 is fixed in advance because the decisive treatment must jointly learn the transition and a descriptor-conditioned binding rule that is reused across multiple cardinalities.

## Seeds

Three independent seeds:

- train `20261021`, eval `20261101`;
- train `20261022`, eval `20261102`;
- train `20261023`, eval `20261103`.

No seed may be dropped or replaced based on outcome.

## Evaluation suites

Evaluate every model on cardinalities `n=2,3,4,5,6`.

For each `n`, evaluate:

- IID/fresh depth 8;
- held-out composition depth 12;
- held-out composition depth 24;
- stress depth 48;
- stress depth 96.

Use `eval_n=256` per `(n,depth)` suite to control CPU cost while retaining a large multi-suite evaluation.

For `n=5,6`, every suite is cardinality OOD because those cardinalities never occur during training.

Record both soft-binding and unrepaired hard-row-argmax execution.

## Metrics

For each cardinality/suite/regime record:

- final full-state exactness;
- full step-state exactness;
- per-register accuracy;
- answer-register final accuracy;
- answer-register step accuracy;
- hidden-register accuracy;
- hidden-final exactness.

For each generated binding record:

- mean row maximum;
- mean row entropy;
- independent argmax assignment;
- unique selected slot count;
- collision count;
- maximum/minimum column occupancy;
- total binding mass.

Report metrics separately for every seed and cardinality. No cross-seed average may rescue a failed seed.

## Integrity requirements

Before training, tests must establish all of the following:

1. the decisive model contains no learned external-variable ID embedding;
2. it contains no free parameter shaped as `[max_external_variables, candidate_slots]` or equivalent per-variable binding table;
3. descriptors are deterministic and identical for identical `(e,n)` inputs across batches;
4. descriptors for unseen `n=5,6` are finite and within the documented numerical ranges;
5. descriptor construction reads only `(e,n)`;
6. near-random shared-generator initialization produces finite row-stochastic bindings for every `n=2..6`;
7. every generated row sums to 1 within `1e-6`;
8. total binding mass equals `n`;
9. capacity-normalized internal categorical states sum to 1 under normal and adversarial colliding generated rows;
10. external decoded categorical distributions sum to 1;
11. hard evaluation preserves row-argmax collisions and performs no matching/collision repair;
12. fixed-answer loss is invariant to all intermediate targets, final targets for registers `1..n-1`, and semantic labels;
13. changing final register-0 target can change the loss;
14. gradients reach the shared binding generator and transition network;
15. no gradient is required through binding in canonical/diffuse controls;
16. no direct external-register embedding bypass exists;
17. variable-cardinality batches produce correct target and rollout shapes for every `n=2..6` at depth 96;
18. canonical functional binding `e->slot e` is valid for every `n<=6`;
19. the deterministic training-cardinality schedule is exactly `2,3,4` repeated by step;
20. neither descriptor nor active cardinality is derived from target states.

Training must abort on non-finite or negative categorical loss.

## Preregistered interpretation

### Positive-control prerequisite

On **every seed and every cardinality `n=2..6`**, `canonical_functional` must achieve:

- hard answer-final accuracy >=99% on every suite;
- hard step-state exactness >=95% at depths 24, 48 and 96;
- hard hidden-register accuracy >=99% at depths 24, 48 and 96.

If this fails for unseen `n=5,6`, the variable-cardinality transition/data implementation is not a valid test and no shared-generator extrapolation claim is made.

### Strong seen-cardinality shared binding

For `n=2,3,4`, `shared_generator_dense` passes the seen-cardinality prerequisite only if **every seed** satisfies:

- hard and soft answer-final accuracy >=99% on every suite;
- hard and soft step-state exactness >=95% at depths 24,48,96;
- hard and soft hidden-register accuracy >=99% at depths 24,48,96;
- hard independent argmax selects exactly `n` unique slots;
- hard collision count is 0;
- mean row maximum >=0.90.

If the canonical control passes but learned treatment IID depth-8 answer accuracy is below 80% on any seen cardinality, classify the setup as binding-generator optimization failure rather than evidence against cardinality generalization.

### Strong unseen-cardinality generalization

If the seen-cardinality prerequisite passes, X9 supports strong cardinality extrapolation only if **every seed**, for both unseen `n=5` and `n=6`, satisfies:

- hard and soft answer-final accuracy >=99% on every suite;
- hard and soft step-state exactness >=95% at depths 24,48,96;
- hard and soft hidden-register accuracy >=99% at depths 24,48,96;
- hard independent argmax selects exactly `n` unique slots;
- hard collision count is 0;
- mean row maximum >=0.90.

No averaging across seeds, cardinalities or depths may rescue a failed cell.

### Partial unseen generalization

If the strong criterion fails but every unseen-cardinality seed has:

- answer-final accuracy >=90% on every suite; and
- deep step-state exactness and hidden-register accuracy >=80%,

classify the result as partial cardinality generalization and report exactly where degradation begins.

### Seen-only solution

If seen `n=2,3,4` passes strongly but either unseen `n=5` or `n=6` has IID depth-8 answer-final accuracy below 80% or deep hidden-register accuracy below 80%, conclude that the shared generator learned a seen-cardinality binding rule without robust cardinality extrapolation.

### Distributed topology

If unseen execution thresholds pass but hard argmax collides or row sharpness stays below 0.90 while soft execution remains strong, classify the result as distributed variable binding rather than discrete collision-free slot generalization.

### Answer-only shortcut

If unseen answer-final accuracy is >=95% but deep full step-state exactness or hidden-register accuracy falls below 80%, classify the result as an answer strategy without recovery of the full hidden executable state.

### Diffuse control

The diffuse control is diagnostic. Strong diffuse performance would weaken the interpretation that variable identity resolution is necessary.

## Claim boundary after a positive result

Even a strong X9 result would establish only:

> a shared binding rule can extrapolate across supplied variable identities and supplied active cardinalities beyond those seen in training while supporting long-horizon hidden execution from one answer channel.

It would still **not** establish:

- inference of active cardinality from raw observations;
- discovery of which candidate external entities are state variables;
- discovery of variable descriptors;
- discovery of the categorical value ontology;
- discovery of the explicit-state transition interface.

A natural successor would present a superset of candidate external entities and require the model to infer the active variable set/cardinality and binding jointly from observations/instructions.