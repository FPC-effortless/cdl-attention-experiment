# CASM-X17 — Dual-horizon convergence falsifier

## Status and purpose

CASM-X17 is preregistered before implementation or execution.

X16 established that an 8-round projected primal-dual slot-price state makes the learned independent relational allocator exactly competent on trained cardinalities `n=2,3,4` on all three new seeds, while the matched zero-price control is not robust. However every priced X16 seed still collides at unseen `n=5,6` and fails even partial generalization.

The immediate ambiguity is whether X16 failed because eight dual rounds are too shallow for the resource-price dynamics to converge, or because the learned variable→slot preference function itself does not extrapolate to newly appearing variable descriptors.

X17 changes no learned representation, data, executor, supervision, objective, optimizer, candidate-slot count or train/unseen split. It changes only the fixed number of primal-dual refinement rounds.

## Question

> Does extending the same projected primal-dual allocator from 8 rounds to 64 rounds turn the learned answer-supervised preference function into a robust unseen-cardinality allocator?

A negative result would not prove that every possible resource allocator is insufficient. It would specifically rule out the simple explanation that X16 failed merely because its frozen 8-round unit-step dual horizon was too short.

## Frozen learned model and executor

Retain X16 exactly:

- X10/X13 independent relational variable↔slot scorer unchanged;
- deterministic external-variable descriptors unchanged;
- deterministic candidate-slot descriptors unchanged;
- no learned external-ID, active-cardinality-ID or slot-ID embedding/table;
- eight candidate slots;
- X9R2 slot-identity-invariant local executor;
- fixed answer register 0 only;
- X13 row-spread + logarithmic collision-barrier structural objective.

## Frozen primal-dual rule

For learned base logits `L ∈ R^{n×8}` and slot prices `lambda`:

- initialize `lambda^0 = zeros(8)`;
- at every round: `P^t = softmax(L - lambda^t[None,:], dim=slot)`;
- occupancy: `c^t_s = sum_i P^t_{i,s}`;
- projected update: `lambda^{t+1}_s = relu(lambda^t_s + (c^t_s - 1))`;
- final binding: `softmax(L - lambda^T[None,:], dim=slot)`.

Dual step size remains exactly `eta=1.0`.

## Regimes

Three trainable/evaluable regimes receive identical task batches and optimizer schedules:

1. `canonical_functional` — deterministic positive control.
2. `dual_short_8` — X16-equivalent priced allocator with exactly 8 dual rounds.
3. `dual_long_64` — parameter-identical priced allocator with exactly 64 dual rounds.

`dual_short_8` and `dual_long_64` must begin bit-identically and have identical trainable/total parameter counts. The only treatment difference is the parameter-free dual iteration count.

### Post-training inference-only diagnostic

After all 10,000 optimization steps are complete and before no scientific classification is changed, evaluate the **trained `dual_short_8` parameters** with the same 64-round allocator as an inference-only diagnostic, named `dual_short_eval64`.

This diagnostic receives no separate optimizer and no additional training. It answers whether merely allowing the already-trained X16-style preference function more dual iterations at inference can rescue unseen topology.

The canonical success classification remains based on the independently trained `dual_short_8` and `dual_long_64` regimes. `dual_short_eval64` is mechanistic diagnostic evidence only.

## Frozen objective

For final binding `B`:

- `R_spread(B) = mean_i H(B_i)/log(8)`;
- with epsilon `1e-3`, `R_barrier(B)=mean_{i<j} -log(1-(1-epsilon) dot(B_i,B_j))`;
- `L_total = L_answer + R_spread + R_barrier`, coefficients `1.0`, `1.0`.

No separate capacity loss is added.

## Frozen training/evaluation

Retain X16:

- train cardinalities exactly `{2,3,4}`;
- deterministic schedule `2,3,4` repeated;
- train depth `8`;
- batch `128`;
- `10,000` optimizer steps;
- AdamW weight decay `1e-4`;
- cosine LR `2e-3 -> 2e-4`, no warmup;
- global gradient clipping `1.0`;
- no teacher forcing;
- no intermediate/hidden targets;
- no semantic operator labels;
- no binding labels;
- no train-time forward, structural diagnostic or loss at unseen `n=5,6`.

Evaluation: `n=2..6`, depths `8,12,24,48,96`, `eval_n=256`, hard unrepaired row argmax and soft binding.

## New seeds

- train `20261121`, eval `20261201`;
- train `20261122`, eval `20261202`;
- train `20261123`, eval `20261203`.

No seed replacement, omission, selective rerun or post-hoc horizon tuning is allowed.

## Integrity requirements

Before training, tests must establish:

1. inherited X9R2/X13/X16 executor, leakage and descriptor contracts;
2. exact projected dual update unchanged from X16;
3. `dual_short_8` executes exactly 8 rounds;
4. `dual_long_64` executes exactly 64 rounds;
5. eta is exactly 1.0 for both;
6. learned parameters begin bit-identically and counts match exactly;
7. both allocators preserve row and slot permutation equivariance;
8. every primal row is categorical every round and all prices are finite/nonnegative;
9. exact symmetric logits remain symmetric for both horizons and hard collisions remain unrepaired;
10. no hard injectivity, matching/Hungarian, Sinkhorn, sorting assignment, hard masking or collision repair exists;
11. changing hidden/intermediate targets or semantics while preserving final register-0 target leaves every loss unchanged;
12. changing final register-0 target can change answer/total loss;
13. gradients from total loss reach scorer and executor through both horizons;
14. all trainable regimes consume identical batches and optimizer schedules;
15. no `n=5,6` model forward or binding diagnostic occurs before optimization completes;
16. inference-only `dual_short_eval64` is created only after training and shares the exact trained short-model parameters without an optimizer step.

## Frozen interpretation

### Positive-control prerequisite

`canonical_functional` must satisfy every seed and every `n=2..6`:

- hard answer-final >=99% every suite;
- hard deep step-state exactness >=95% at depths 24/48/96;
- hard deep hidden-register accuracy >=99% at depths 24/48/96.

Failure invalidates X17.

### Seen competence

A learned regime is unseen-eligible only if every seed at `n=2,3,4` has:

- hard and soft answer-final >=99% every suite;
- hard and soft deep step-state exactness >=95%;
- hard and soft deep hidden-register accuracy >=99%;
- exactly `n` unique hard slots;
- zero collisions;
- mean final soft row max >=0.90.

IID depth-8 answer-final <80% on any seen cardinality is optimization failure.

### Strong unseen generalization

Every seed at both `n=5,6` must have:

- hard and soft answer-final >=99% every suite;
- hard and soft deep step-state exactness >=95%;
- hard and soft deep hidden-register accuracy >=99%;
- exactly `n` unique hard slots;
- zero collisions;
- mean soft row max >=0.90.

No averaging rescues a failed cell.

### Partial unseen generalization

If strong fails but every unseen cell has answer-final >=90% and deep step-state/hidden-register >=80%, classify partial.

### Horizon effect

A causal horizon claim is eligible only if both `dual_short_8` and `dual_long_64` satisfy seen competence on every seed.

If eligible:

- long passes strong unseen and short does not: 8-round convergence horizon is supported as the decisive limitation under this fixed primal-dual rule;
- both pass strongly: 64 rounds are not necessary under the shared scorer;
- both fail unseen: 64-round unit-step projected dual pricing is insufficient, ruling out the simple '8 rounds were merely too short' explanation;
- long is partial and short fails partial: report partial horizon improvement only;
- long fails seen: classify long-horizon optimization failure, not evidence against long convergence in general.

### Inference-only diagnostic

If `dual_short_eval64` repairs unseen topology/performance while native `dual_short_8` does not, report that longer inference convergence can rescue the short-trained preference function even if independently trained long-horizon treatment behaves differently. If it remains colliding, this strengthens the conclusion that extra inference iterations alone are insufficient.

## Successor boundary

Only a strong every-seed unseen PASS authorizes moving to active-cardinality or variable-identity inference. If both horizons fail unseen, the next experiment should alter the learned preference representation or allocation procedure rather than continue increasing the same dual horizon.
