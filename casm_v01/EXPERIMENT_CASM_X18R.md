# CASM-X18R — Detached dual-gradient robustness repair

## Status and purpose

CASM-X18R is preregistered before implementation or execution.

X18 is a valid experiment but its paired descriptor-frame extrapolation comparison is ineligible because the exact relative/X16 learned path is seen-competent on only 2/3 fresh seeds and the global-frame treatment is seen-competent on only 1/3. The failure trajectories show large seed-sensitive changes in the learned binding topology, including a pre-clip gradient spike near 99 on one relative-control seed followed by dual-price saturation and an unrecovered collision state.

X18R does **not** test cardinality extrapolation. It isolates whether differentiating through the eight projected dual-price iterations is itself destabilizing optimization on the already-trained cardinality range.

## Question

> Can the existing eight-round projected dual allocator become reliably optimizable across fresh seeds if its forward price trajectory is preserved exactly but the iterative price-state history is detached from backpropagation?

## Frozen regimes

Three regimes receive identical task batches and optimizer schedules:

1. `canonical_functional` — deterministic positive control.
2. `dual_fullgrad` — exact X16/X17 relative-descriptor priced model with the full differentiable eight-round projected dual trajectory.
3. `dual_detached_prices` — parameter-identical model with the same relative descriptor and exactly the same eight forward dual updates, but every updated price vector is detached before the next round. The final binding remains differentiable with respect to the learned base logits while the gradient does not backpropagate through the price-update history.

The two learned regimes must begin bit-identically and have exactly equal total/trainable learned parameter counts.

## Frozen forward allocator

For learned base logits `L ∈ R^{n×8}` and price state `lambda ∈ R^8`:

- initialize `lambda^0 = zeros(8)`;
- for exactly 8 rounds:
  - `P^t = softmax(L - lambda^t)`;
  - `c^t = sum_rows(P^t)`;
  - `u^{t+1} = relu(lambda^t + (c^t - 1))`;
  - full-gradient control: `lambda^{t+1} = u^{t+1}`;
  - detached treatment: `lambda^{t+1} = stop_gradient(u^{t+1})`;
- final binding `B = softmax(L - lambda^8)`.

Because `stop_gradient` changes only autodiff metadata, the two allocators must be bit-identical in forward binding values and price values for identical logits.

Freeze:

- rounds = 8;
- eta = 1.0;
- unit soft slot capacity;
- zero initial prices;
- no learned price network;
- no hard injectivity, Hungarian/matching, Sinkhorn, sorting assignment, masking, collision repair, or target assignment.

## Frozen representation / executor / objective

Use the exact legacy cardinality-relative external descriptor from X16/X17/X18:

- `e/(n-1)`;
- `n/6`;
- `sin(pi*e/n)`, `cos(pi*e/n)`;
- `sin(2*pi*e/n)`, `cos(2*pi*e/n)`;
- three deterministic binary bits of `e`.

Retain unchanged:

- X16 relational variable↔slot scorer;
- X9R2 slot-identity-invariant local executor;
- final register-0 answer loss only;
- X13 normalized row-spread coefficient 1.0;
- X13 saturation-resistant pairwise collision barrier coefficient 1.0 with epsilon `1e-3`.

No binding labels, hidden-state targets, intermediate targets, semantic operator labels, teacher forcing, descriptor regularizer, or target slot information may be added.

## Frozen data / optimization

This is a **seen-only robustness qualification**.

Training:

- active cardinalities `n ∈ {2,3,4}` only;
- deterministic `2,3,4` repeated schedule;
- depth 8;
- batch size 128;
- 10,000 optimizer steps;
- AdamW;
- weight decay `1e-4`;
- cosine LR `2e-3 -> 2e-4`, no warmup;
- global gradient clipping 1.0;
- fixed final external register 0 is the only task target.

Evaluation:

- cardinalities `n=2,3,4` only;
- depths 8,12,24,48,96;
- `eval_n=256`;
- both final soft binding and unrepaired independent-row-argmax hard binding.

**No model forward, binding diagnostic, descriptor call, loss call, or evaluation at `n=5,6` is permitted anywhere in X18R.** X18R must not produce unseen-cardinality evidence.

## Primary fresh seed panel

Six preregistered fresh seeds:

- train `20261141`, eval `20261221`;
- train `20261142`, eval `20261222`;
- train `20261143`, eval `20261223`;
- train `20261144`, eval `20261224`;
- train `20261145`, eval `20261225`;
- train `20261146`, eval `20261226`.

No seed may be replaced, omitted, selectively rerun, or promoted/demoted based on outcome.

The previously observed X18 seeds are not part of the primary panel and must not be substituted for these seeds.

## Integrity requirements

Before training, tests must establish:

1. inherited X9R2 executor slot-permutation equivalence;
2. exact legacy relative descriptor equality;
3. exact eight-round full-gradient X16 forward dynamics;
4. for a broad deterministic set of random and adversarial logits, `dual_fullgrad` and `dual_detached_prices` produce bit-identical forward price vectors and final bindings;
5. the same equality holds for exact symmetric logits and high-overload logits;
6. the detached treatment still has nonzero finite gradient from final binding/loss to learned base logits;
7. on at least one nonsymmetric crafted case, the full-gradient and detached treatments have measurably different gradients while retaining exactly equal forward values;
8. the detached price state has no gradient history after each update;
9. both learned models start bit-identically and have equal parameter counts;
10. scorer architecture, executor, descriptor, objective, optimizer, LR schedule and batches are otherwise identical;
11. no learned external-ID, cardinality-ID, slot-ID or price-state parameter/table is introduced;
12. hidden/intermediate targets and semantic labels do not affect any loss;
13. changing only final register-0 target can change answer/total loss;
14. structural and answer gradients reach the scorer and executor in both learned regimes;
15. hard evaluation remains independent row argmax with no collision repair;
16. the training cardinality schedule is exactly `2,3,4` repeated;
17. no code path in the X18R runner requests `n=5,6`;
18. all binding values, losses, prices and gradients are finite; training aborts otherwise.

The runner must record pre-clip gradient norm, max dual price, collision barrier, row-max, hard assignment and collision count through training so instability can be localized rather than inferred only from final accuracy.

## Frozen classification

### Positive-control prerequisite

For every seed and every `n=2,3,4`, `canonical_functional` must satisfy:

- hard answer-final >=99% on every suite;
- hard deep step-state exactness >=95% at depths 24/48/96;
- hard deep hidden-register accuracy >=99% at depths 24/48/96.

Failure invalidates X18R.

### Seen competence per learned seed

A learned seed is competent only if for every `n=2,3,4`:

- hard and soft answer-final >=99% on every suite;
- hard and soft deep step-state exactness >=95%;
- hard and soft deep hidden-register accuracy >=99%;
- exactly `n` unique hard-argmax slots;
- zero hard collisions;
- mean final soft row maximum >=0.90.

IID depth-8 answer-final <80% on any seen cardinality is an optimization failure.

### Robust repair

`dual_detached_prices` is a **ROBUST REPAIR PASS** only if all 6/6 fresh seeds satisfy seen competence.

If 5/6 pass, classify **NEAR-ROBUST BUT INSUFFICIENT**.

If <=4/6 pass, classify **REPAIR FAIL**.

No averaging across seeds or cardinalities rescues a failed seed.

### Backward-path effect

A causal backward-path statement is eligible because the learned regimes are parameter-identical, start bit-identically, receive identical batches, and have bit-identical forward allocator values at matched logits; the only treatment difference is autodiff through the dual-price history.

Classify:

- **STRONG STABILIZATION** if detached is 6/6 competent and full-gradient fails at least one seed;
- **FORWARD MECHANISM ROBUST WITHOUT DETACHMENT** if both are 6/6 competent;
- **DETACHMENT INSUFFICIENT** if detached is not 6/6 competent, regardless of the full-gradient count.

Report the exact paired pass/fail pattern and instability diagnostics for every seed. Do not use a seed-count difference alone to claim statistical population-level superiority.

## Claim boundary

A robust X18R pass would show only that removing the backward path through the iterative dual-price history stabilizes optimization of this supplied explicit allocation system across the six frozen seeds. It would not show cardinality extrapolation, variable discovery, slot discovery, learned cardinality inference, or ontology formation.

## Successor boundary

Only a 6/6 detached-price robust repair pass authorizes using that optimizer/allocator path as the base for the next cardinality-extrapolation experiment.

If it passes, the next experiment should return to the unresolved representation question with a shared recursive/procedural external-role generator trained on `n=2,3,4` and tested on unseen `n=5,6`.

If it does not pass, do not proceed to the recursive-role claim; continue optimization diagnosis.
