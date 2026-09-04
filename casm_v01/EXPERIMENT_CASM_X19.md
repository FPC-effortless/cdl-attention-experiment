# CASM-X19 — Recursive role construction

## Status and purpose

CASM-X19 is preregistered before implementation or execution.

CASM-X18R closes the allocator-repair line: a backward-only dual-price intervention does not robustly stabilize seen allocation across six fresh seeds. The next scientific object is therefore **computational-role construction**, not another variable-to-slot allocator modification.

The architectural distinction is:

- a **role** is a computational identity that should exist independently of storage location;
- a **slot** is only a physical storage address used by the current validated executor.

X19 retains the fixed eight-slot store only as an implementation bridge so the existing executor can run. The treatment question is whether role identities themselves can be generated recursively beyond the role horizon exposed during training.

## Question

> Can a shared recursive role transition, trained only through the first four role positions under answer-only execution supervision, generate additional computational roles that remain collision-free and executable at unseen role positions 4 and 5, relative to a parameter-matched static global-coordinate role generator?

This is a controlled structural-extension test. It is not variable discovery, cardinality discovery, dynamic memory creation, program induction, or cross-episode persistence.

## Frozen regimes

Three regimes receive identical task batches and optimization schedules:

1. `canonical_functional` — deterministic positive control using the validated canonical external-variable-to-slot mapping.
2. `static_global_roles` — nonrecursive parameter-matched control. Each role is generated directly from a deterministic global external-index coordinate.
3. `recursive_roles` — treatment. A learned seed role is extended by repeatedly applying one shared role-transition cell. After the seed, the recursive transition receives **no external index and no active-cardinality input**.

The two learned regimes must have exactly equal total/trainable parameter counts and begin with bit-identical shared-module parameters.

## Frozen role representation

Use role dimension `32`.

Both learned regimes contain:

- one learned seed vector `r_seed in R^32`;
- one shared role cell `F_theta`;
- one identical shared role-to-slot scorer `phi_theta`;
- the same validated X9R2 local-equivariant executor.

No parameter may have first dimension equal to the maximum number of external variables. No learned per-index role table, external-ID embedding, cardinality-ID embedding, or slot-ID embedding is permitted.

### Shared role cell

Use one parameter-shared residual cell for both regimes:

`u = r + C_theta(q)`

`F_theta(r,q) = LayerNorm(u + MLP_theta(u))`

where:

- `C_theta` maps a deterministic 9-dimensional context code into role space;
- `MLP_theta` is shared at every use of the cell;
- the cell has no step-specific parameters.

The exact hidden width and activation must be fixed in implementation tests and identical between treatments.

### Recursive treatment

For `recursive_roles`:

- `r_0 = normalize(r_seed)`;
- use a fixed all-zero 9-dimensional recurrence context `q_step = 0`;
- `r_{i+1} = F_theta(r_i, q_step)` for every subsequent role.

Therefore role position is represented only by how many times the same learned transition has been applied. The model is never given `i`, `n`, or a learned step embedding inside the recursive transition.

Training cardinalities `n=2,3,4` instantiate at most `r_0..r_3`. Roles `r_4` and `r_5` must not be generated, inspected, regularized, logged, or evaluated before all 10,000 optimizer steps finish.

### Static control

For `static_global_roles`:

- retain the same learned seed vector and same shared role cell;
- for each role index `i`, compute the deterministic X18 global coordinate
  `[i/7, 1, sin(pi*i/8), cos(pi*i/8), sin(2*pi*i/8), cos(2*pi*i/8), bit0, bit1, bit2]`;
- generate the role directly as `r_i = F_theta(normalize(r_seed), q_global(i))`.

The static control therefore has direct cardinality-invariant index information but does not recursively construct later roles.

## Frozen storage bridge

Role identity must be separate from storage location.

Both learned regimes use the **same parameterization and initialization** for a shared role-to-slot scorer:

`L_{i,j} = phi_theta(r_i, s_j)`

where `s_j` is the existing deterministic slot descriptor for physical slot `j`.

Requirements:

- the scorer is shared across every role and slot;
- no direct external index or cardinality enters the scorer;
- no recurrent resource state enters the scorer;
- no dual prices, occupancy updates, capacity controller, matching, Sinkhorn, Hungarian projection, sorting assignment, hard mask, or collision repair is used in X19;
- final soft binding is independent row softmax over the eight physical slots;
- hard evaluation is independent row argmax with no repair.

The physical slot bank is explicitly an implementation bridge, not the learned computational ontology.

## Frozen structural objective

Retain the X13 final-binding structural objective only as a storage-bridge regularizer:

- normalized row entropy/spread coefficient `1.0`;
- saturation-resistant pairwise collision barrier coefficient `1.0`, epsilon `1e-3`.

These penalties operate on final role-to-storage probabilities. They are not permitted to modify or create role vectors directly, and there is no explicit correct-slot target.

## Frozen executor and supervision

Retain the validated X9R2 local-equivariant executor unchanged.

Retain answer-only supervision:

- final external register `0` is the only task target;
- no teacher forcing;
- no intermediate-state targets;
- no hidden-register targets;
- no semantic operator labels;
- no binding labels;
- command family and arguments/destination remain supplied exactly as in the current CASM controlled world.

Thus X19 tests role construction, not program induction.

## Frozen data / optimization

Retain the existing variable-contextual benchmark and X18 optimization contract:

- training cardinalities `n in {2,3,4}` only;
- deterministic `2,3,4` repeated schedule;
- train depth `8`;
- batch size `128`;
- exactly `10,000` optimizer steps;
- AdamW;
- weight decay `1e-4`;
- cosine LR `2e-3 -> 2e-4`, no warmup;
- global gradient clipping `1.0`;
- eight physical candidate slots;
- evaluation `n=2..6`, depths `8,12,24,48,96`, `eval_n=256`;
- both hard and soft execution evaluation.

There is **no train-time role generation, forward pass, role diagnostic, storage-binding diagnostic, structural loss, or task loss at n=5,6**. Unseen role generation begins only after optimizer step 10,000.

## Frozen new seed panel

Use six fresh paired seeds:

- train `20261151`, eval `20261231`;
- train `20261152`, eval `20261232`;
- train `20261153`, eval `20261233`;
- train `20261154`, eval `20261234`;
- train `20261155`, eval `20261235`;
- train `20261156`, eval `20261236`.

No seed may be replaced, omitted, rerun selectively, or promoted based on outcome.

## Integrity requirements

Before training, tests must establish:

1. inherited X9R2 executor slot-permutation equivalence;
2. the recursive and static learned models have exactly equal total/trainable parameter counts;
3. shared modules start bit-identically;
4. both learned regimes use role dimension 32;
5. no learned external-ID, cardinality-ID, role-index, step-index, or slot-ID embedding/table exists;
6. the recursive role cell is the exact same module object/parameter set at every recurrence step;
7. recursive generation after `r_0` receives no external index and no active cardinality value;
8. changing a forbidden external-index/cardinality diagnostic argument cannot affect recursive roles;
9. static global context exactly matches the frozen X18 global coordinate for indices 0..5;
10. roles are finite and nonzero at initialization;
11. the role-to-slot scorer is identical and bit-identically initialized between learned regimes;
12. the scorer consumes only role vectors and deterministic slot descriptors;
13. no dual-price, occupancy, capacity, matching, Sinkhorn, Hungarian, hard-mask, or repair path exists in the X19 learned binding function;
14. exact identical role vectors produce identical binding rows and unrepaired hard collisions;
15. row probabilities are normalized and finite;
16. external-row permutation of role vectors permutes binding rows equivalently;
17. physical-slot permutation with correspondingly permuted slot descriptors permutes binding columns equivalently;
18. changing hidden/intermediate targets or semantic labels while preserving final register-0 target leaves all losses unchanged;
19. changing final register-0 target can change answer/total loss;
20. gradients from answer/structural loss reach the recursive role cell, role-to-slot scorer, and executor;
21. static and recursive regimes receive identical batches, optimizer hyperparameters, LR schedule and gradient clipping;
22. training schedule is exactly `2,3,4` repeated;
23. the runner cannot generate or inspect `r_4`, `r_5`, or any n=5/6 learned binding before training completes;
24. hard evaluation is independent row argmax without repair;
25. all losses, roles, logits, bindings and executor outputs abort on non-finite values.

## Post-training structural diagnostics

After optimization only, generate roles through `r_7` for both learned regimes, even though task execution remains evaluated only through n=6.

Report without using these diagnostics to rescue task thresholds:

- role norms for indices 0..7;
- pairwise cosine-similarity matrix;
- minimum pairwise cosine distance;
- consecutive-step cosine similarity;
- role-to-slot hard preferences for indices 0..7;
- number of unique hard storage preferences through indices 0..7.

These diagnostics are intended to detect role collapse, cycles, or trivial saturation beyond the task-evaluated horizon. They do not constitute capability evidence by themselves.

## Frozen classification

### Positive-control prerequisite

`canonical_functional` must satisfy every seed and every `n=2..6`:

- hard answer-final >=99% on every suite;
- hard deep step-state exactness >=95% at depths 24/48/96;
- hard deep hidden-register accuracy >=99% at depths 24/48/96.

Failure invalidates X19.

### Seen-role competence

A learned regime is structural-extension-eligible only if every seed at `n=2,3,4` has:

- hard and soft answer-final >=99% on every suite;
- hard and soft deep step-state exactness >=95%;
- hard and soft deep hidden-register accuracy >=99%;
- exactly `n` unique hard storage slots;
- zero hard collisions;
- mean final soft row maximum >=0.90.

IID depth-8 answer-final <80% on any seen cardinality is optimization failure.

### Strong structural extension

A learned regime passes strongly only if every seed at both unseen `n=5,6` has:

- hard and soft answer-final >=99% on every suite;
- hard and soft deep step-state exactness >=95%;
- hard and soft deep hidden-register accuracy >=99%;
- exactly `n` unique hard storage slots;
- zero collisions;
- mean final soft row maximum >=0.90.

No averaging rescues a failed seed/cell.

### Partial structural extension

If strong fails but every unseen hard and soft cell has answer-final >=90% and deep step-state/hidden-register >=80%, classify partial and report the exact boundary.

### Recursive-construction effect

A causal recursive-vs-static structural-extension comparison is eligible only if **both learned regimes satisfy seen-role competence on all six seeds**.

If eligible:

- recursive strong PASS + static fail: support recursive role construction as the decisive treatment under the supplied fixed storage bridge;
- both strong PASS: recurrence is not necessary under this benchmark/bridge;
- recursive partial + static below partial: report partial improvement only;
- both fail unseen: recursive role transition is insufficient; do not return to allocator tuning;
- either regime fails six-seed seen competence: paired extrapolation comparison is ineligible; diagnose role-generator/storage-bridge optimization, but do not start another allocator-repair series.

## Claim boundary

Even a strong recursive PASS would show only that a supplied shared role-transition architecture can extrapolate executable computational identities beyond the role horizon seen during training while using a fixed physical storage bank and supplied task/controller interface.

It would **not** show:

- discovery of variables from raw observation;
- discovery of active cardinality;
- dynamic memory instantiation;
- removal of the physical slot bank;
- program induction;
- persistence across episodes;
- verifier-guided structure repair.

## Successor boundary

If recursive roles pass strongly on every seed, the next experiment should remove the fixed pre-existing physical slot ontology by dynamically instantiating working-state records from generated roles.

If recursive roles fail unseen while remaining seen-competent, the next model-development step should investigate the role-transition/constructor representation itself, not re-enter dual-price/collision-allocator optimization.
