# CASM-X12 Results — Soft scarcity-economy binding

## Provenance

- preregistration: `8e4bd2f1d5a4b77e36e0e409f4a11530058be6f2`
- evaluated implementation head: `d635d893506c2437ecedd68c61605fefc02fcefd`
- workflow: `33466324276`
- integrity gate: PASS
- train/evaluate jobs: PASS for all three preregistered seeds

Artifacts:

- seed `20261071 / 20261151`: artifact `9785395061`, sha256 `3f40020524120b420bc022fd04bd08a7ea8edf8326cc0fce81c4c39c01440d79`
- seed `20261072 / 20261152`: artifact `9785290776`, sha256 `5cc21700666ad2c072c62efed1a766596c8e87023bdf8b86d0b272df1daf7879`
- seed `20261073 / 20261153`: artifact `9785406803`, sha256 `651217e6e64fb6ee904d08e0f2683aa8c1fa17508732bd54ee92444c53b021f8`

The workflow and all artifacts are bound to exact evaluated head `d635d893506c2437ecedd68c61605fefc02fcefd`.

## Frozen result

**CASM-X12 does not pass.**

The canonical functional positive control is exactly competent on every seed, every cardinality `n=2..6`, and every evaluated depth through 96. X12 is therefore valid under its preregistered positive-control prerequisite.

Neither scarcity regime satisfies seen-cardinality competence on any seed. Both hit the preregistered optimization-failure condition (`IID depth-8 answer-final <80%` on at least one trained cardinality) for every seed. Consequently neither scarcity regime is eligible for unseen-cardinality or coordination interpretation.

The scarcity objective is therefore **not supported as a robustness improvement over overlap competition** under the frozen X12 architecture, weights, data, optimizer and budget.

## Per-seed topology

### Seed 20261071

`relational_independent_scarcity`:

- `n=2`: `[0,0]`, 1 unique slot, 1 collision, row-max mean `0.5000`;
- `n=3`: `[0,0,0]`, 1 unique, 2 collisions;
- `n=4`: `[0,0,0,0]`, 1 unique, 3 collisions, row-max mean `0.5000`;
- `n=5`: 1 unique, 4 collisions;
- `n=6`: 2 unique, 4 collisions.

Seen IID answer accuracy is only `31.64% / 20.31% / 14.84%` for `n=2/3/4`.

`relational_coordinated_scarcity`:

- `n=2`: `[1,0]`, collision-free;
- `n=3`: `[1,0,0]`, 1 collision;
- `n=4`: `[1,0,0,0]`, 2 collisions;
- `n=5`: 2 unique slots, 3 collisions;
- `n=6`: 2 unique slots, 4 collisions.

It reaches `98.83%` IID answer at `n=2` but collapses to `34.77%` at `n=3` and `30.86%` at `n=4`.

### Seed 20261072

`relational_independent_scarcity` is exact at `n=2`, then sharply collapses:

- `n=3`: `[3,6,6]`, 1 collision;
- `n=4`: `[3,6,6,6]`, 2 collisions;
- `n=5`: 2 unique slots, 3 collisions;
- `n=6`: 2 unique slots, 4 collisions.

Seen IID answer falls to `37.11%` at `n=3` and `29.69%` at `n=4`.

`relational_coordinated_scarcity` likewise uses only two slots from `n=3` onward and has seen IID answer `96.88% / 21.88% / 31.25%` for `n=2/3/4`.

By contrast, both paired overlap controls are seen-competent on this seed, demonstrating that the scarcity treatment can be materially worse than its paired comparator.

### Seed 20261073

`relational_independent_scarcity` collapses to one dominant hard slot for every cardinality; row-max mean remains only about `0.3391`, consistent with a three-way soft mixture whose rows are effectively identical. Seen IID answer is `33.98% / 19.92% / 17.58%`.

`relational_coordinated_scarcity` improves row sharpness but still collides:

- `n=2`: collision-free;
- `n=3`: 2 unique slots;
- `n=4`: 3 unique slots;
- `n=5`: 3 unique slots;
- `n=6`: 3 unique slots.

Seen IID answer is only `44.92% / 30.86% / 31.64%`.

The paired independent-overlap control is exactly seen-competent on this seed, while the paired coordinated-overlap control itself collapses. Scarcity therefore does not provide a robust rescue.

## Preregistered classification

- positive-control prerequisite: **PASS**;
- independent scarcity seen-cardinality competence: **FAIL / optimization failure**;
- coordinated scarcity seen-cardinality competence: **FAIL / optimization failure**;
- strong unseen-cardinality generalization: **INELIGIBLE** for both scarcity regimes;
- partial unseen-cardinality generalization: **INELIGIBLE** for both scarcity regimes;
- scarcity-vs-overlap robustness improvement: **NOT SUPPORTED**;
- scarcity-vs-overlap extrapolation improvement: **NOT SUPPORTED**;
- coordination under scarcity: **INELIGIBLE**.

## Mechanistic diagnosis

The implementation matches the preregistered objective: normalized row entropy penalizes diffuse rows and quadratic overflow penalizes probability mass above unit slot capacity. No binding labels, matching, hard injectivity or collision repair are present.

The observed failures expose an optimization problem rather than a missing global optimum. Distinct one-hot assignments have zero structural cost, but several trained models terminate in sharply colliding configurations with positive capacity cost. For example, seed 20261072 independent-scarcity at `n=4` has three rows sharing slot 6, row-max mean `1.0`, normalized spread `0`, and capacity overflow `1.0`.

A plausible mechanism is softmax saturation: the entropy term can make rows nearly one-hot before the quadratic probability-space capacity term has separated them. Once a row is saturated, the derivative from probability-space overflow back through its logits can become very small even while the overflow cost remains positive. X12 is consistent with that explanation but does not by itself prove it.

## Claim boundary

X12 does **not** show that scarcity/resource constraints are generally harmful or that collision-free allocation cannot be learned. It shows that this specific entropy-plus-quadratic-overflow objective, at the frozen weights and optimization budget, is not a robust answer-only allocation mechanism.

The next experiment should directly test the saturation diagnosis rather than adding more architecture. A suitable treatment is to retain row-compactness pressure but replace quadratic overflow with a smooth collision barrier whose repulsive gradient increases as two binding rows approach identical one-hot occupancy. This remains a soft structural prior and should be compared against the X12 scarcity objective from bit-identical initialization.