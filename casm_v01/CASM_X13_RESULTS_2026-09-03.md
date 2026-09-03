# CASM-X13 Results — Saturation-resistant collision barrier

## Provenance

- preregistration: `3d08986b4c401162213db8dc69c0a4f051c97250`
- evaluated implementation head: `ef8fdfc1166dca06ee2f82f3426be32c00a9cc5b`
- workflow: `33513870452`
- integrity gate: **PASS** (`39 passed`)
- train/evaluate jobs: **PASS** for all three preregistered seeds

Artifacts:

- seed `20261081 / 20261161`: artifact `9803610088`, sha256 `4aafcc6a52092293a3cc53ea49ad92cea66918b455776ab6fe1e6bf07ee9c3b7`
- seed `20261082 / 20261162`: artifact `9803658631`, sha256 `46f9d001e596789f451c4105200bd880efe8a744f6a00a4a82697fbfb14e0ce8`
- seed `20261083 / 20261163`: artifact `9803396139`, sha256 `32fc08689f82300cf98ad0ced1b60e7338ab846ed965827d21238758ce9a0abc`

All artifacts are bound to exact evaluated head `ef8fdfc1166dca06ee2f82f3426be32c00a9cc5b`.

## Frozen result

**CASM-X13 partially succeeds at its mechanistic question but does not pass strong unseen-cardinality generalization.**

The canonical functional positive control is exactly competent on every seed, every cardinality `n=2..6`, and every evaluated depth through 96, so X13 is valid.

The preregistered near-collision integrity falsifier passed: the logarithmic barrier supplies at least `100x` the quadratic-capacity corrective gradient norm with respect to the frozen near-saturated logits construction. X13 therefore genuinely tests the proposed saturation-resistance mechanism.

### Independent scorer: seen optimization PASS

`relational_independent_barrier` satisfies the full seen-cardinality competence criterion on **every seed** at `n=2,3,4`:

- hard answer-final: `100%` on every suite;
- soft answer-final: `100%` on every suite;
- hard deep step-state exactness: `100%`;
- soft deep step-state exactness: `100%`;
- hard deep hidden-register accuracy: `100%`;
- soft deep hidden-register accuracy: `100%`;
- exactly `n` unique hard-argmax slots;
- zero collisions;
- row-max mean effectively `1.0`.

The paired `relational_independent_quadratic` comparator fails seen competence on all three same seeds and hits the `<80%` IID answer optimization-failure condition.

Therefore the preregistered **saturation-resistance effect is supported for the independent architecture**: replacing quadratic overflow with the stronger logarithmic barrier converts a non-robust seen optimization problem into an every-seed exact solution under the frozen architecture, data, supervision, optimizer and budget.

This does not establish that the barrier is generally necessary; it establishes the paired causal treatment effect within X13's controlled comparison.

### Independent scorer: unseen extrapolation FAIL

The same independent-barrier models fail at both unseen cardinalities `n=5,6` on every seed.

Seed `20261081`:

- `n=4`: `[7,1,6,0]`, four unique slots, exact execution;
- `n=5`: `[7,1,6,0,6]`, four unique slots, one collision, IID answer `58.20%`;
- `n=6`: `[7,1,6,0,6,1]`, four unique slots, two collisions, IID answer `48.83%`.

Seed `20261082`:

- `n=4`: `[1,0,5,2]`, four unique slots, exact execution;
- `n=5`: `[1,0,5,2,1]`, one collision, IID answer `29.30%`;
- `n=6`: `[1,0,5,2,1,0]`, two collisions, IID answer `21.48%`.

Seed `20261083`:

- `n=4`: `[3,7,4,5]`, four unique slots, exact execution;
- `n=5`: `[3,7,4,5,4]`, one collision, IID answer `60.16%`;
- `n=6`: `[3,7,4,5,4,7]`, two collisions, IID answer `44.92%`.

The unseen bindings remain sharply concentrated: row-max means range from approximately `0.980` to effectively `1.0`. Deep step-state exactness falls to only a few percent or less and hidden-register accuracy is far below the preregistered partial-generalization boundary. Thus the failure is **not** residual softmax diffusion or insufficient row sharpening.

Strong unseen generalization: **FAIL**.

Partial unseen generalization: **FAIL**.

Barrier extrapolation improvement: **NOT SUPPORTED** under the frozen classification.

### Coordinated scorer: non-robust

`relational_coordinated_barrier` is exactly seen-competent on seed `20261083`, but fails seen competence on seeds `20261081` and `20261082`.

- seed `20261081` collapses all trained cardinalities onto one sharp slot;
- seed `20261082` remains largely diffuse with row-max about `0.25` and becomes colliding;
- seed `20261083` is exact at `n=2,3,4` but collides badly at `n=5,6`.

Therefore coordinated-barrier seen competence is **FAIL / non-robust**, and the preregistered coordination comparison is **INELIGIBLE**.

## Preregistered classification

- positive-control prerequisite: **PASS**;
- mechanistic `>=100x` gradient falsifier: **PASS**;
- independent barrier seen-cardinality competence: **PASS on every seed**;
- independent quadratic seen-cardinality competence: **FAIL on every seed**;
- independent saturation-resistance effect: **SUPPORTED**;
- independent strong unseen-cardinality generalization: **FAIL**;
- independent partial unseen generalization: **FAIL**;
- coordinated barrier seen-cardinality competence: **FAIL / non-robust**;
- coordinated strong unseen-cardinality generalization: **INELIGIBLE**;
- barrier extrapolation improvement: **NOT SUPPORTED**;
- coordination-under-barrier claim: **INELIGIBLE**.

## Mechanistic interpretation

X13 separates the X12 failure into two mechanisms.

First, the X12 seen-cardinality collapse was genuinely sensitive to collision-gradient shape. Under bit-identical initialization, batches and optimization, the logarithmic barrier makes the independent scorer exactly collision-free at every trained cardinality on every seed while quadratic overflow fails. Together with the preregistered near-saturation gradient gate, this supports the narrow diagnosis that X12's quadratic correction was too weak after sharpening for robust seen optimization.

Second, eliminating that optimization pathology does not create an allocation algorithm that extrapolates in cardinality. The independent scorer remains a one-shot per-variable map. It learns distinct sharp assignments for the descriptor regions encountered at `n<=4`, but at `n=5,6` new rows map sharply onto slots already occupied by earlier rows. The barrier is a training objective, not an inference-time constraint, so nothing in the independent feed-forward architecture explicitly carries current resource occupancy into the decision for a new row.

This is a structural extrapolation boundary, not a residual sharpening boundary.

The current variable descriptor also depends on both external index and active cardinality, so `n=5,6` introduce descriptor combinations outside the supervised/structurally-trained support. X13 alone cannot determine whether the decisive missing ingredient is explicit resource occupancy, descriptor extrapolation, or another allocation prior.

## Claim boundary

X13 does **not** establish general cardinality generalization, spontaneous state/ontology discovery, or that logarithmic barriers are universally superior collision objectives. It establishes a controlled every-seed seen-optimization rescue for the independent relational scorer and a clean failure of that rescued scorer to extrapolate collision-free allocation to unseen cardinalities.

Per the preregistered successor boundary, removal of supplied active cardinality or external-variable identity is **not authorized**. Work must remain inside binding-allocation learning.

The next experiment should directly test whether the unseen failure is caused by the absence of explicit resource-occupancy state during allocation. A suitable X14 should preserve the barrier, descriptors, executor, supervision and train/unseen cardinality split, while comparing a parameter-matched one-shot allocator against a permutation-equivariant iterative allocator whose only new information is generated slot occupancy from its own current soft binding. No hard masking, matching, target assignment or collision repair should be introduced.