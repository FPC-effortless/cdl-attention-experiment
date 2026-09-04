# CASM-X14 Results — Explicit soft occupancy-state allocation

## Provenance

- preregistration: `ce88f2c83729a35774114b8094a75464f5ce3336`
- evaluated implementation head: `be64e00c396704e775cd9d21b42b63fd5c8a94dc`
- workflow: `33810083447`
- integrity gate: PASS (`45 passed`)
- all three train/evaluate/provenance jobs: PASS

Artifacts:

- seed `20261091 / 20261171`: artifact `9915014545`, sha256 `7d6ed3006b3fc8894167257fad4fc111eed41964256f4c562ab25a048026121e`
- seed `20261092 / 20261172`: artifact `9915007271`, sha256 `f687d71a770fff52d1dfef2605644efbead4403843d8ef1bc33f6035b6843d1d`
- seed `20261093 / 20261173`: artifact `9914999809`, sha256 `59f7b9f0025a32b8a8bb66233cf0cf7d6fdb83dce3b2ffecd1f053c2e02d2a1a`

All artifacts are bound to exact evaluated head `be64e00c396704e775cd9d21b42b63fd5c8a94dc`.

## Frozen classification

**CASM-X14 is valid, but the preregistered occupancy-state causal extrapolation claim is INELIGIBLE.**

The canonical functional positive control passes exactly on every seed, every cardinality `n=2..6`, and every evaluated depth through 96.

However:

1. `iterative_no_occupancy` fails seen-cardinality competence on all three seeds, including the preregistered IID-depth-8 optimization-failure boundary.
2. `iterative_occupancy` is exactly seen-competent on seeds 20261091 and 20261092, but seed 20261093 collapses already at seen `n=3,4` and therefore hits optimization failure.
3. The `x13_one_shot_barrier` replication passes seen competence on seeds 20261091 and 20261093 but fails on seed 20261092, so X13's prior every-seed seen result does not strictly replicate on this new seed set.

Because both iterative variants do not satisfy seen competence on every seed, the frozen X14 occupancy-state causal comparison is not eligible. X14 does not establish that occupancy state is or is not sufficient for cardinality extrapolation in general.

## Per-seed morphology

### Seed 20261091

`iterative_occupancy` is exactly competent on all trained cardinalities and generalizes perfectly one step beyond the training maximum:

- `n=2`: assignment `[3,5]`, 2/2 unique, row-max 1.0000, depth-96 hard/soft execution exact;
- `n=3`: `[3,5,0]`, 3/3 unique, exact;
- `n=4`: `[3,5,7,0]`, 4/4 unique, exact;
- unseen `n=5`: `[3,5,2,7,0]`, 5/5 unique, row-max 1.0000, hard and soft answer/trajectory/hidden metrics all 100% through depth 96;
- unseen `n=6`: `[3,5,2,7,2,0]`, 5/6 unique, 1 collision; depth-96 hard answer `16.80%`, step-state exact `2.18%`, hidden-register accuracy `24.76%`.

This is strong diagnostic evidence that the occupancy-aware architecture can implement an extensible allocation policy at least through `4 -> 5` for some initialization, but it does not satisfy the every-seed criterion.

### Seed 20261092

`iterative_occupancy` is exactly competent at trained `n=2,3,4`:

- `n=2`: `[6,0]`;
- `n=3`: `[6,0,5]`;
- `n=4`: `[6,0,5,1]`;

all collision-free and exact through depth 96.

At unseen cardinalities it fails sharply:

- `n=5`: `[6,0,5,1,6]`, 4/5 unique, 1 collision; IID hard answer `30.08%` and depth-96 hard answer `16.02%`;
- `n=6`: `[6,0,5,0,7,1]`, 5/6 unique, 1 collision; depth-96 hard answer `16.02%`, step-state exact `2.00%`, hidden-register accuracy `25.47%`.

### Seed 20261093

`iterative_occupancy` solves only the two-variable seen case robustly:

- `n=2`: `[7,6]`, collision-free and essentially exact;
- `n=3`: `[7,6,6]`, only 2/3 unique, row-max `0.8647`, IID hard answer `36.72%`;
- `n=4`: `[7,6,6,6]`, only 2/4 unique, row-max `0.8244`, IID hard answer `36.72%`.

The collapse persists throughout training rather than appearing only late: diagnostics at steps 500, 2000, 5000 and 8000 repeatedly map the additional `n=3` row onto slot 6, while `n=2` remains solved. Thus this is a seed-sensitive allocation/optimization failure, not merely a final-checkpoint regression.

## Ablation and replication controls

`iterative_no_occupancy` is not a competent control architecture under this setup:

- seed 91 reaches only 2 unique slots for `n=3,4`;
- seed 92 collides already at `n=2`;
- seed 93 collapses every row to one slot.

Therefore X14 cannot attribute any extrapolation difference causally to exposing occupancy, because the occupancy-ablated iterative architecture itself is not consistently trainable.

The X13 one-shot baseline also shows seed sensitivity on this new seed set. Seeds 91 and 93 are exactly seen-competent, while seed 92 remains diffuse/colliding (`n=4` only 2 unique slots, row-max about `0.6301`, IID hard answer `13.67%`). This weakens any interpretation that the new seed set provides a perfect X13 replication baseline.

## What X14 supports descriptively

The strongest descriptive fact is that explicit occupancy-aware refinement can produce a genuinely new allocation for an unseen fifth variable: seed 91 is exact and collision-free at `n=5` despite receiving no train-time forward/loss at `n=5,6`. Seed 92 also learns a fully collision-free seen allocation with the occupancy channel, while its one-shot and no-occupancy comparators fail seen competence.

But this effect is not robust across seeds, and the preregistered causal criteria deliberately disallow averaging or rescuing the claim with the successful seeds.

## Mechanistic diagnosis

X13 showed that a saturation-resistant anti-collision objective is sufficient to repair seen optimization for an independent one-shot scorer on its original seed set. X14 shows that simply exposing generated occupancy to a learned parallel refiner does not turn that into a robust allocation algorithm.

The seed-93 failure is particularly informative: the model learns the two-variable allocation while repeatedly sending the third/fourth rows to an occupied slot. This suggests the remaining bottleneck is not lack of occupancy information itself, but learning a stable **response law** that conserves slot capacity as the variable set grows.

A next experiment should therefore test an explicit soft resource-conservation update rather than another free learned occupancy-response MLP. Such a treatment should still avoid hard matching, hard injectivity and collision repair, and any success must be claimed as evidence for a supplied conservation prior rather than spontaneous topology discovery.

## Claim boundary

X14 does **not** authorize removal of supplied active cardinality or external-variable identity. It does not establish a causal occupancy-state extrapolation effect because the matched iterative comparison fails its every-seed seen-competence prerequisite.

The evidence does justify continuing inside allocation learning with a more algorithmic but still differentiable resource-conservation mechanism.