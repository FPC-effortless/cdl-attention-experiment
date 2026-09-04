# CASM-X15 Results — Soft capacity-conservation allocation

## Provenance

- preregistration: `c4cf322d1ee12a8dd28ae9e94303a2bdade7b295`
- evaluated implementation head: `1285032b1ba6f5528e7544d116c696944dbd01a8`
- workflow: `33833682412`
- integrity gate: PASS
- all three train/evaluate/provenance jobs: PASS

Artifacts:

- seed `20261101 / 20261181`: artifact `9922815622`, sha256 `8d7259d77d6a9afc2524f404aac75e93f723ea1f7e94e8962bcbd47d0b894b81`
- seed `20261102 / 20261182`: artifact `9922812303`, sha256 `e57b160bff06780c192d3ff06b099312585f264a8e2dcafd77d5ad92655105b9`
- seed `20261103 / 20261183`: artifact `9922806524`, sha256 `d2aa2ac1136d6b4e8d5523795fa5857eab6635599ea0c9fba4e6242e9d828d6c`

All artifacts are bound to exact evaluated head `1285032b1ba6f5528e7544d116c696944dbd01a8`.

## Frozen classification

**CASM-X15 is valid and shows a material descriptive robustness improvement from the supplied soft capacity law, but it does not pass the preregistered every-seed criterion and does not achieve strong or partial unseen-cardinality generalization.**

The canonical functional positive control is competent on every seed, every cardinality `n=2..6`, and every evaluated depth through 96. X15 is therefore valid under its positive-control prerequisite.

The parameter-identical `capacity_neutral` scorer fails seen-cardinality competence on all three seeds. The `capacity_conserving` treatment improves that to exact seen competence on seeds 20261101 and 20261103, but seed 20261102 collides already at the trained `n=4` case and crosses the preregistered IID-depth-8 `<80%` optimization-failure boundary.

Consequently the supplied capacity law is **not a robust PASS** under the frozen criteria. No averaging rescues the failed seed.

Neither learned regime satisfies strong or partial unseen-cardinality generalization at both `n=5,6` on every seed.

## Per-seed topology and execution

### Seed 20261101

`capacity_neutral` is not seen-competent. Its final hard assignments are:

- `n=2`: `[7,4]`, 2/2 unique, row-max mean about `0.7525`;
- `n=3`: `[7,4,4]`, 2/3 unique;
- `n=4`: `[7,4,4,4]`, 2/4 unique;
- `n=5`: `[7,4,4,4,4]`;
- `n=6`: `[7,4,4,4,4,4]`.

At `n=4`, depth-96 hard answer accuracy is about `30.47%`, step-state exactness about `4.13%`, and hidden-register accuracy about `21.70%`.

`capacity_conserving` is exactly competent on all trained cardinalities:

- `n=2`: `[3,1]`, exact hard/soft execution;
- `n=3`: `[3,1,2]`, exact;
- `n=4`: `[3,1,2,0]`, exact;
- all three are collision-free with row maxima effectively 1.0 and 100% hard/soft execution through depth 96.

At unseen `n=5`, the hard topology is also correct:

- assignment `[3,1,2,4,0]`, 5/5 unique;
- hard answer/trajectory/hidden execution is 100% through depth 96.

However the final soft binding is not sufficiently sharp: row-max mean is about `0.851852`, below the frozen `0.90` threshold. Soft evaluation also fails badly (IID answer about `68.75%`; depth-96 soft answer about `24.61%`, step-state exactness about `5.20%`, hidden-register accuracy about `28.59%`). Therefore unseen `n=5` is not a strong or partial PASS under the frozen criteria despite the exact hard support.

At unseen `n=6`, assignment `[3,1,2,1,0,4]` uses only 5/6 unique slots and has one collision. Depth-96 hard answer is about `27.34%`, step-state exactness about `1.65%`, and hidden-register accuracy about `24.87%`.

### Seed 20261102

`capacity_neutral` again fails seen competence:

- `n=2`: `[0,5]`, only moderately sharp;
- `n=3`: `[0,5,5]`, one collision;
- `n=4`: `[0,5,5,5]`, two collisions;
- additional variables continue to reuse slot 5.

`capacity_conserving` solves `n=2,3` exactly but fails at the trained `n=4` case:

- `n=2`: `[1,2]`, exact;
- `n=3`: `[1,0,2]`, exact;
- `n=4`: `[1,0,2,0]`, only 3/4 unique, one collision, row-max mean about `0.876099`.

At `n=4` IID depth 8, hard answer accuracy is about `49.22%` and soft answer about `44.53%`, which triggers the preregistered optimization-failure category. At depth 96, hard answer is about `32.42%`, step-state exactness about `5.87%`, and hidden-register accuracy about `26.61%`.

Unseen allocation degrades further:

- `n=5`: `[1,0,2,0,2]`, 3/5 unique;
- `n=6`: `[1,0,2,2,2,2]`, 3/6 unique.

This single seed is sufficient to falsify an every-seed capacity-law PASS.

### Seed 20261103

`capacity_neutral` fails seen competence:

- `n=2`: `[1,5]`, moderately sharp;
- `n=3`: `[1,1,5]`, one collision;
- `n=4`: `[1,1,1,5]`, two collisions;
- later rows continue to reuse those slots.

`capacity_conserving` is exactly seen-competent:

- `n=2`: `[0,3]`;
- `n=3`: `[0,3,4]`;
- `n=4`: `[0,3,4,7]`;
- hard and soft answer/trajectory/hidden metrics are 100% through depth 96.

But unseen cardinalities collide immediately:

- `n=5`: `[0,3,4,7,4]`, only 4/5 unique, one collision, row-max mean about `0.9863`;
- `n=6`: `[0,3,4,7,4,7]`, only 4/6 unique, two collisions, row-max mean about `0.9990`.

At unseen `n=5`, depth-96 hard answer is about `23.05%`, step-state exactness about `3.66%`, hidden-register accuracy about `25.94%`. At `n=6`, those are about `18.36%`, `0.42%`, and `17.43%` respectively.

## Preregistered decisions

- positive-control prerequisite: **PASS**;
- `capacity_neutral` seen competence: **FAIL on all three seeds**;
- `capacity_conserving` seen competence: **PASS on 2/3 seeds, FAIL / optimization failure on seed 20261102**;
- capacity-conserving strong unseen generalization: **FAIL**;
- capacity-conserving partial unseen generalization: **FAIL**;
- robust capacity-law treatment PASS: **FAIL**;
- capacity-law descriptive robustness improvement over neutral: **SUPPORTED descriptively, but not sufficient for the preregistered causal success criterion**.

## Mechanistic interpretation

X14 and X15 now show a consistent hierarchy.

1. Exposing generated occupancy to a learned refiner can help and can occasionally extrapolate one cardinality beyond training, but is seed-sensitive.
2. Replacing the learned occupancy response with an explicit parameter-free remaining-capacity law improves seen allocation robustness relative to the matched neutral scorer.
3. The remaining-capacity update is nevertheless myopic: each round reacts to current occupancy but does not maintain a persistent global resource price or dual state that accumulates capacity violations across rounds.

Seed 20261101 is particularly informative. The supplied capacity law finds a collision-free hard five-variable allocation despite no `n=5` training exposure, but the corresponding soft state is not stable/sharp enough to satisfy the frozen probabilistic execution criterion. At `n=6` the hard allocation itself collides. Thus the treatment can extend support selection beyond the training maximum without learning a robust soft allocation algorithm.

The next experiment should therefore test a persistent global resource-state mechanism rather than another local occupancy penalty or response MLP. A suitable falsifier is an entropic primal-dual allocator in which each slot carries a persistent nonnegative price that accumulates overload across fixed refinement rounds and feeds back into all row preferences.

## Claim boundary

X15 does not authorize removal of supplied active cardinality or external-variable identity. It does not establish that capacity constraints alone solve allocation generalization. It shows that the supplied parameter-free remaining-capacity prior materially improves allocation robustness over a parameter-identical neutral scorer but remains insufficient for every-seed seen stability and unseen cardinality generalization.