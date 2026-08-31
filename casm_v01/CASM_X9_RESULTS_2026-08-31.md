# CASM-X9 results — 2026-08-31

## Status

**FORMAL CLASSIFICATION: INVALID TEST (positive-control prerequisite failed).**

CASM-X9 must not be cited as evidence either for or against strong shared-generator cardinality extrapolation. The frozen preregistration states that if `canonical_functional` fails its positive-control thresholds, the variable-cardinality transition/data implementation is not a valid test and no shared-generator extrapolation claim is made.

The run nevertheless contains a highly reproducible diagnostic observation: `shared_generator_dense` learns a collision-free, exact seen-cardinality solution for `n=2,3,4`, then every seed reuses already-occupied slots for the new fifth and sixth variables. That observation motivates a follow-up only after the positive-control validity defect is repaired.

## Frozen provenance

- preregistration commit: `e8f2858f2dc7e1ec278039324a3a1e1f2cb7ae0f`
- evaluated implementation head: `e3963aafba8a5a59363d7fa416862c4d5c7918e9`
- workflow run: `33391624966`
- integrity gate: PASS
- training budget: 10,000 optimizer steps per seed
- train cardinalities: `2,3,4` only, deterministic repeating schedule
- eval cardinalities: `2,3,4,5,6`
- hard evaluation: independent row argmax, no collision repair

Seeds and artifacts:

| train seed | eval seed | artifact id | SHA-256 digest |
|---|---|---:|---|
| 20261021 | 20261101 | 9758304670 | `faff7655fb3e0590f20d2b4121622519479e10880c63c85f5944931ca1a5f0b2` |
| 20261022 | 20261102 | 9758068880 | `6096bc5644a5fb3d20a29fbe95eb7eb845bcf4a0530b3a791448f5791e9f3825` |
| 20261023 | 20261103 | 9758310515 | `a494ea350a7831b0919fb5977b5c896b4f8229fc10c20b5b7e7a91d17d88322a` |

All contract, train/evaluate, provenance validator, and artifact-upload jobs completed successfully on the evaluated head.

## Frozen positive-control prerequisite

For `canonical_functional`, every seed and every `n=2..6` was required to achieve:

- hard answer-final accuracy >=99% on every suite;
- hard step-state exactness >=95% at depths 24/48/96;
- hard hidden-register accuracy >=99% at depths 24/48/96.

This prerequisite failed.

Worst per-cardinality canonical metrics (minimum over required/evaluated suites shown):

| seed | n | min answer-final | min deep step-state exact | min deep hidden-register |
|---|---:|---:|---:|---:|
| 20261021 | 2 | 99.61% | 99.72% | 99.80% |
| 20261021 | 3 | **98.44%** | 97.80% | **98.32%** |
| 20261021 | 4 | **98.44%** | 97.54% | **98.56%** |
| 20261021 | 5 | **96.48%** | 96.14% | **97.50%** |
| 20261021 | 6 | **95.31%** | 95.43% | **97.42%** |
| 20261022 | 2 | 100% | 100% | 100% |
| 20261022 | 3 | 100% | 100% | 100% |
| 20261022 | 4 | 100% | 100% | 100% |
| 20261022 | 5 | 99.61% | 99.67% | 99.79% |
| 20261022 | 6 | **98.83%** | 98.62% | 99.25% |
| 20261023 | 2 | 100% | 100% | 100% |
| 20261023 | 3 | 100% | 100% | 100% |
| 20261023 | 4 | 100% | 100% | 100% |
| 20261023 | 5 | **97.66%** | 98.91% | 99.26% |
| 20261023 | 6 | **98.44%** | 97.84% | **98.71%** |

Because the positive-control rule is conjunctive and per-seed/per-cell, these misses invalidate the strong test before the decisive treatment is interpreted.

## Descriptive treatment behavior (not a formal extrapolation verdict)

### Seen cardinalities

The learned shared generator is extremely strong on the trained support.

At depth 96:

- seeds 20261022 and 20261023 are 100% hard and soft answer-final, step-state exact and hidden-register accurate for every `n=2,3,4`;
- seed 20261021 is 100% on `n=2,3`; at `n=4` the minimum answer cell across all suites is 99.22%, deep step-state minimum 99.40%, and deep hidden-register minimum 99.64%;
- every seed has exactly `n` unique hard argmax slots and zero collisions for `n=2,3,4`;
- all seen binding row-max means exceed 0.99.

Thus the shared-generator optimization did not fail on its training cardinalities.

### Unseen topology

Every seed exhibits the same discrete boundary:

| seed | n=5 hard assignment | unique/collisions | n=6 hard assignment | unique/collisions |
|---|---|---|---|---|
| 20261021 | `[7,1,3,2,7]` | 4 / 1 | `[7,1,3,2,7,2]` | 4 / 2 |
| 20261022 | `[5,4,6,3,6]` | 4 / 1 | `[5,4,6,3,6,3]` | 4 / 2 |
| 20261023 | `[4,3,7,0,4]` | 4 / 1 | `[4,3,7,0,7,3]` | 4 / 2 |

The first four variables preserve the learned four-slot decomposition. Variable 5 is mapped onto an occupied slot; variable 6 is mapped onto another occupied slot. There is no matching/collision repair.

### Depth-96 unseen execution

`shared_generator_dense` hard metrics:

| seed | n | answer-final | step-state exact | hidden-register | full final-state exact |
|---|---:|---:|---:|---:|---:|
| 20261021 | 5 | 21.88% | 3.03% | 27.74% | 3.52% |
| 20261021 | 6 | 15.62% | 0.37% | 19.16% | 0.00% |
| 20261022 | 5 | 26.56% | 3.04% | 25.34% | 3.91% |
| 20261022 | 6 | 16.80% | 0.71% | 18.02% | 0.78% |
| 20261023 | 5 | 23.83% | 3.53% | 28.88% | 4.69% |
| 20261023 | 6 | 17.58% | 0.80% | 18.40% | 0.78% |

Soft execution does not rescue the result; its unseen depth-96 metrics are similarly low. This is therefore not merely a hard-discretization artifact in the observed runs.

The fixed diffuse control remains poor, so the learned seen-cardinality behavior is not explained by a generally useful diffuse binding.

## Diagnosis

The most important validity defect is architectural slot/cardinality coverage in the positive control.

With deterministic canonical binding `e -> slot e` and training restricted to `n<=4`, slots 4 and 5 are never used as active canonical external-variable slots during training. Unseen `n=5,6` therefore introduces both greater simultaneous state cardinality and previously unseen active canonical slot positions. The positive control consequently does not cleanly isolate the binding-generator question.

Separately, the learned independent descriptor-to-row generator has no mechanism that coordinates rows as a set. Descriptively, it extrapolates new descriptors by reusing the four learned attractor slots rather than allocating new ones. That is a plausible binding-topology bottleneck, but X9's frozen rules prevent elevating it to the formal conclusion until the control is repaired.

## Qualified conclusion

X9 establishes **no cardinality-extrapolation claim** because its positive-control prerequisite failed.

What may be stated descriptively is:

> Under the evaluated X9 architecture, a shared independent descriptor-to-binding MLP learned exact collision-free bindings on trained cardinalities 2/3/4, while all three seeds mapped unseen fifth and sixth variables onto already occupied slots and lost hidden execution. However, the canonical functional control also missed preregistered generalization thresholds, so the experiment is invalid as a decisive test of binding-rule extrapolation.

## Required follow-up before X10-style binding claims

Run a separately preregistered validity-repair experiment that fixes **slot/cardinality coverage of the executor/control without using X9 unseen treatment targets or changing the X9 interpretation retroactively**. The preferred repair is to make the explicit workspace/transition path slot-permutation equivariant or to use training-time gauge permutations that exercise all candidate slots while retaining training cardinalities `2,3,4`.

Only after the canonical control passes the frozen unseen-cardinality thresholds should a coordinated/set-aware binding generator be compared against the independent X9 generator.
