# CASM-X10 results — 2026-08-31

## Frozen provenance

- preregistration: `e7677e04afa8671130c222aabebec9b0bf7d0b14`
- evaluated implementation head: `5daa1832d14802be44b8bacaef1d9aa94513445a`
- workflow run: `33407322419`
- integrity gate: PASS
- all three train/evaluate jobs: PASS

Artifacts:

- seed `20261051`: artifact `9764307641`, sha256 `c8245c9e880dbd718b6c3c27a31acd406d461e2510bd7672dc6ecf9a98fbff37`
- seed `20261052`: artifact `9764274266`, sha256 `54fa79ab66837f31d0828ec86cee52a4b7cce3ae0a2e28c2237cf0cd25d81f6a`
- seed `20261053`: artifact `9764296916`, sha256 `62546de0dbed87452912fc12b820223b387bfcfd52980459f80a7e8d8cd0be50`

## Frozen classification

X10 yields two distinct results.

1. **`x9_direct_independent` is a clean FAIL of unseen-cardinality binding generalization on a now-valid executor.** It is exactly 100% on every seen-cardinality hard and soft execution cell for all three seeds, then collides reproducibly on unseen variables and collapses in execution.
2. **The two relational regimes are optimization failures under the preregistered contract, so X10 does not establish whether cross-variable coordination is required.** `relational_independent` fails the seen-cardinality prerequisite on seeds 52/53; `relational_coordinated` fails it on every seed.

The positive control is exactly 100% on every seed/cardinality/depth through 96, so unlike X9 the learned-binding result is interpretable.

## Positive control

`canonical_functional` is exactly 100% for hard and soft answer-final, full step-state exactness and hidden-register accuracy for every seed, every `n=2..6`, and every suite through depth 96.

Therefore the executor/data validity prerequisite is satisfied.

## Direct independent binding: clean extrapolation failure

For every seed, `x9_direct_independent` is exactly 100% on all hard and soft execution metrics for trained cardinalities `n=2,3,4` through depth 96.

Its learned topology is also exact on seen cardinalities and then fails at the namespace boundary:

| seed | n=4 assignment | n=5 unique / collisions | n=6 unique / collisions |
| --- | --- | ---: | ---: |
| 20261051 | `[1,6,5,4]` | 4 / 1 | 4 / 2 |
| 20261052 | `[7,0,2,1]` | 4 / 1 | 4 / 2 |
| 20261053 | `[4,6,1,7]` | 4 / 1 | 4 / 2 |

Every new fifth variable is assigned to an already-used slot; every sixth-variable case still uses only four unique slots.

Minimum unseen metrics over `n=5,6` and all suites:

| seed | hard answer-final | hard step-state exact | hard hidden-register | soft answer-final |
| --- | ---: | ---: | ---: | ---: |
| 20261051 | 14.45% | 0.16% | 17.05% | 13.67% |
| 20261052 | 14.45% | 0.13% | 18.19% | 14.84% |
| 20261053 | 16.02% | 0.10% | 18.28% | 16.41% |

This is not seed noise and is not executor failure. The direct descriptor-to-eight-fixed-logit rule learned the observed variable namespace but did not learn an extensible allocation rule for new variables.

## Relational independent

Seed `20261051` becomes a competent seen-cardinality learner: 100% on all hard/soft seen execution cells with collision-free assignments for `n=2,3,4`. It still collides on unseen cardinalities (`n=5`: 4 unique slots; `n=6`: 3 unique slots) and execution collapses.

Seeds `20261052` and `20261053` do not satisfy the preregistered seen-cardinality prerequisite. They collapse their binding rows almost completely onto one slot:

- seed 52: `n=4 -> [0,0,0,0]`;
- seed 53: `n=4 -> [0,0,0,0]`.

Minimum seen hard answer-final is 14.84% and 5.86% respectively, far below the preregistered 80% optimization-failure boundary.

Therefore `relational_independent` is classified as **optimization failure across seeds**, not a valid unseen-generalization test.

## Relational coordinated

`relational_coordinated` fails the seen-cardinality prerequisite on all three seeds.

Representative final seen topology:

- seed 51, `n=4`: `[4,6,4,4]` — only 2 unique slots;
- seed 52, `n=4`: `[3,6,6,6]` — only 2 unique slots;
- seed 53, `n=4`: `[7,7,7,7]` — 1 unique slot.

Minimum seen hard answer-final across suites is 13.28%, 23.05% and 14.06% respectively. These are optimization failures under the frozen X10 criterion.

Thus X10 provides **no valid evidence that cross-variable coordination is either necessary or sufficient**. The coordinated treatment never became a competent seen-cardinality learner.

## Mechanistic diagnosis

The validated executor is not the limiting component. The binding side exhibits two different failure modes:

1. **direct fixed-column classifier:** stable, sharp and exact on the observed namespace, but reuses old slots for unseen variables;
2. **relational scorer:** symmetry collapse under answer-only training, often concentrating many variables on one or two slots before a competent seen-cardinality allocation is learned.

The next experiment should therefore repair relational allocation collapse while retaining:

- the validated X9R2 executor;
- answer-only supervision;
- no binding labels;
- no hard injectivity/matching/collision repair.

A narrow test is a differentiable resource-competition/overlap cost between variable binding rows. This is a soft anti-collapse signal rather than an enforced assignment.

## Claim boundary

X10 establishes that, in this controlled variable-cardinality world:

> A shared independent descriptor-to-fixed-slot classifier can perfectly learn and execute the trained variable namespace but does not extrapolate slot allocation to unseen variables; new variables reproducibly reuse occupied slots.

X10 does **not** establish whether factorized relational scoring or cross-variable coordination can solve the extrapolation problem, because the corresponding regimes fail the preregistered seen-cardinality competence requirement on multiple/all seeds.

It remains a controlled synthetic state-binding result, not general ontology discovery or variable discovery from raw observations.
