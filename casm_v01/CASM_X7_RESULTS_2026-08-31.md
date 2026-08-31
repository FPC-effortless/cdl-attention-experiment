# CASM-X7 results — 2026-08-31

## Verdict

**PASS — strong sparse injective binding.**

CASM-X7 asked whether fixed-answer-only supervision can select four distinct useful internal slots from eight candidates while preserving hidden executable state. The preregistered strong criterion required every seed to clear all capability and assignment thresholds; no cross-seed averaging was allowed.

All three seeds passed.

## Exact provenance

Preregistration: `casm_v01/EXPERIMENT_CASM_X7.md`

Exact evaluated head:

`cf607987fc610445286bd91e94ed86928c7a8656`

Workflow run:

`33385696407`

Artifacts:

- seed `20261001`: artifact `9755903033`, sha256 `7e03dbf9afe069ed7d91228b2debfc3444dfc4f7b45d6d0c8815a5bbfd9de2f0`
- seed `20261002`: artifact `9755918183`, sha256 `110a63d72eb9f62b4f0924ba6fdb48f248afa82669d9037a8fd3ed627c194cb1`
- seed `20261003`: artifact `9755905088`, sha256 `58d44d602d4e6b84d13db840a50570a652e2b88b35e91e752a3748561ddcbcd5`

All integrity tests and all three 8,000-step training/evaluation/validator jobs completed successfully.

## Frozen criterion versus worst observed value

| Criterion | Required on every seed | Worst observed |
|---|---:|---:|
| answer-final accuracy, every suite | >=99% | **100%** |
| step-state exactness, depths 24/48/96 | >=95% | **100%** |
| hidden-register accuracy, depths 24/48/96 | >=99% | **100%** |
| mean binding row maximum | >=0.90 | **0.982643** |
| best injective-assignment score | >=3.60 / 4 | **3.930574 / 4** |
| projected selected-slot count | 4 distinct | **4 on every seed** |

The canonical sparse positive control also achieved 100% on all three capability criteria across every relevant seed/suite.

The learned soft binding itself also remained at 100% capability through depth 96 on every seed, so the result is not created only by the final discrete projection.

## Learned assignments

The learned near-uniform assignment sharpened independently to different non-canonical sparse subsets:

| Seed | Projected register -> slot assignment | Row-max mean | Best assignment score |
|---|---|---:|---:|
| `20261001` | `[7, 2, 6, 0]` | 0.982643 | 3.930574 / 4 |
| `20261002` | `[7, 5, 3, 4]` | 0.984381 | 3.937524 / 4 |
| `20261003` | `[6, 3, 7, 2]` | 0.986241 | 3.944963 / 4 |

Initial learned row maxima were approximately `0.127`, close to the uniform 1/8 value. The different final slot IDs are expected gauge symmetry; internal slot labels have no canonical semantic identity.

## Diffuse negative control

The permanently diffuse 4x8 assignment remained poor. At depth 96, three-seed means were:

- answer-final accuracy: **5.56%**;
- step-state exactness: **0.70%**;
- hidden-register accuracy: **7.86%**.

This is consistent with resolved variable-to-slot identity being important in this benchmark. It does not establish necessity against every possible dense learned representation because X7 compares the sparse learned model to a fixed uniform diffuse control, not to an unconstrained learned dense assignment.

## Qualified claim

Within a supplied four-external-variable ontology, eight candidate internal slots, an explicit recurrent-state interface, and a supplied injective-assignment family, fixed-answer-only supervision can select four distinct internal slots from surplus capacity and learn essentially exact hidden executable dynamics that extrapolate from training depth 8 to depth 96.

X7 therefore weakens the exact-slot-count prior established in X6R: the model is not told which four of eight candidate slots should carry computational state.

This is **not autonomous ontology discovery**. Still supplied are:

- the existence and cardinality of four external variables;
- eight candidate internal slots;
- categorical value domain and `EMPTY` surplus symbol;
- the injective-assignment family;
- command, argument and destination identities;
- explicit recurrent state;
- the shared local transition architecture.

## Next falsifier

The next experiment should relax the **injective binding prior itself**, rather than adding more surplus slots. A useful design is to compare:

1. the X7 injective sparse assignment;
2. a learned row-normalized dense 4x8 binding with no all-different constraint;
3. a sparsity-capable learned binding whose cardinality/topology is not fixed to one slot per external variable.

The key question is whether answer-only learning independently discovers a discrete, reusable variable decomposition when one-to-one topology is no longer guaranteed by architecture.