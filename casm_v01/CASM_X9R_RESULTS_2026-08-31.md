# CASM-X9R results — 2026-08-31

## Frozen provenance

- preregistration: `87820f74a8c6b4f6940bd4015c60df1942d3f6f3`
- evaluated implementation head: `a9a403bc1d84747067ef3ca3497e9901d29c7efb`
- workflow run: `33394005080`
- integrity gate: PASS
- all three train/evaluate jobs: PASS

Artifacts:

- seed `20261031`: artifact `9758925686`, sha256 `c26da2d098d9cf02446c232112242f6958c3cdbda04ebc716d110ee3457c19d6`
- seed `20261032`: artifact `9758924847`, sha256 `ed188c2c8516a18f81e1c46a01981ced97516d41c6d47b8184d14bb339a5982f`
- seed `20261033`: artifact `9758945582`, sha256 `b276807438bfa001cb7eddd32dabe0ded6626cde37451854ef8884e6ba076613`

## Frozen classification

**X9R does not establish a cardinality-valid executor.**

It fails the strict preregistered validity PASS and also fails the preregistered near-pass criterion. It does **not** trigger the preregistered catastrophic FAIL clause, because every seed keeps IID depth-8 answer accuracy above 90% and every deep hidden-register accuracy above 90%.

Therefore the exact classification is:

> **strictly insufficient; below near-pass; catastrophic FAIL trigger not reached. Do not proceed to binding-generalization claims.**

The preregistration explicitly requires another executor/data validity repair before binding generation is modified.

## Decisive local-equivariant control

Minimum metric over all cardinalities/suites for each seed:

| seed | min answer-final | min deep step-state exact | min deep hidden-register |
| --- | ---: | ---: | ---: |
| 20261031 | 100.00% | 100.00% | 100.00% |
| 20261032 | 96.48% | 95.98% | 97.67% |
| 20261033 | 85.16% | 86.77% | 92.00% |

The near-pass floor was 98% answer-final, 95% deep step-state exactness and 98% deep hidden-register accuracy in every cell. Seeds `20261032` and `20261033` therefore fail near-pass.

The catastrophic FAIL trigger required either IID depth-8 answer-final below 90% or deep hidden-register below 90%. Minimum IID answer/deep hidden were:

| seed | min IID answer-final | min deep hidden-register |
| --- | ---: | ---: |
| 20261031 | 100.00% | 100.00% |
| 20261032 | 98.83% | 97.67% |
| 20261033 | 98.05% | 92.00% |

So that stronger FAIL trigger is not met.

## Unseen-cardinality stress depth 96

### `local_equivariant_control`

| seed | n | answer-final | step-state exact | hidden-register |
| --- | ---: | ---: | ---: | ---: |
| 20261031 | 5 | 100.00% | 100.00% | 100.00% |
| 20261031 | 6 | 100.00% | 100.00% | 100.00% |
| 20261032 | 5 | 96.48% | 96.74% | 97.93% |
| 20261032 | 6 | 96.88% | 96.03% | 97.67% |
| 20261033 | 5 | 89.84% | 87.99% | 92.00% |
| 20261033 | 6 | 85.16% | 86.77% | 92.42% |

### `x9_absolute_slot_control`

| seed | n | answer-final | step-state exact | hidden-register |
| --- | ---: | ---: | ---: | ---: |
| 20261031 | 5 | 98.05% | 98.25% | 98.80% |
| 20261031 | 6 | 91.41% | 93.01% | 95.79% |
| 20261032 | 5 | 97.27% | 97.95% | 99.02% |
| 20261032 | 6 | 98.05% | 97.72% | 98.81% |
| 20261033 | 5 | 100.00% | 100.00% | 100.00% |
| 20261033 | 6 | 96.09% | 95.74% | 97.58% |

The absolute-slot control also fails strict validity, so X9R's preregistered diagnostic branch says not to alter binding generation next.

## Optimization diagnosis

The local-equivariant architecture is demonstrably capable of the task: seed `20261031` is exactly 100% across every evaluated cardinality and depth through 96. The failure is therefore not a simple representational impossibility.

However, training is strongly seed-sensitive:

- `20261031` converges to final local loss `1.37e-6` and remains stable;
- `20261032` ends at local loss `2.26e-3`, with late recurrent loss spikes (for example `0.1748` at step 9900);
- `20261033` ends at local loss `0.1180`, after late instability and very large pre-clipping gradient norms (including `30.30` at step 10000 and a recorded `69.43` at step 6000).

The original absolute-slot control is comparatively stable on seeds `20261032/33`, despite not meeting the strict threshold on every cell.

This supports a narrow next test of optimization stability for the already-permutation-equivariant local executor rather than another representational modification.

## Claim boundary

X9R establishes only that:

1. exact cardinality transfer through depth 96 is achievable by the local-equivariant executor on at least one seed;
2. removing absolute slot identity does not by itself make the control robust across seeds;
3. the current training recipe is insufficiently stable to validate the executor for a learned-binding experiment.

It does **not** validate cardinality extrapolation, learned binding, or ontology discovery.
