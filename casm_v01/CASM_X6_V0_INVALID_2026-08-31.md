# CASM-X6 v0 — INVALIDATED

## Status

CASM-X6 v0 at exact head `fcce778dedb737c3395c6237a0d937ba2a6c952a` is **invalid for scientific interpretation**.

Workflow run: `33381114864`.

All three seeds completed training and emitted artifacts, but the post-training validation step failed. Inspection showed the failure exposed a real probabilistic-contract defect rather than a harmless validator tolerance issue.

## Root cause

The learned external-register ↔ internal-slot matrix is used in both directions:

- external register distributions → internal slot distributions;
- internal slot distributions → decoded external register distributions.

For both directions to preserve categorical probability mass, the binding must be doubly stochastic: each row and each column must sum to 1.

The v0 implementation performed 12 alternating log-space row/column normalizations and returned immediately after the column normalization. At sharp learned logits, the resulting matrices had column sums near 1 but row-sum deviations of roughly 5.4%.

Example, seed `20260911`:

- mean row maximum: `0.9784106`;
- best-permutation score: `3.9136424 / 4`;
- projected permutation: `[1, 2, 0, 3]`;
- maximum column-sum error: approximately `5e-8`;
- maximum row-sum error: approximately `0.0536`.

Because decoded rows were not normalized, the selected target mass could exceed 1. This produced negative values for the supposed `-log(p)` objective late in training. A negative categorical negative-log-likelihood is direct evidence that the probability contract was violated.

Therefore the learned-binding metrics from this run cannot be used to support or reject the preregistered X6 hypothesis.

## Descriptive-only observations

The invalid run did show that gradients pushed the soft matrix toward sharp one-to-one structures on every seed, but these observations are non-claim evidence only.

Across the three invalid artifacts:

- mean learned row maximum was about `0.978` on every seed;
- best-permutation score was about `3.91 / 4` on every seed;
- the fixed diffuse negative control remained poor;
- decoded learned-binding execution ranged from very strong to materially seed-unstable.

These results are not accepted because the training objective operated on non-normalized decoded masses.

## Required correction

CASM-X6R must preserve the original question and preregistered interpretation while replacing the finite non-converged normalization with a binding transform that is numerically doubly stochastic to a tested tolerance throughout training.

The corrected integrity contract must additionally assert:

1. row and column sums are within the declared tolerance for adversarially sharp binding logits;
2. initial internal slot distributions sum to 1;
3. decoded external register distributions sum to 1;
4. the fixed-answer loss is non-negative and finite;
5. these checks hold after an optimizer update that sharpens the binding.

X6 v0 is not to be rescued by reinterpretation or by using the already-generated metrics.