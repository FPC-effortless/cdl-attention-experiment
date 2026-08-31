# CASM-X9R2 — Optimizer-stable cardinality-valid executor

## Status and purpose

CASM-X9R2 is a preregistered optimization-stability repair of CASM-X9R.

X9R showed that the `local_equivariant_control` architecture can achieve exact transfer through unseen cardinalities and depth 96 on one seed, but it was not robust across three seeds. The failing seeds showed late loss spikes and very large pre-clipping gradient norms despite otherwise unchanged architecture/data/supervision.

X9R2 therefore changes **only the learning-rate schedule** for the decisive local-equivariant executor. It does not change the architecture, data distribution, binding, supervision, training cardinalities, evaluation suites, batch size, number of optimizer steps, weight decay or gradient-clipping threshold.

No X9R2 result may retroactively alter X9 or X9R classifications.

## Question

> Does a decayed AdamW learning-rate schedule remove the cross-seed instability of the already permutation-equivariant local executor strongly enough to satisfy the frozen cardinality-validity contract on new independent seeds?

A positive result validates only the executor/training recipe for a later binding-generalization experiment.

## Frozen architecture

Use the exact X9R `local_equivariant_control` architecture from evaluated head `a9a403bc1d84747067ef3ca3497e9901d29c7efb`:

- deterministic canonical external-to-slot binding `e -> slot e`;
- eight-slot categorical workspace;
- no learned absolute slot embedding;
- no flattened slot-position-sensitive workspace input;
- no cardinality feature;
- no external-variable-ID feature;
- shared local transition consuming only opaque command embedding and binding-gathered values at `a`, `b`, and `dst`;
- identical state transport/update semantics.

The X9R slot-permutation equivalence integrity test remains mandatory.

## Data and supervision

Exactly as X9R:

- train cardinalities: `n ∈ {2,3,4}` only;
- deterministic schedule: `2,3,4` repeated by optimizer step;
- train depth: 8;
- batch size: 128;
- optimizer steps: 10,000;
- weight decay: `1e-4`;
- gradient clipping: global norm `1.0`;
- fixed final external register `0` is the only target entering the loss;
- no teacher forcing;
- no intermediate-state targets;
- no hidden-register targets;
- no semantic labels;
- no binding labels.

Evaluation remains:

- `n=2,3,4,5,6` separately;
- IID depth 8;
- held-out composition depths 12 and 24;
- stress depths 48 and 96;
- `eval_n=256` per `(n,depth)` suite.

## Regimes

Two optimizer regimes start from cloned identical model parameters on each seed and receive identical training batches.

### `fixed_lr_replication`

Diagnostic replication of the X9R recipe:

- AdamW learning rate `2e-3` constant for all 10,000 steps;
- weight decay `1e-4`;
- global gradient clipping `1.0`.

This regime is diagnostic and is not required to pass.

### `cosine_decay_stable`

Decisive stability treatment:

- AdamW initial learning rate `2e-3`;
- cosine decay applied from optimizer step 1 through step 10,000;
- final learning rate `2e-4` at step 10,000;
- no warmup;
- weight decay `1e-4`;
- global gradient clipping `1.0`.

The exact learning rate at step `t`, for `t in [1,10000]`, is:

`lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * (t - 1) / (10000 - 1)))`

with `lr_max=2e-3` and `lr_min=2e-4`.

No validation metric, training loss or gradient statistic may alter this schedule during a run.

## New independent seeds

Decisive seeds are new and must not be replaced based on outcome:

- train `20261041`, eval `20261121`;
- train `20261042`, eval `20261122`;
- train `20261043`, eval `20261123`.

The already-observed X9R seeds `20261031..33` are not decisive X9R2 seeds.

## Integrity requirements

Before training, tests must establish:

1. the decisive model architecture is state-dict compatible with the X9R `local_equivariant_control` architecture;
2. simultaneous permutation of internal slot columns and binding columns leaves external soft/hard rollouts equivalent within the existing X9R tolerance;
3. the two optimizer regimes begin from bit-identical model parameters;
4. both regimes receive the identical batch at every optimizer step;
5. the fixed regime remains at exactly `2e-3` every step;
6. the cosine regime starts at exactly `2e-3` and ends at exactly `2e-4`;
7. cosine learning rate is non-increasing for all steps;
8. the schedule equals the preregistered analytical formula at representative boundary/interior steps;
9. weight decay remains `1e-4` in both regimes;
10. global gradient clipping threshold remains `1.0` in both regimes;
11. training cardinality schedule is exactly `2,3,4` repeated;
12. fixed-answer loss remains invariant to hidden/intermediate targets and semantic labels;
13. changing the final register-0 target can change the loss;
14. training aborts on non-finite or negative categorical loss;
15. evaluation code and thresholds are unchanged from X9R.

For diagnostics, record per regime:

- training loss every 100 steps plus step 1 and step 10,000;
- pre-clipping gradient norm;
- applied learning rate;
- minimum and final training loss;
- maximum recorded pre-clipping gradient norm after step 4,000.

## Frozen interpretation

### Validity PASS

X9R2 validates `cosine_decay_stable` only if **every new seed and every cardinality `n=2..6`** satisfies:

- hard answer-final accuracy >=99% on every suite;
- hard step-state exactness >=95% at depths 24, 48 and 96;
- hard hidden-register accuracy >=99% at depths 24, 48 and 96.

No averaging across seeds, cardinalities or depths may rescue a failed cell.

### Near-pass

If every cosine-treatment cell has:

- answer-final >=98%;
- deep step-state exact >=95%;
- deep hidden-register >=98%;

but the strict PASS fails, classify as **near-pass but still insufficient for binding work**.

### Optimization-stability support

Independently of PASS, the stability hypothesis is supported descriptively only if the cosine treatment has, on every seed:

- lower or equal maximum recorded post-step-4000 pre-clipping gradient norm than its paired fixed-LR replication; and
- lower or equal final training loss than its paired fixed-LR replication.

This diagnostic does not override the validity thresholds.

### Failure

If the cosine treatment fails near-pass, do not proceed to binding generation. The next repair must remain inside executor/training validity.

If cosine treatment is materially more stable in training but still fails deep execution, investigate optimization/objective geometry rather than slot representation.

If cosine treatment is not more stable than fixed LR, the X9R seed variance is not adequately explained by late learning-rate instability and a different executor/training repair is required.

## Successor boundary

Only a strict X9R2 Validity PASS authorizes the next learned-binding experiment. That successor should use the validated local-equivariant executor/training recipe and compare independent descriptor-to-row binding against coordinated permutation-equivariant set binding.
