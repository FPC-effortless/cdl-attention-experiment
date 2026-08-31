# CASM-X6 — Learned external-register ↔ internal-slot binding

## Question

CASM-X5 showed that an explicit shared transition can recover an essentially exact hidden trajectory when only one fixed final answer register is supervised. However, X5 still hard-codes the correspondence between external register identities and the model's internal state slots.

CASM-X6 removes that hard-coded correspondence.

> Can answer-only supervision learn a stable, executable binding between external register identities and unlabeled internal state slots while preserving long-horizon hidden-state execution?

A crucial identifiability note is fixed before execution: the absolute internal slot labels are a **gauge symmetry**. Any consistent permutation of the four internal slots represents the same computation. Therefore X6 will not score whether the learned binding equals the identity permutation. It will score whether the learned binding becomes permutation-like and whether the decoded external computation is correct.

## Base

CASM-X6 is stacked on the qualified CASM-X5 result. It preserves:

- four external registers;
- value domain `0..15`;
- contextual command semantics;
- train depth 8;
- fixed terminal answer supervision on external register `0`;
- own differentiable predicted-state rollout;
- no teacher forcing;
- no semantic-operator labels;
- no intermediate state targets.

The intervention is only the register-to-slot correspondence.

## Binding architecture

The model maintains four unlabeled internal slots. A 4×4 binding matrix maps external register identities to internal slots.

Every register-specific operation must pass through the same binding matrix:

1. initial external register values → internal slots;
2. source register `a` lookup;
3. source register `b` lookup;
4. destination register lookup;
5. destination update mask;
6. register-position representation supplied to the transition;
7. internal slots → decoded external registers for the answer loss and evaluation.

There is no independent external-register embedding or direct canonical-slot lookup that can bypass the binding.

For the learned condition, binding logits start from small random noise around the uninformative state and are converted to a doubly-stochastic soft matrix by Sinkhorn normalization during training. No binding labels, permutation targets, identity prior, or entropy/permutation regularizer are supplied.

For discrete evaluation, the learned matrix is projected to the highest-scoring one-to-one permutation. With four slots this projection is evaluated exactly over all 24 permutations.

## Regimes

All regimes use the same transition architecture and identical initial transition parameters.

1. `canonical_binding` — fixed identity external↔internal binding. This is the architectural positive control.
2. `learned_binding` — trainable Sinkhorn binding from near-uniform random logits.
3. `diffuse_binding` — fixed uniform 1/4 binding. This destroys one-to-one register identity and is the negative control.

Only final external register `0` is supervised in all three regimes.

The 16 learned binding logits add less than 0.01% to the model's parameter count and exist in all three model objects; only the learned regime consumes them in its forward pass.

## Budget

- train depth: 8
- steps: 6,000
- batch size: 128
- optimizer: AdamW
- learning rate: 2e-3
- weight decay: 1e-4
- three independent seeds
- IID depth 8; held-out composition depths 12 and 24; stress depths 48 and 96

The larger 6,000-step budget is preregistered because X6 adds an unsupervised binding optimization problem. All regimes receive the same budget.

## Metrics

For each regime report decoded external-state metrics:

- fixed answer-register final accuracy;
- fixed answer-register step accuracy;
- full final-state exactness;
- full step-state exactness;
- all-register accuracy;
- hidden-register accuracy over external registers 1–3;
- hidden-final exactness over external registers 1–3.

For `learned_binding`, also report:

- soft binding matrix;
- mean row maximum;
- mean column maximum;
- mean row entropy;
- best-permutation score (sum of selected soft masses; maximum 4.0);
- projected permutation.

Absolute permutation identity is not a success criterion.

## Preregistered interpretation

### Control prerequisite

The experiment is interpretable only if `canonical_binding` reaches, on every seed:

- at least 99% answer-final accuracy on every suite;
- at least 95% full step-state exactness at depths 24, 48 and 96;
- at least 99% hidden-register accuracy at depths 24, 48 and 96.

If the canonical control fails, X6 is an implementation/optimization failure and no learned-binding claim is made.

### Strong learned executable binding

If the control prerequisite passes, `learned_binding` supports the strong result only if, on every seed:

- answer-final accuracy is at least 99% on every suite;
- full step-state exactness is at least 95% at depths 24, 48 and 96 under the projected discrete binding;
- hidden-register accuracy is at least 99% at depths 24, 48 and 96;
- mean soft row maximum is at least 0.90;
- mean soft column maximum is at least 0.90;
- best-permutation score is at least 3.60 / 4.00.

This would show that answer-only supervision plus the supplied slot-count/value/transition prior can learn a discrete register↔slot alignment sufficient for executable hidden-state computation.

### Computation without discrete slot identification

If decoded answer and hidden trajectory meet the accuracy thresholds but the binding-sharpness thresholds fail, the result supports a learned **distributed binding**, not discrete slot alignment.

### Answer learned, hidden trajectory not recovered

If answer-final accuracy reaches at least 95% but full step-state exactness or hidden-register accuracy stays below 80% at depth 24 or deeper, answer-only supervision has not identified the hidden executable state.

### Binding optimization failure

If `canonical_binding` passes but `learned_binding` remains below 80% answer-final accuracy on IID depth 8, X6 demonstrates a failure of the learned-binding optimization under this preregistered setup; it does not establish impossibility.

## Claim boundary

Even a strong X6 result would still be materially structured. The number of internal slots, categorical value domain, command/argument/destination interface, sparse local transition form, and existence of an explicit recurrent state are all supplied.

X6 tests **binding discovery inside a supplied state ontology**, not discovery of the ontology itself.
