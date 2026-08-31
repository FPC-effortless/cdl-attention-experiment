# CASM-X3 — Weak supervision over explicit computational state

## Question

CASM-X2B qualified a depth-generalization advantage for a stateless local transition over explicit sufficient state, but both decisive X2B models were trained with the true previous state at every transition.

CASM-X3 removes that teacher-forced process-state input. The model must roll its own differentiable categorical state forward during training and is varied only in how many intermediate state targets are exposed to the loss.

## Fixed architecture and data

All regimes use the same `SoftExplicitTransitionModel` initialized from identical parameters.

- four registers, values `0..15`;
- contextual opaque commands from CASM-X2;
- command semantics depend on the current state;
- fixed training depth: 8;
- hard autoregressive evaluation at depth 8, 12, 24, 48, and 96;
- no semantic-operator labels;
- no true previous-state teacher forcing;
- every regime rolls its own predicted soft state forward during training;
- evaluation uses its own predicted hard discrete state.

## Supervision regimes

- `process`: target state after all 8 transitions;
- `quarter`: target state only after transitions 4 and 8;
- `final`: target state only after transition 8.

The final-only condition may read the final state target, but must not read any intermediate target state. The integrity suite corrupts unsupervised intermediate targets and requires the relevant loss to remain bit-identical.

## Execution contract

Three independent train/eval seed pairs are used. Each regime receives 4,000 optimizer steps, batch size 128, AdamW at learning rate `2e-3`, weight decay `1e-4`, and identical batches within each seed.

Primary metrics are exact final-state accuracy and exact step-state accuracy. The latter is essential: a final-only model that obtains the right final answer through a shortcut but does not reconstruct the intervening computational states is not evidence for learned persistent state dynamics.

## Preregistered interpretation

### Positive-control competence

The experiment is interpretable only if `process` reaches at least 95% IID depth-8 final-state exactness averaged across the three seeds. If it does not, X3 v0 is an optimization failure and the weaker-supervision regimes are not causally interpretable.

### Strong final-only result

A strong result requires all of:

1. `final` mean IID depth-8 final-state exactness >= 90%;
2. `final` mean IID depth-8 step-state exactness >= 90%;
3. `final` mean composition depth-24 final-state exactness within 10 percentage points of `process`;
4. no seed below 80% IID depth-8 final-state exactness.

This would support the claim that the explicit transition can learn and maintain the correct latent-to-explicit computational trajectory from end-state supervision alone in this controlled world.

### Partial result

If `quarter` satisfies the strong criteria but `final` does not, the evidence supports sparse process supervision rather than outcome-only learning.

If `final` learns IID depth-8 final states but its intermediate step-state exactness is substantially lower, treat that as evidence of endpoint fitting or shortcut computation, not successful discovery of the intended state trajectory.

### Negative result

If `process` is competent but `final` remains below 50% IID depth-8 final-state exactness, outcome-only supervision is insufficient for this formulation at the preregistered optimization budget.

Intermediate results between these thresholds are reported quantitatively without upgrading them into a binary architectural claim.

## Claim boundary

Even a strong result would establish only controlled differentiable state learning from sparse/final state outcomes. It would not establish open-ended state discovery, natural-language reasoning competence, or learning state variables whose schema is itself unknown.
