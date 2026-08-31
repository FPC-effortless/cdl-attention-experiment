# CASM-X5 — Fixed answer-register supervision

## Question

CASM-X4 showed that one randomly queried final register value per training example is sufficient to recover the complete hidden state trajectory. That result still distributes terminal supervision across all four registers over the dataset.

CASM-X5 removes that distributed coverage.

> Can the same explicit shared transition recover the complete executable trajectory when every training example exposes only the final value of one fixed answer register, while the other three registers are permanently invisible to the loss?

This is closer to ordinary answer-only supervision: one stable output channel is observed, while latent working state is never directly labeled.

## Base

CASM-X5 is stacked on the completed CASM-X4 result branch. It preserves the same contextual program generator, explicit categorical state, differentiable own-state rollout, model width, optimizer family, train depth and evaluation depths.

## Fixed contract

- training depth: 8
- training steps: 4,000
- batch size: 128
- model: `SoftExplicitTransitionModel(d_model=96)`
- teacher forcing: **none**
- semantic operator labels: **none**
- intermediate state targets: **none**
- fixed answer register: register `0`
- full hidden trajectory is used only for evaluation

Three supervision regimes start from identical parameters:

1. `full_final` — all four final register values;
2. `random_register` — one uniformly sampled final register value per example, reproducing the X4 positive control;
3. `fixed_register` — final value of register `0` for every example; registers `1`, `2`, and `3` are never loss targets.

The query identity is loss-only metadata and is never passed into the transition model.

## Evaluation

Evaluate IID depth 8, held-out composition depths 12 and 24, and stress depths 48 and 96.

Report separately:

- exact full final-state accuracy;
- exact full step-state accuracy;
- all-register accuracy;
- fixed answer-register final accuracy;
- fixed answer-register step accuracy;
- hidden-register accuracy over registers 1–3;
- exact hidden final-state accuracy over registers 1–3.

## Preregistered interpretation

### Strong hidden-trajectory recovery

`fixed_register` supports the strong result only if, on every seed:

- fixed answer-register final accuracy is at least 99% at every suite;
- full step-state exactness is at least 95% at depths 24, 48 and 96;
- hidden-register accuracy is at least 99% at depths 24, 48 and 96.

This would show that a permanently unobserved working-state trajectory can emerge from one stable answer channel under the supplied explicit-state transition architecture.

### Answer learned, hidden state not identified

If fixed answer-register final accuracy is at least 95% but full step-state exactness or hidden-register accuracy remains below 80% at depth 24 or deeper, the result is **not** hidden-trajectory recovery. It shows that the architecture can optimize the observed answer while leaving invisible working state underdetermined.

### Optimization / credit-assignment failure

If fixed answer-register final accuracy remains below 80% on IID depth 8, do not make a state-identifiability claim. The fixed-answer condition has not learned the observed task reliably enough under the preregistered budget.

## Claim boundary

Even a strong positive would not establish autonomous state-schema discovery. CASM-X5 still supplies the register ontology, value domain, command/argument interface, destination identity, explicit differentiable state and shared local transition parameterization.

The experiment isolates supervision coverage, not state-representation discovery.
