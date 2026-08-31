# CASM-X4 — Partial final-answer supervision

## Question

CASM-X3 showed that the qualified explicit-state transition can recover the exact intermediate trajectory from the **complete final machine state** alone, without teacher-forced intermediate states.

CASM-X4 asks whether that result survives when the terminal target exposes much less of the state.

## Fixed architecture

All regimes use the same `SoftExplicitTransitionModel` from X3, cloned from identical initialization.

- fixed training depth 8;
- no semantic-operator labels;
- no intermediate state targets;
- no teacher forcing;
- all models recursively roll their own differentiable categorical state forward;
- same contextual machine and held-out-composition evaluation as X3;
- full hidden trajectory remains available only to the evaluator, not to weak supervision losses.

## Terminal supervision regimes

### `full_final`

Positive control. Cross-entropy over all four registers of the final target state, equivalent in information content to X3 final-only supervision.

### `one_register`

For every training example, one register is sampled uniformly at random. The loss observes only the final value of that queried register. The other three final register values and all intermediate target states are hidden from the loss.

The query register is used only to select which predicted final-state component is scored. It is not an extra transition-model input.

### `one_parity_bit`

For every training example, one register is sampled uniformly at random. The loss observes only whether that final register value is even or odd. This exposes one terminal bit per program.

Again, the query index is used only by the loss and is not supplied to the transition kernel.

## Integrity requirements

Before training, tests must establish that:

1. all regimes start from identical parameters;
2. rollout ignores all target states;
3. one-register loss is invariant to corruption of all intermediate targets and of the three unqueried final registers;
4. one-parity-bit loss is invariant to corruption of all intermediate targets and to changes of the queried final value that preserve parity;
5. one-parity-bit loss changes when queried final parity is flipped;
6. neither weak loss reads semantic-operator labels;
7. query indices are approximately uniform and generated independently of model predictions;
8. finite gradients reach the shared transition through depth 8.

## Execution contract

Three independent seeds. Each regime receives identical program batches, identical query registers, identical initialization, and the same 4,000-step optimizer budget used in X3.

Evaluation reports full-state metrics at IID depth 8 and held-out composition/stress depths 12, 24, 48, and 96.

## Preregistered interpretation

### Positive-control competence

`full_final` must achieve at least 95% mean IID depth-8 final-state exactness. Otherwise the experiment is not interpretable.

### Strong one-register result

Requires all of:

- mean IID depth-8 full final-state exactness >=90%;
- mean IID depth-8 step-state exactness >=90%;
- mean composition depth-24 full final-state exactness within 10 percentage points of `full_final`;
- no seed below 80% IID full final-state exactness.

Meeting this criterion would show that random one-register terminal projections across programs are sufficient to identify the shared transition and recover the unobserved full trajectory in this controlled world.

### Strong one-parity-bit result

Uses the same criteria. If satisfied, it would show that one terminal bit per program is sufficient under this architecture/data distribution to identify the executable transition.

### Partial result

If one-register passes but parity-bit fails, the evidence supports learning from partial final-state values but not from a single coarse terminal bit at this optimization budget.

### Negative result

If `full_final` is competent but one-register remains below 50% IID full final-state exactness, then the X3 result depends materially on receiving the complete terminal state rather than merely weak terminal projections.

Intermediate outcomes are reported quantitatively without relabeling the thresholds after observation.

## Claim boundary

Even a strong parity result would not show autonomous state-schema discovery. Register identity, value domain, write target, sparse access pattern, and shared explicit transition architecture remain supplied. The experiment tests observability/identifiability of that known state-transition system under progressively weaker terminal supervision.
