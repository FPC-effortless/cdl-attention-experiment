# CASM-X2A — Equal-state-access control

## Why this control is required

Post-run audit of CASM-X2 v0 found a process-information asymmetry.

`SharedTransitionModel.training_loss()` receives the true previous target state as the input state for every transition after step zero. The hidden-only GRU and Transformer receive the true intermediate states only as prediction targets, not as next-step inputs.

Therefore X2 v0 compares a state-exposed, teacher-forced transition learner against sequence models required to reconstruct state internally. Its 100% versus near-zero separation cannot by itself isolate whether the advantage comes from explicit state representation, state exposure during training, local transition factorization, or optimization/sample efficiency.

## Equal-state-access control

CASM-X2A adds a parameter-matched `StateAccessGRUControl`.

During training:

- explicit transition MLP: true previous target state + current instruction -> destination value;
- state-access GRU: true previous target state + current instruction + persistent latent hidden state -> destination value;
- hidden-only GRU: initial state once + instruction history -> full state targets.

During evaluation:

- explicit transition MLP consumes its own predicted previous state;
- state-access GRU consumes its own predicted previous state and also retains latent hidden memory;
- hidden-only GRU retains only latent hidden memory after seeing the initial state.

The state-access GRU is intentionally conservative: it receives at least as much recurrent memory as the explicit transition model.

## Interpretation

Primary comparison: `explicit_transition` vs `state_access_gru`.

If state-access GRU reaches within two percentage points of explicit transition across depth 24/48/96, X2 v0's large hidden-only gap should be attributed primarily to explicit state feedback/access and local Markov decomposition, not to a unique representational advantage of the transition MLP.

If the state-access GRU becomes competent on IID depth 4 but still loses >=10 points at depth 48/96, that is stronger evidence that the simpler explicit transition bottleneck has a depth-generalization advantage even after state access is equalized.

If state-access GRU remains incompetent on IID depth 4, the comparison remains optimization-confounded and should not support an expressivity claim.

The hidden-only GRU is retained only as a replication reference for X2 v0.
