# CASM-X2B — Feature-matched recurrent control

## Audit finding

CASM-X2A equalized access to the previous state but still left a computational-feature asymmetry. `SharedTransitionModel.step_logits()` receives direct embeddings of `state[a]`, `state[b]`, and `state[dst]`, while X2A's state-access GRU had to retrieve those values from a compressed whole-state representation using register identities.

Because the benchmark's contextual semantic rule is explicitly defined from those indexed values, that lookup difference is material.

## Control

`FeatureMatchedStateGRUControl` receives exactly the same transition feature groups as the explicit transition model:

1. whole-state representation;
2. command/argument representation;
3. direct embedding of `state[a]`;
4. direct embedding of `state[b]`;
5. direct embedding of `state[dst]`;
6. destination-register embedding.

During training both models receive the true previous target state. During rollout both consume their own predicted previous state.

The feature-matched control additionally keeps a persistent GRU hidden state. Its parameter count must remain within 5% of the explicit transition model.

The X2A state-only GRU is retained as a reference showing the cost of forcing the recurrent model to perform indexed retrieval from the compressed whole-state embedding.

## Interpretation

Primary comparison: `explicit_transition` vs `feature_matched_gru`.

If the feature-matched GRU reaches within two percentage points of explicit transition across depth 24/48/96, the evidence does not support a unique advantage for the explicit transition MLP. The surviving mechanism is then the broader computational contract: explicit state feedback + local transition factorization + direct sparse access to relevant state fields.

If the feature-matched GRU reaches at least 80% IID depth-4 exactness but remains >=10 percentage points below explicit transition at depth 48/96, that supports a depth-generalization advantage for the simpler transition bottleneck after state and feature access are controlled.

If the feature-matched GRU remains below 80% IID exactness, the model-class comparison remains optimization-confounded and no expressivity claim should be made.
