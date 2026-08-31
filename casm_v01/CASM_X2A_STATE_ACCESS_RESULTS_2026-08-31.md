# CASM-X2A equal-state-access results — 2026-08-31

## Status

Workflow run `33363391113` completed successfully across three seeds from exact head `f4dcb694726bba3ef4b7c0af20d3e508074461a4`.

Parameter counts:

- explicit transition: 246,160
- state-access GRU: 244,400 (`0.9929x`)
- hidden-only GRU: 245,848 (`0.9987x`)

All three models therefore satisfy the 5% capacity gate.

## Mean final-state exact accuracy

| Suite | Explicit transition | State-access GRU | Hidden-only GRU |
|---|---:|---:|---:|
| IID depth 4 | **100.00%** | 5.12% | 4.86% |
| composition depth 6 | **100.00%** | 4.17% | 3.21% |
| composition depth 12 | **100.00%** | 2.69% | 0.87% |
| stress depth 24 | **100.00%** | 2.52% | 0.43% |
| stress depth 48 | **100.00%** | 2.34% | 1.04% |
| stress depth 96 | **100.00%** | 2.26% | 1.22% |

The state-access GRU is not competent on IID depth 4, so the preregistered causal comparison does not qualify.

## Post-run audit

X2A fixed one information asymmetry but exposed another.

Both the explicit model and state-access GRU receive the true previous target state during training and their own predicted previous state during rollout. However, they do not receive the same **derived transition features**.

`SharedTransitionModel.step_logits()` receives:

- a whole-state representation;
- a command/argument representation;
- direct embedding of `state[a]`;
- direct embedding of `state[b]`;
- direct embedding of `state[dst]`;
- destination-register embedding.

X2A's state-access GRU receives only the first two. It must recover the indexed register values from a compressed whole-state representation while interpreting the register IDs in the command representation.

That difference is material because CASM-X2's contextual semantic selector is explicitly defined from the values at `a`, `b`, and `dst`.

## Verdict

> **X2A is inconclusive as a model-class comparison.**

It does show that merely handing a recurrent network a compressed whole-state embedding does not reproduce the explicit transition model's optimization behavior.

It does **not** establish that the MLP transition architecture is superior, because direct sparse access to the relevant state fields remains unmatched.

CASM-X2B therefore equalizes the complete transition feature interface before comparing the explicit transition MLP with a parameter-matched GRU.
