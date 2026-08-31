# CASM-X2B feature-matched control results — 2026-08-31

## Status

CASM-X2B completed successfully across three independent seeds under workflow run `33363719995`.

Exact evaluated head: `e9cb4bbf8a537de1ef8d4df8d5d78f5042fb1ceb`.

Seeds:

- train `20260871`, eval `20260951`
- train `20260872`, eval `20260952`
- train `20260873`, eval `20260953`

The contract-test gate passed before training. All three train/evaluate jobs passed, including the parameter-match assertions and result artifact uploads.

## Controlled comparison

CASM-X2B removes the two major information-access confounds found after X2 v0 and X2A.

Both the explicit transition model and the feature-matched GRU receive during training:

1. the true previous target state;
2. the whole-state representation;
3. the command/argument representation;
4. direct embedding of `state[a]`;
5. direct embedding of `state[b]`;
6. direct embedding of `state[dst]`;
7. destination-register embedding;
8. the same destination-value target.

During rollout both consume their own predicted previous state.

The feature-matched GRU additionally retains a latent recurrent hidden state. It is therefore a conservative control with at least as much persistent internal memory as the explicit transition MLP.

Parameter counts:

- explicit transition: **246,160**
- feature-matched GRU: **244,832** (`0.9946x` explicit)
- state-only GRU reference: **244,400** (`0.9929x` explicit)

The capacity gate passes cleanly.

## Mean final-state exact accuracy across three seeds

| Suite | Explicit transition | Feature-matched GRU | State-only GRU |
|---|---:|---:|---:|
| IID depth 4 | **100.00%** | **99.05%** | 6.08% |
| held-out composition depth 6 | **100.00%** | **97.05%** | 3.21% |
| held-out composition depth 12 | **100.00%** | **94.36%** | 2.69% |
| stress depth 24 | **100.00%** | **90.97%** | 2.60% |
| stress depth 48 | **100.00%** | **89.67%** | 2.78% |
| stress depth 96 | **100.00%** | **86.98%** | 2.34% |

Per-seed feature-matched GRU final exactness at depth 96:

- `20260871`: 87.76%
- `20260872`: 86.20%
- `20260873`: 86.98%

The depth-96 mean is therefore stable across seeds rather than being driven by one failed run.

## Intermediate-state metrics

Mean feature-matched GRU step-state exactness:

- IID depth 4: **99.52%**
- composition depth 6: **98.50%**
- composition depth 12: **96.91%**
- depth 24: **94.38%**
- depth 48: **92.74%**
- depth 96: **89.90%**

Mean depth-96 per-register accuracy remains **93.32%**, compared with 100% for the explicit transition model.

This pattern is consistent with a small local error rate accumulating over repeated recurrent execution rather than an immediate inability to learn the contextual transition rule.

## Training competence

The feature-matched GRU satisfies the preregistered competence requirement decisively: **99.05% IID depth-4 final-state exactness**.

Its final recorded training losses across the three seeds were approximately `0.0093–0.0277`, compared with roughly `1e-4` for the explicit transition model. The state-only GRU remained near loss `2.0`, confirming that direct sparse indexed-value access was a material part of the earlier gap.

## Preregistered verdict

The relevant preregistered criterion was:

> If the feature-matched GRU reaches at least 80% IID depth-4 exactness but remains >=10 percentage points below explicit transition at depth 48/96, that supports a depth-generalization advantage for the simpler transition bottleneck after state and feature access are controlled.

Observed gaps versus explicit transition:

- depth 24: **9.03 percentage points**
- depth 48: **10.33 percentage points**
- depth 96: **13.02 percentage points**

The criterion is therefore met at both decisive long-depth suites.

Verdict:

> **CASM-X2B supports a real depth-generalization advantage for the explicit stateless transition bottleneck over a parameter-matched recurrent hidden-state control after previous-state access and sparse transition features are equalized.**

## What this result does and does not mean

The result does **not** show that GRUs cannot learn the algorithm. They clearly can: the feature-matched GRU reaches 99% IID accuracy and remains high even at depth 96.

It shows something narrower and more useful:

> once the relevant state is made explicit and sparse transition inputs are exposed, adding a persistent latent recurrent state is unnecessary for this Markov computation and produces measurable long-horizon error accumulation, while the stateless local transition kernel extrapolates exactly.

The evidence therefore shifts the justified architectural claim from “explicit state beats generic neural networks” to:

> **For computation whose next state is conditionally determined by a compact current state and local transition inputs, forcing recurrence through the explicit state itself can generalize more cleanly with execution depth than carrying additional latent recurrent state.**

This remains a controlled synthetic mechanism result. It is not yet evidence for general reasoning, natural-language planning, autonomous operator discovery, or reduced-supervision learning.

## Consequence for the next experiment

The next useful intervention is no longer another hidden-state architecture comparison under full teacher forcing.

The next phase should remove or reduce the true intermediate-state supervision and test whether the explicit computational state can still be learned and maintained from weaker signals such as final outcomes, demonstrations, consistency constraints, or verifier feedback.

That is the point at which verifier-guided learning may become relevant again: not for reranking an already exact transition, but for credit assignment when the correct intermediate state is no longer directly supplied.
