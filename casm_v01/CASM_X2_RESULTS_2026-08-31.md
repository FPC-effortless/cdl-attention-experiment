# CASM-X2 results — 2026-08-31

## Status

CASM-X2 v0 completed three independent seeds under workflow run `33362789144`.

Exact branch head evaluated: `548143910b4497e42e3cbe45983ba8c0ed7faa48`.

Seeds:

- train `20260841`, eval `20260921`
- train `20260842`, eval `20260922`
- train `20260843`, eval `20260923`

The contract suite passed before training. All three training/evaluation jobs passed. Every evaluation suite contained all eight contextual semantic operators.

## Post-run audit correction

A post-run semantic audit found an important process-information asymmetry that was not called out strongly enough in the original preregistration.

`SharedTransitionModel.training_loss()` teacher-forces the explicit transition model: after step zero, the model receives `batch.target_states[:, t - 1]` — the **true previous target state** — as the state input for the next transition.

The hidden-only GRU and Transformer receive those intermediate states only as prediction targets. They are not fed the true previous target state as the next-step input and must reconstruct the evolving machine state inside their latent computation.

Therefore the v0 comparison does **not** isolate an explicit-state representational bottleneck. It tests a broader intervention that combines:

1. explicit state exposure;
2. teacher-forced local transition decomposition during training;
3. recursive predicted-state execution at inference;
4. a different optimization path from end-to-end latent sequence reconstruction.

This does not invalidate the measured 100% result. It narrows what that result can support. X2 v0 is evidence that the state-exposed transition-learning formulation is an exceptionally strong inductive bias for this world; it is not a clean proof that explicit state representation alone beats an equally informed latent architecture.

CASM-X2A was added after this audit to equalize true previous-state access between the explicit transition model and a parameter-matched recurrent control.

## Parameter match

- explicit predicted-state transition: **246,160** parameters
- GRU hidden-state control: **245,848** parameters (`0.9987x` explicit)
- causal Transformer control: **242,852** parameters (`0.9866x` explicit)

The capacity gate therefore passes cleanly.

## Mean exact final-state accuracy across three seeds

| Suite | Explicit state | GRU | Transformer |
|---|---:|---:|---:|
| IID depth 4 | **100.00%** | 5.38% | 6.86% |
| held-out composition depth 6 | **100.00%** | 0.87% | 0.52% |
| held-out composition depth 12 | **100.00%** | 1.04% | 0.00% |
| stress depth 24 | **100.00%** | 1.22% | 0.00% |
| stress depth 48 | **100.00%** | 0.87% | 0.00% |
| stress depth 96 | **100.00%** | 0.87% | 0.00% |

Mean depth-96 step-state exactness:

- explicit state: **100.00%**
- GRU: **1.85%**
- Transformer: **0.95%**

Mean depth-96 per-register accuracy:

- explicit state: **100.00%**
- GRU: **22.91%**
- Transformer: **11.06%**

## What the result establishes

The state-exposed transition architecture learned the contextual local transition rule to effectively exact precision and then recursively executed it without detectable error accumulation from training depths 1–4 through evaluation depth 96.

This is materially stronger than CASM-X v0 because command semantics are state dependent: the same opaque command can require a different operator depending on the current predicted register state. A fixed command-to-operator classifier is therefore insufficient.

The result shows a very strong inductive-bias, decomposition, and optimization advantage for learning through explicit typed state in this controlled contextual machine.

## Why the strong causal claim is NOT yet accepted

Two independent qualification problems remain.

First, the preregistered interpretation required the generic controls to learn the IID regime sufficiently well for OOD comparison to be meaningful. That condition failed. At IID depth 4, mean exact final-state accuracy was only 5.38% for GRU and 6.86% for Transformer.

Second, the post-run audit found the teacher-forced previous-state input asymmetry described above. Equal parameter count and equal target supervision do not imply equal intermediate information access.

Therefore the current evidence does **not** distinguish cleanly among:

1. explicit state exposure;
2. local transition factorization;
3. teacher-forced process information;
4. optimization/sample-efficiency advantages;
5. an actual representational advantage of explicit state.

The result should not be described as proving that a GRU or Transformer cannot represent the algorithm.

## Preregistered outcome

The numerical separation easily exceeds the preregistered 10-point OOD margin, but the prerequisite competent-baseline condition is not met and the post-run state-access audit adds a further confound.

Verdict:

> **Strong positive mechanism signal for state-exposed transition learning; causal representation claim not qualified.**

The falsifier that a matched generic control remains within two points at depth 24/48/96 is not triggered. However, this alone is insufficient because both hidden-only controls underfit the IID task and do not receive the same teacher-forced previous-state input.

## Qualification experiments

Two controls are now required and are kept conceptually separate.

### Hidden-only convergence rescue

Give the hidden-only controls substantially more favorable optimization while leaving their information interface unchanged. This measures whether the original gap is simply short-run undertraining and quantifies sample-efficiency differences.

### Equal-state-access control

Give a parameter-matched recurrent control the same true previous-state input during training as the explicit transition model, then feed both models their own predicted state at rollout. This is the cleaner test of whether the simple explicit transition bottleneck adds value beyond access to an explicit recurrent state.

The equal-state-access comparison takes priority for causal interpretation.

## Architectural consequence

The justified claim at this checkpoint is:

> exposing a compact typed world state and training a local recurrent transition over that state is a highly effective computational inductive bias in this controlled program world.

It is not yet justified to claim that explicit state representation itself is uniquely necessary, that latent recurrent/Transformer computation is incapable of the algorithm, or that the transition MLP is uniquely responsible.

Separate operator banks and verifier reranking remain unsupported defaults from X1.

Compression remains orthogonal and should be reintroduced later for state storage, retrieval, and routing rather than used to explain the computational result.
