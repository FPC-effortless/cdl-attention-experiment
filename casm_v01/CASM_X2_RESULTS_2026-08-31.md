# CASM-X2 results — 2026-08-31

## Status

CASM-X2 v0 completed three independent seeds under workflow run `33362789144`.

Exact branch head evaluated: `548143910b4497e42e3cbe45983ba8c0ed7faa48`.

Seeds:

- train `20260841`, eval `20260921`
- train `20260842`, eval `20260922`
- train `20260843`, eval `20260923`

The contract suite passed before training. All three training/evaluation jobs passed. Every evaluation suite contained all eight contextual semantic operators.

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

The explicit-state architecture learned the contextual local transition rule to effectively exact precision and then recursively executed it without detectable error accumulation from training depths 1–4 through evaluation depth 96.

This is materially stronger than CASM-X v0 because command semantics are state dependent: the same opaque command can require a different operator depending on the current predicted register state. A fixed command-to-operator classifier is therefore insufficient.

The result shows a very strong inductive-bias and optimization advantage for forcing computation through explicit predicted typed state in this controlled contextual machine.

## Why the strong causal claim is NOT yet accepted

The preregistered interpretation required the generic controls to learn the IID regime sufficiently well for OOD comparison to be meaningful.

That condition failed.

At the training-depth-matched IID depth-4 suite, mean exact final-state accuracy was only:

- GRU: **5.38%**
- Transformer: **6.86%**

Their training losses also remained high after 2,400 steps, while the explicit-state model converged to approximately `1e-4` loss.

Therefore the current evidence does **not** distinguish cleanly among:

1. an expressivity advantage from explicit typed state;
2. an optimization advantage from the state-transition bottleneck;
3. a sample-efficiency advantage;
4. insufficient optimization/curriculum for the generic controls.

The result should not be described as proving that a GRU or Transformer cannot represent the algorithm.

## Preregistered outcome

The numerical separation easily exceeds the preregistered 10-point OOD margin, but the prerequisite competent-baseline condition is not met.

Verdict:

> **Strong positive mechanism signal, causal architecture claim not yet qualified.**

The falsifier that a matched generic control remains within two points at depth 24/48/96 is not triggered. However, this alone is insufficient because both controls underfit the IID task.

## Required rescue experiment

Before advancing to reduced supervision, run a one-sided convergence rescue that deliberately favors the generic controls.

The rescue should:

1. train substantially longer;
2. use a staged depth curriculum rather than uniformly interleaving depths 1–4;
3. keep parameter counts and input information unchanged;
4. evaluate IID depth 4 first;
5. only interpret depth 24/48/96 if a control becomes competent on IID;
6. record sample/step count required to reach fixed IID thresholds if possible.

A useful conservative criterion is:

- if a generic control reaches at least 80% IID final-state exactness, compare its OOD/depth extrapolation directly against explicit state;
- if generic controls remain far below competence despite substantially more favorable optimization, reinterpret the result as evidence primarily about **optimization and sample efficiency induced by explicit state**, not impossibility of latent-state computation.

## Architectural consequence

The current evidence further narrows CASM's justified core:

> explicit persistent state + local learned transition + recurrent execution

remains strongly supported as a computational inductive bias in controlled program worlds.

Separate operator banks and verifier reranking remain unsupported defaults from X1.

Compression remains orthogonal and should be reintroduced later for state storage, retrieval, and routing rather than used to explain the computational result.
