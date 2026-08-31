# CASM-X2 results — 2026-08-31

## Executive result

CASM-X2 now has a qualified positive result after three successive controls removed confounds in the original comparison.

The strongest justified finding is:

> **For this controlled contextual Markov program world, a stateless learned transition that repeatedly updates an explicit machine state generalizes more cleanly with execution depth than a parameter-matched GRU given the same previous-state access and the same sparse transition features.**

This is not a claim that GRUs or Transformers cannot represent the computation. The feature-matched GRU learns it to 99% IID exactness. The difference appears in repeated execution: its small local error accumulates with depth, while the explicit transition remains exact through depth 96.

## X2 v0 — contextual state machine

Workflow run: `33362789144`.

Exact evaluated head: `548143910b4497e42e3cbe45983ba8c0ed7faa48`.

Three seeds:

- `20260841` / eval `20260921`
- `20260842` / eval `20260922`
- `20260843` / eval `20260923`

The benchmark uses four registers with values `0..15`. The same opaque command family can invoke different semantic operators depending on a context bit computed from the current `state[a]`, `state[b]`, and `state[dst]`. Training uses depths 1–4; held-out command-family bigrams are evaluated through depths 6, 12, 24, 48, and 96.

Parameter counts:

- explicit transition: 246,160
- hidden-only GRU: 245,848
- causal Transformer: 242,852

Mean final-state exact accuracy:

| Suite | Explicit transition | Hidden-only GRU | Transformer |
|---|---:|---:|---:|
| IID depth 4 | **100.00%** | 5.38% | 6.86% |
| composition depth 6 | **100.00%** | 0.87% | 0.52% |
| composition depth 12 | **100.00%** | 1.04% | 0.00% |
| stress depth 24 | **100.00%** | 1.22% | 0.00% |
| stress depth 48 | **100.00%** | 0.87% | 0.00% |
| stress depth 96 | **100.00%** | 0.87% | 0.00% |

This was a strong mechanism signal but not a clean model-class comparison. Post-run audit found that the explicit transition model was teacher-forced with the true previous state during training, while the hidden-only controls had to reconstruct the evolving state internally.

## Hidden-only convergence rescue

Workflow run: `33363140352`.

The hidden-only controls alone received a favorable 10,000-step staged curriculum, versus 2,400 steps in X2 v0.

Mean final-state exactness after rescue:

| Suite | GRU | Transformer |
|---|---:|---:|
| IID depth 4 | 3.04% | 18.84% |
| composition depth 6 | 1.13% | 0.95% |
| composition depth 12 | 0.43% | 0.09% |
| stress depth 24 | 0.61% | 0.09% |
| stress depth 48 | 0.61% | 0.17% |
| stress depth 96 | 0.43% | 0.00% |

This rules out a modest optimization-step shortage as the sole explanation and supports a large sample/optimization-efficiency advantage for local state-exposed transition learning. It does not resolve the information-access asymmetry.

## X2A — equal previous-state access

Workflow run: `33363391113`.

A parameter-matched GRU was given the same true previous state during training and its own predicted state during rollout.

Mean final-state exactness:

| Suite | Explicit transition | State-access GRU |
|---|---:|---:|
| IID depth 4 | **100.00%** | 5.12% |
| composition depth 6 | **100.00%** | 4.17% |
| composition depth 12 | **100.00%** | 2.69% |
| stress depth 24 | **100.00%** | 2.52% |
| stress depth 48 | **100.00%** | 2.34% |
| stress depth 96 | **100.00%** | 2.26% |

Audit then found a second material asymmetry. `SharedTransitionModel` receives direct embeddings of `state[a]`, `state[b]`, and `state[dst]` in addition to a whole-state representation and command representation. X2A's GRU received only the compressed whole-state and command representations, so it still had to learn indexed retrieval internally.

X2A is therefore informative but inconclusive as a model-class comparison.

## X2B — feature-matched recurrent control

Workflow run: `33363719995`.

Exact evaluated head: `e9cb4bbf8a537de1ef8d4df8d5d78f5042fb1ceb`.

Three seeds:

- `20260871` / eval `20260951`
- `20260872` / eval `20260952`
- `20260873` / eval `20260953`

X2B equalizes the decisive transition interface. Both the explicit transition and feature-matched GRU receive:

- true previous target state during training;
- own predicted previous state during rollout;
- whole-state representation;
- command/argument representation;
- direct `state[a]`, `state[b]`, and `state[dst]` value embeddings;
- destination-register embedding;
- identical destination-value supervision.

The GRU additionally retains a latent hidden state.

Parameter counts:

- explicit transition: **246,160**
- feature-matched GRU: **244,832** (`0.9946x`)
- state-only GRU reference: **244,400** (`0.9929x`)

### Decisive accuracy

| Suite | Explicit transition | Feature-matched GRU | State-only GRU |
|---|---:|---:|---:|
| IID depth 4 | **100.00%** | **99.05%** | 6.08% |
| composition depth 6 | **100.00%** | **97.05%** | 3.21% |
| composition depth 12 | **100.00%** | **94.36%** | 2.69% |
| stress depth 24 | **100.00%** | **90.97%** | 2.60% |
| stress depth 48 | **100.00%** | **89.67%** | 2.78% |
| stress depth 96 | **100.00%** | **86.98%** | 2.34% |

The feature-matched GRU is a competent baseline. Its IID depth-4 exactness is 99.05%, well above the preregistered 80% competence threshold.

Its mean step-state exactness declines with depth:

- depth 4: 99.52%
- depth 6: 98.50%
- depth 12: 96.91%
- depth 24: 94.38%
- depth 48: 92.74%
- depth 96: 89.90%

Depth-96 final exactness by seed was 87.76%, 86.20%, and 86.98%, showing that the effect is stable across seeds.

### Preregistered criterion

X2B preregistered support for a depth-generalization advantage if the feature-matched GRU:

1. reached at least 80% IID depth-4 exactness; and
2. remained at least 10 percentage points below explicit transition at depth 48 or 96.

Observed gaps:

- depth 24: 9.03 percentage points
- depth 48: **10.33 percentage points**
- depth 96: **13.02 percentage points**

The criterion is met at both decisive long-depth suites.

## Final interpretation

The evidence no longer supports the broad statement that “explicit state beats generic neural networks.” A feature-matched GRU learns the contextual transition almost perfectly at training depth and remains reasonably strong through depth 96.

The supported claim is narrower:

> **When the sufficient computational state is explicit and the relevant sparse fields are directly available, forcing recurrence through that explicit state alone yields cleaner length extrapolation than adding persistent latent recurrent memory.**

In this benchmark, the explicit transition kernel is stateless apart from the world state it updates. The feature-matched GRU carries an additional hidden state. The GRU's small transition error accumulates as execution length grows; the stateless transition kernel remains exact.

This suggests that, for Markovian computation, hidden recurrent state can become an unnecessary second state system whose approximation error or drift harms long-horizon execution.

## What is still not established

CASM-X2 does not establish:

- general reasoning;
- natural-language planning;
- autonomous operator discovery;
- semantic state discovery from raw observations;
- learning without process supervision;
- that latent state is always harmful;
- that explicit state will remain superior when the true sufficient state is unknown or partially observable.

## Next experiment

The next decisive phase should reduce or remove teacher-forced intermediate-state supervision.

The central question becomes:

> Can the explicit computational state be discovered, maintained, and corrected when only weaker signals are available — final outcomes, demonstrations, consistency constraints, or verifier feedback?

That is a more appropriate place to reintroduce verification: not to rerank an already exact learned transition, but to provide credit assignment and repair when intermediate state is latent or only partially supervised.
