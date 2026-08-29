# CASM-U v0 result — causal answer-utility Q/K

Run: `33237643397`

Head: `e7ef959dd46f27ef43bc7d917dd12d092b78870f`

Seed: `20261011`

All models: 1,347,361 parameters, 1,000 training steps, corrected balanced graph curriculum, identical compatible initialization.

## Compared objectives

- `qk-memory`: ordinary normalized Q/K memory routing.
- `compression-qk`: Q/K trained with the prior future-byte compression objective.
- `answer-utility-qk`: Q/K trained with detached counterfactual reduction in actual answer-token NLL, using only memories available before the current chunk.

All three deploy the same cheap Q/K memory mechanism; the utility evaluator is training-only.

## Packed hard evaluation

| model | answer NLL ↓ | token accuracy ↑ |
|---|---:|---:|
| qk-memory | **1.97252** | 0.69899 |
| compression-qk | 2.02305 | **0.69992** |
| answer-utility-qk | 1.98518 | 0.69676 |

## Corrected heterogeneous hard evaluation

| model | answer NLL ↓ | answer-byte acc ↑ | teacher-forced exact ↑ |
|---|---:|---:|---:|
| qk-memory | **1.58741** | 0.40021 | 0.12000 |
| compression-qk | 1.60432 | 0.39588 | 0.11833 |
| answer-utility-qk | 1.58965 | **0.40418** | 0.12000 |

`answer-utility-qk - qk-memory = +0.00224` nats overall, i.e. effectively tied/slightly worse rather than a general improvement.

Per-task hard answer-NLL delta for answer-utility Q/K minus ordinary Q/K:

| task | Δ NLL (negative is better) |
|---|---:|
| associative recall | -0.00070 |
| state tracking | +0.00378 |
| arithmetic | +0.00324 |
| rule induction | -0.00968 |
| balanced graph reachability | -0.00678 |
| reverse/copy | +0.02356 |

The objective therefore does not produce a broad task-level advantage.

## Long-horizon stress

Answer utility gives small NLL improvements on associative recall at 12/24/48/96 keys (`-0.0062`, `-0.0062`, `-0.0107`, `-0.0068` vs Q/K), but state-tracking effects are mixed and mostly negative beyond 12 updates.

Future-byte compression remains locally useful on some retrieval-style probes but is inconsistent across state horizons.

## True autoregressive solve rate

The free-generation probe does not show an answer-utility advantage.

Balanced graphs remain unsolved: all three models collapse to predicting `no`, yielding the negative-class frequency as apparent accuracy rather than graph reasoning.

Long state tracking remains low (roughly 3–17% depending on horizon) and associative recall is almost entirely unsolved. Answer utility does not improve these solve rates over ordinary Q/K.

## Teacher diagnostics

The answer-utility signal is non-zero but weakly discriminative between candidates.

At step 1000:

- mean candidate utility gain: ~0.326 nats;
- mean within-position gain standard deviation: ~0.100 nats;
- router entropy: ~1.856 nats.

Earlier in training the gain standard deviation is often only ~0.02–0.03 nats. The target therefore carries substantially less candidate contrast than its positive mean gain suggests.

## Interpretation

CASM-U v0 falsifies the simple claim that **single-memory answer-NLL reduction, supervised only at teacher-forced answer bytes, is sufficient to improve general memory routing**.

The architecture currently performs only one memory retrieval/update before each prediction. This is a structural limitation for compositional tasks. A graph path or a sequence of state updates may require multiple retrieve-update cycles, while the utility teacher only evaluates one candidate injected into one local hidden state.

The next experiment should therefore change computation, not merely increase the auxiliary loss weight:

1. parameter-shared recurrent memory reasoning (`retrieve → update working state → retrieve`);
2. a one-step recurrent Q/K control to isolate extra compute;
3. a multi-step Q/K model with identical parameters but increased recurrent depth;
4. a set-conditioned utility target based on leave-one-memory-out effect after the full recurrent reasoning loop;
5. true autoregressive solve rate as the primary capability gate.

No claim of a general compression/utility-routing advantage is retained from CASM-U v0.
