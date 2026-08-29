# CASM-R scoped-memory result

Status: completed controlled run, GitHub Actions run `33248747390`.

## Why this control existed

Earlier CASM-R training packed multiple independent problems into one token stream while episodic and persistent memory continued across the `SEP` boundary. Evaluation started every problem from clean memory. This branch removed that train/eval scope mismatch by using exactly one independent task per batch row and therefore one memory lifetime.

## Controlled comparison

Both deployed models have 1,347,361 parameters.

| Model | packed hard answer NLL | corrected 6-task hard answer NLL | corrected answer-byte acc |
|---|---:|---:|---:|
| scoped Q/K, 1 recurrent step | 1.41793 | 1.43848 | 44.21% |
| scoped Q/K, 3 shared recurrent steps | **1.32147** | **1.41274** | **45.07%** |

Relative to the previous unscoped 3-step run, corrected-task hard answer NLL improved from 1.48053 to 1.41274. Thus state/memory scoping improves probability quality.

## Primary falsifier: true autoregressive solve rate

It did **not** improve solving.

Across the ten free-generation stress groups (balanced graph easy/hard, four long state groups, four associative-memory groups):

- scoped 1-step mean exact solve rate: **12.33%**;
- scoped 3-step mean exact solve rate: **12.17%**;
- prior unscoped 1-step and 3-step runs: **15.33%** each.

Associative long-context exact generation remained 0% at 12/24/48/96 keys. State generation remained low. Both graph models generated `no` for every sampled graph, so their ~50% graph exact accuracy is class collapse, not graph reasoning.

## Recurrent depth

The trained 3-step model evaluated at 1/3/5 recurrent iterations shows a strong depth-distribution effect. Three steps substantially improve graph likelihood relative to forcing the checkpoint to one step, but five steps sharply degrade state/associative NLL. This is an overthinking / out-of-training-depth failure, not monotonic test-time-compute scaling.

## Benchmark defect discovered during audit

The legacy `state_tracking` and `state_long` generators initialized every object at a random location but did **not serialize those initial locations into the prompt**. If the queried object was never moved, its answer was impossible to infer from the input. For the easy 5-object/6-update generator this occurs with probability approximately `(4/5)^6 = 26.2%`.

Therefore old state solve-rate numbers must not be treated as pure reasoning measurements. The next curriculum explicitly serializes initial state.

## Decision

Do not add more memory machinery based on this result. Memory scope and recurrence improve likelihood but have not converted into actual problem solving.

Next experiment: keep the same deployed recurrent Q/K core but provide dense verifier-generated supervision for intermediate latent computation states. Judge that phase primarily by true autoregressive solve rate on verifier-complete tasks.
