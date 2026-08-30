# CASM-Bench v1.0 — canonical benchmark contract

Status: **frozen after integrity validation**. Any change to cases, seeds, task generators, answer normalization, primary metrics, or suite sizes requires a new benchmark version. Do not edit v1.0 in place to improve a model score.

## What we are measuring

CASM-Bench v1 is a **mechanism benchmark for small recurrent/memory models**, not a claim of general natural-language intelligence. It measures six capabilities with equal task weight:

1. **Associative retrieval** — retrieve the correct value from a distractor-filled memory set.
2. **State tracking** — maintain and update explicit world state through multiple guaranteed mutations of the queried object.
3. **Algorithmic arithmetic** — execute exact multi-step symbolic computation.
4. **Rule induction** — infer and continue a latent transformation rule.
5. **Graph reachability** — perform multi-hop relational reasoning with exact class balance and matched counterfactual positive/negative pairs.
6. **Exact sequence transformation** — preserve and manipulate token-level structure via reverse/copy.

These axes separately stress memory addressing, persistent state, computation, abstraction, compositional graph reasoning, and exact symbolic manipulation. They remain fixed for CASM-Bench v1.

## Suites

### DEV-CORE
- 60 fixed unique cases per task; 360 total.
- Used after **every experimental interaction**.
- This is the suite we are allowed to optimize architecture against directly.

### DEV-OOD
- 60 fixed unique cases per task; 360 total.
- Used after every experimental interaction.
- Structurally disjoint extrapolation domains: longer memories/state sequences, larger graphs, longer transforms, larger/deeper arithmetic, and held-out rule coefficients.

### HOLDOUT-CORE
- 200 fixed unique cases per task; 1,200 total.
- Promotion-only. Do not run after every tweak.

### HOLDOUT-OOD
- 200 fixed unique cases per task; 1,200 total.
- Promotion-only and structurally disjoint from the core domain.

The DEV/HOLDOUT split exists because repeatedly inspecting a test set makes the research process itself adapt to the test set. A model family that is changed after seeing HOLDOUT no longer has an untouched HOLDOUT result; a new model family/version must be declared before another promotion evaluation.

## Primary score

**Primary behavior is greedy autoregressive answer generation with no teacher forcing and no oracle answer length.** Generation begins after the literal `answer ` marker and stops on EOS/SEP/newline or a fixed 64-byte cap.

For every task we report:

- `raw_exact`: task-aware semantic exact accuracy;
- `majority_baseline`: accuracy of always predicting the most frequent gold answer in that fixed task corpus;
- `adjusted_exact = (raw_exact - majority_baseline) / (1 - majority_baseline)`.

The headline within a suite is:

`NormalizedSolveMacro = mean(adjusted_exact across the six tasks)`.

We also report `RawSolveMacro` for interpretability. Majority adjustment is mandatory because otherwise a collapsed graph model that always predicts one class scores 50% and can dominate a macro average despite learning no reachability.

**The canonical model report always contains both DEV-CORE and DEV-OOD NormalizedSolveMacro. Do not collapse them into one number when making architecture decisions.** If a single display number is required, use their arithmetic mean and always show the two components beside it.

## Diagnostic metrics — never the headline

Teacher-forced answer NLL, teacher-forced byte accuracy, and teacher-forced exact match are diagnostics of probability quality/calibration only. They feed previous gold answer bytes into later answer positions and therefore are **not solve-rate metrics**.

Also report, when applicable:
- parameter count;
- recurrent reasoning steps;
- evaluation wall time / throughput on the same runner;
- memory-zero/shuffle causal ablations;
- 1/3/5-step test-time depth sweep for recurrent models;
- training wall time and training-token count.

Do not combine efficiency diagnostics into the capability score. Compare efficiency/capability as a Pareto tradeoff.

## Leakage and validity audit

The following issues were found in earlier CASM evaluations and are explicitly prohibited in v1:

| Risk | v1 rule |
|---|---|
| Graph generator historically always answered `yes` | Graph cases are verifier-checked and **exactly 50/50 yes/no** in every suite. |
| Positive and negative graphs came from visibly different generator families | Graphs are generated as **matched counterfactual pairs**: same source, destination, node count, edge count, path scaffold and shared distractors. The negative removes one internal path edge and replaces it with one non-path edge; both variants are independently verified. |
| State generator sampled hidden initial state | Every state prompt explicitly serializes all initial object locations. |
| State query could require no tracking | The queried object undergoes at least 3 real state changes in CORE and 6 in OOD; no-op moves are forbidden. |
| Teacher-forced exact treated as solving | Teacher-forced metrics are diagnostic only. Primary is free generation. |
| Gold answer length used during generation | Forbidden. Fixed termination protocol and cap only. |
| Different seeds assumed to guarantee separation | Forbidden. HOLDOUT explicitly excludes any DEV prompt hash. |
| Duplicate synthetic prompts | Generator resamples until every prompt in a suite is unique. |
| One RNG stream shared across task types | Each task has an independent deterministic RNG stream. |
| Python `set` iteration changed graph prompts across processes | Edges are sorted before deterministic shuffling; frozen suite SHA256 digests are tested. |
| Repeated test inspection/adaptive leakage | DEV is iterative; HOLDOUT is promotion-only. |
| Training/test exact overlap | Every future training run must emit SHA256 hashes of visible problem prefixes; official HOLDOUT results must report exact overlap count. |
| Process traces or verifier state visible at test time | Forbidden. Benchmark cases contain only task prompt and answer; training-only traces never enter evaluation input. |
| Same-distribution success misreported as generalization | CORE and structurally disjoint OOD scores are always reported separately. |

## Exact contamination certification

Training code must record one SHA256 hash per generated problem prefix, where the prefix ends at `answer ` and excludes gold answer bytes. The canonical evaluator compares this log with the benchmark prompt hashes.

An official HOLDOUT run is `certified_clean=true` only when exact prompt overlap is zero. Historical runs without training hashes may be reported as **uncertified historical results**, never as contamination-certified results.

Exact-hash checks do not prove absence of structural/template contamination. That is why OOD domains are independently defined and held fixed.

## OOD contract

OOD is not merely a new random seed.

- associative: 24 keys vs 12 core;
- state: 24 updates, at least 6 queried-object transitions, and larger object/location sets vs 12 updates and at least 3 queried transitions in core;
- arithmetic: four 3-digit operands and deeper operator patterns vs three 2-digit operands;
- rule induction: multipliers {4,5} and addends 11–20 vs core multipliers {2,3};
- graph: 14 nodes and minimum verified path depth 6 vs 10 nodes and minimum path depth 4;
- reverse: 28 symbols vs 14.

If future training deliberately includes these OOD domains, CASM-Bench v1 OOD can no longer support an extrapolation claim for that run; the training-domain declaration must state this and a new holdout domain is required.

## Comparison protocol for every future experiment

Every experimental interaction must state the same scoreboard in this order:

1. DEV-CORE `NormalizedSolveMacro` and `RawSolveMacro`.
2. DEV-OOD `NormalizedSolveMacro` and `RawSolveMacro`.
3. Six per-task raw exact accuracies for CORE and OOD.
4. Majority/class-collapse diagnostics, including graph accuracy by class.
5. Teacher-forced answer NLL as diagnostic only.
6. Parameter count, reasoning steps and inference cost.
7. Relevant causal ablations.
8. Training seed and benchmark version/digests.

No statement such as “better”, “wins”, “reasoning improved”, or “more efficient” should be made from NLL alone.

## Promotion rules

A one-seed experiment is **directional evidence only**.

A model becomes a promotion candidate only after at least three independent training seeds on DEV and must:

- improve mean DEV-CORE NormalizedSolveMacro;
- not reduce mean DEV-OOD NormalizedSolveMacro;
- show the primary improvement in at least 2/3 seeds;
- not hide a >5 percentage-point raw exact regression on any task without explicitly treating it as a tradeoff;
- preserve or explicitly price any increase in inference compute.

Only then run HOLDOUT-CORE and HOLDOUT-OOD. Official holdout claims must include training-overlap certification and per-seed results.

## Versioning rule

The benchmark implementation stores deterministic suite digests. If an integrity test or scientific review requires changing v1.0, create v1.1/v2.0 and continue reporting v1.0 for historical comparability where possible. Never rewrite the frozen digests in order to accommodate a changed generator silently.
