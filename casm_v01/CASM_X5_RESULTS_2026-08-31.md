# CASM-X5 results — 2026-08-31

## Executive result

CASM-X5 passes the preregistered strong hidden-trajectory-recovery criterion.

> **For this controlled contextual Markov program world, a shared explicit-state transition can recover an essentially exact hidden working-state trajectory even when training exposes only the final value of one fixed answer register and the other three registers are never terminal loss targets.**

This is stronger than CASM-X4 because terminal supervision no longer rotates across the state variables.

## Execution

Workflow run: `33379918544`

Exact evaluated head: `cabfaa5660d14c09994af57c659189fbf2ace227`

Three train/eval seed pairs:

- `20260901` / `20260981`
- `20260902` / `20260982`
- `20260903` / `20260983`

The integrity contract passed before training. All three training/evaluation jobs and artifact validations passed.

## Contract

All three regimes use the same `SoftExplicitTransitionModel(d_model=96)` and start from identical parameters.

- no teacher forcing;
- no semantic-operator labels;
- no intermediate state targets;
- own differentiable predicted state is rolled forward during training;
- train depth 8;
- 4,000 optimizer steps;
- batch size 128.

Regimes:

- `full_final`: all four final registers;
- `random_register`: one uniformly sampled final register per example, reproducing the X4 positive control;
- `fixed_register`: only final register `0` on every example.

In `fixed_register`, registers `1`, `2`, and `3` are never loss targets. Integrity tests verify that changing every intermediate target and all three hidden final registers leaves the fixed-register loss exactly unchanged, while changing final register `0` changes the loss.

## Controls

`full_final` and `random_register` remain exact across every seed and suite, reproducing X3/X4.

Random-query counts remain balanced across the X4-style control: each register receives approximately 128k terminal observations per seed over 512k training examples.

## Fixed-register result

### Mean across three seeds

| Suite | Answer final | Full final exact | Full step exact | Hidden-register accuracy | Hidden-final exact |
|---|---:|---:|---:|---:|---:|
| IID depth 8 | **100.0000%** | 99.9132% | 99.9566% | 99.9964% | 99.9132% |
| composition depth 12 | **100.0000%** | 100.0000% | 99.9711% | 99.9904% | 100.0000% |
| composition depth 24 | **100.0000%** | 100.0000% | 99.9819% | **100.0000%** | 100.0000% |
| stress depth 48 | **100.0000%** | 100.0000% | 99.9855% | **100.0000%** | 100.0000% |
| stress depth 96 | **100.0000%** | 100.0000% | **99.9973%** | **99.9991%** | 100.0000% |

### Per-seed stress-depth-96 fixed-register result

| Seed | Answer final | Full final exact | Full step exact | Hidden-register accuracy | Hidden-final exact |
|---|---:|---:|---:|---:|---:|
| 20260901 | 100.0000% | 100.0000% | 99.9919% | 99.9973% | 100.0000% |
| 20260902 | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 100.0000% |
| 20260903 | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 100.0000% |

The small non-zero error on seed `20260901` occurs mainly at shallower evaluation steps; it disappears at the final state for depth 12 and deeper and is negligible relative to the preregistered 95%/99% thresholds.

## Optimization cost

The fixed-answer channel is somewhat harder to optimize than dense/full terminal supervision, but not catastrophically so.

First logged checkpoint below loss `0.001`:

- seed 20260901: full 2500, random 2400, fixed 3800;
- seed 20260902: full 3700, random 3600, fixed 3000;
- seed 20260903: full 1800, random 2900, fixed did not cross `0.001` at a logged checkpoint, ending at `0.00169`.

Despite this optimization variance, the fixed-answer model generalizes essentially exactly on all three seeds.

## Interpretation

The X4 positive result cannot be explained solely by distributing terminal labels over all four registers. X5 removes that coverage and still recovers the complete hidden trajectory.

The most plausible mechanism supported by this benchmark is:

1. the architecture supplies a typed explicit state and a shared local transition;
2. only one persistent answer channel is supervised;
3. gradients propagate backward through the differentiable recurrent state updates that causally contribute to that answer;
4. because transition parameters are shared across registers, commands and repeated program contexts, learning the answer-relevant transition law constrains behavior outside the directly supervised register;
5. the learned local transition then executes correctly on the unobserved registers and at much greater depths.

This is evidence for **implicit recovery of hidden computational state from answer-only supervision under a strong structural prior**.

## What this does not establish

CASM-X5 still does not discover its own state representation.

The model is given:

- the existence of four registers;
- the value domain;
- register identities;
- command and argument identities;
- destination register identity;
- direct sparse access to source/destination values;
- the rule that computation proceeds by repeatedly updating an explicit categorical state with one shared transition network.

Therefore the result should not be described as autonomous latent-program or state-schema discovery.

## Updated research boundary

The supervision bottleneck has now been pushed substantially farther:

- dense process supervision is unnecessary (X3);
- complete terminal-state supervision is unnecessary (X4);
- distributed terminal coverage across state variables is unnecessary (X5).

The dominant supplied structure is now the **state ontology and transition interface themselves**.

The next decisive experiment should weaken that supplied ontology rather than further reducing labels inside the same four-register interface. A useful next step is to hide register identity/slot correspondence behind a learned permutation or latent slot interface, then test whether answer-only supervision can recover a stable executable state alignment before attempting fully free state-schema discovery.
