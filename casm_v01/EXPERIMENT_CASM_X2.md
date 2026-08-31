# CASM-X2 — Is explicit state the computational advantage?

## Motivation

CASM-X v0 established a narrower result than originally hypothesized. A learned typed recurrent transition model executed perfectly far beyond its training depth, but a single shared transition network matched the explicit operator bank and verifier reranking added no value.

CASM-X2 removes those unsupported mechanisms and isolates the surviving question:

> Does repeatedly updating an explicit predicted typed state produce better reusable computation than generic hidden-state sequence models when capacity, examples, process supervision, and instruction access are matched?

## Harder contextual machine

The world retains four registers with values `0..15`, but command semantics are no longer fixed labels.

There are four opaque command families. Each family corresponds to a pair of possible semantic operators. At each transition, the active operator is selected by a context bit derived from the **current machine state**:

```text
context_bit = parity(state[a] XOR state[b] XOR state[dst])
```

Therefore the same command token, with the same architectural embedding, can mean different computations under different states.

Operator pairs are:

1. `copy` / `add mod 16`
2. `sub mod 16` / `max`
3. `min` / `xor`
4. `inc mod 16` / `dec mod 16`

Semantic operator IDs are benchmark-private data. No trainable model receives them as inputs or auxiliary labels.

## Composition split

Training uses depths 1–4 and excludes four ordered command-family bigrams:

`(0,1), (1,2), (2,3), (3,0)`.

IID evaluation uses fresh depth-4 programs with the same exclusion.

Composition evaluation requires at least one withheld family bigram at depths 6 and 12.

Length extrapolation uses the same compositional condition at depths 24, 48, and 96.

## Models

### Explicit predicted-state model

The surviving shared-transition architecture from CASM-X is used as the explicit-state model.

At every step it:

1. receives its current predicted discrete register state;
2. receives the next opaque command plus source/destination registers;
3. predicts the destination value;
4. writes that prediction into the explicit state;
5. feeds the resulting predicted state into the next transition.

It has no operator bank and no verifier reranking.

### Parameter-matched GRU

A generic GRU carries computation only in its learned hidden state. It receives the identical initial state and instruction sequence and is trained on the identical per-step target states.

Width is selected so parameter count is within 5% of the explicit-state model.

### Parameter-matched causal Transformer

A causal Transformer receives an initial-state prefix token followed by the identical instruction sequence. It predicts the full register state after every instruction but does not feed those predicted states back as a typed execution state.

It is also parameter matched within 5%.

## Controlled supervision

All models receive:

- the same training programs;
- the same initial states;
- the same opaque command and register-argument inputs;
- the same target state after every training step.

Thus CASM-X2 does **not** yet test reduced process supervision. It isolates the state representation/execution bottleneck first.

## Primary metric

`final_state_exact`: all four registers must be exactly correct after the full program.

Secondary metrics:

- exact state at every intermediate step;
- per-register accuracy;
- semantic coverage of the generated evaluation set.

Token NLL is not a capability metric for this lane.

## Parameter-match gate

The run is invalid if either generic control is outside `[0.95, 1.05]` times the trainable parameter count of the explicit-state model.

## Preregistered interpretation

### Evidence for explicit typed state

The explicit-state hypothesis receives meaningful support only if, across seeds:

1. all models learn the training/IID regime sufficiently to make OOD comparison meaningful;
2. explicit state exceeds **both** matched generic controls by at least 10 percentage points in mean final-state exactness at depth 48 or 96;
3. the advantage grows or remains stable with execution depth rather than appearing only at one shallow suite.

### Falsifier

The strong explicit-state claim is weakened or falsified if either parameter-matched GRU or Transformer remains within 2 percentage points of the explicit-state model across depth 24/48/96.

If all three architectures fail similarly, the result is inconclusive about state representation and instead indicates that contextual transition learning or the training contract is inadequate.

### Important boundary

Even a positive result would establish only that explicit predicted state is a useful inductive bias for this controlled contextual state machine. It would not establish general reasoning, operator discovery, or natural-language planning.

## Next ablation if X2 survives

Only after this comparison should the program remove intermediate-state supervision and test whether explicit state can be learned from final outcomes, demonstrations, or verifier feedback alone.
