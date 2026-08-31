# CASM-X6R — Valid probability-preserving learned binding rerun

## Status and reason for rerun

CASM-X6 v0 is invalidated in `CASM_X6_V0_INVALID_2026-08-31.md`. Its finite 12-step alternating Sinkhorn transform did not remain doubly stochastic after the binding became sharp. Because the same matrix was used for external→internal transport and internal→external decoding, row-sum drift allowed decoded categorical mass above 1 and produced negative nominal negative-log-likelihood values.

X6R reruns the **same scientific question and the same preregistered success thresholds** after replacing that invalid probability transform. X6 v0 metrics are not used to choose X6R success criteria.

## Question

> Can fixed-answer-only supervision learn a stable, executable binding between external register identities and unlabeled internal state slots while preserving long-horizon hidden-state execution?

As in X6, absolute internal slot labels are a gauge symmetry. A consistent non-identity permutation is fully valid.

## Single validity correction

The learned condition retains a trainable 4×4 external-register/slot score matrix initialized from small zero-centered noise.

Instead of approximate finite Sinkhorn normalization, X6R constructs an exactly normalized point in the Birkhoff polytope:

1. enumerate all 24 one-to-one permutations of four slots;
2. score each permutation by summing the four selected entries from the learned 4×4 score matrix;
3. softmax the 24 scores using the fixed binding temperature;
4. return the convex weighted sum of the 24 permutation matrices.

Every permutation matrix has exactly one unit entry in each row and column. Therefore their convex mixture is doubly stochastic by construction. The model still has only the same 16 learned binding scores and receives no binding labels, identity target, entropy penalty, permutation target or other binding regularizer.

This is a numerical-validity correction to the same supplied one-to-one binding prior, not a new scientific success criterion.

## Regimes

Identical to X6:

1. `canonical_binding` — fixed identity binding positive control;
2. `learned_binding` — learned exact Birkhoff-mixture binding from near-uniform initialization;
3. `diffuse_binding` — fixed uniform 1/4 binding negative control.

All three use identical transition initialization and the same examples.

## Supervision and architecture

Unchanged from X6/X5:

- four external registers and four internal slots;
- values 0..15;
- contextual command semantics;
- train depth 8;
- only final external register 0 is a loss target;
- registers 1–3 are never target labels;
- no teacher forcing;
- no intermediate-state targets;
- no semantic-operator labels;
- all register-specific access passes through the binding;
- no direct external-register embedding bypass;
- own predicted state is rolled forward.

## Budget

Exactly the X6 budget and seeds are reused to isolate the normalization correction:

- 6,000 optimization steps;
- batch size 128;
- AdamW, learning rate 2e-3, weight decay 1e-4;
- binding temperature 1.0;
- seeds `20260911`, `20260912`, `20260913`;
- evaluation seeds `20260991`, `20260992`, `20260993`;
- IID depth 8;
- held-out composition depths 12 and 24;
- stress depths 48 and 96.

Reusing the same seeds is deliberate: X6 v0 is invalid, and X6R is a controlled validity repair rather than a search over seeds.

## Required integrity contract

Before training, tests must establish:

1. near-uniform learned initialization;
2. row and column sums within `1e-6` for ordinary initialization;
3. row and column sums within `1e-6` for adversarially sharp/conflicting score matrices;
4. every initial internal-slot categorical distribution sums to 1 within `1e-6`;
5. every decoded external-register categorical distribution sums to 1 within `1e-6`;
6. fixed-answer loss is finite and non-negative;
7. after at least one optimizer step, conditions 2–6 still hold;
8. answer-only loss ignores all intermediate targets, hidden final targets and private semantic labels;
9. gradients reach both learned binding scores and transition parameters;
10. discrete projected binding is one-to-one.

The workflow must also validate final row/column mass and non-negative finite training losses from the result artifact.

## Preregistered interpretation

The thresholds are unchanged from X6.

### Control prerequisite

On **every seed**, `canonical_binding` must reach:

- ≥99% answer-final accuracy on every suite;
- ≥95% full step-state exactness at depths 24, 48 and 96;
- ≥99% hidden-register accuracy at depths 24, 48 and 96.

Otherwise no learned-binding claim is made.

### Strong learned executable binding

If the control prerequisite passes, `learned_binding` supports the strong result only if, on **every seed**:

- answer-final accuracy ≥99% on every suite;
- full step-state exactness ≥95% at depths 24, 48 and 96 under projected discrete binding;
- hidden-register accuracy ≥99% at depths 24, 48 and 96;
- mean soft row maximum ≥0.90;
- mean soft column maximum ≥0.90;
- best-permutation score ≥3.60 / 4.00.

### Computation without discrete identification

If decoded computation meets the accuracy thresholds but binding sharpness fails, the result supports distributed binding rather than discrete alignment.

### Answer without hidden execution

If answer-final accuracy reaches ≥95% but step-state exactness or hidden-register accuracy is <80% at depth 24 or deeper, answer-only supervision has not identified the hidden executable state.

### Binding optimization failure

If the canonical control passes but learned-binding IID answer-final accuracy is <80%, this setup fails to optimize the binding; it does not establish impossibility.

## Claim boundary

Even a strong X6R result would demonstrate binding learning only **inside a supplied state ontology**. Slot count, categorical value domain, command/argument/destination interface, local transition form and explicit recurrent state are still supplied.