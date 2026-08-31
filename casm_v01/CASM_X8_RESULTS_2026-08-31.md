# CASM-X8 results — 2026-08-31

## Status

CASM-X8 tests whether the collision-free one-variable-per-slot topology observed in X7 still emerges when the learned binding itself is **not constrained to be injective**.

Preregistration commit, frozen before implementation and execution:

`80350109a46ef55a7640153217cf3e0dd8265751`

Final evaluated head:

`f56faf184cf81f56cbc541535a38b567d0ab9156`

Successful workflow run:

`33387854579`

Seeds:

- train `20261011`, eval `20261091`
- train `20261012`, eval `20261092`
- train `20261013`, eval `20261093`

All X8 integrity tests and all three 8,000-step train/evaluate/validator jobs passed on the final evaluated head.

## Artifact provenance

| Seed | Artifact ID | SHA-256 digest |
|---|---:|---|
| `20261011` | `9756857519` | `sha256:7195aa172dd326c9794e19588fa6e3ad499b5ee2e0822b68fff06abf825a3eff` |
| `20261012` | `9756867835` | `sha256:50efe0711f3c98229c926df9086a825b418fa53dff244b261cf56f3697afef6f` |
| `20261013` | `9756748252` | `sha256:338516d970725cbd2c25590dee4ace57362ebeb920763ee9bdd9da40b101e500` |

## Pre-training numerical gate failure

The first X8 integrity run failed **before any training was allowed to start**.

The failure was in the `learned_injective` positive control, not the unconstrained dense treatment. The exact 1,680-assignment convex mixture is mathematically column-capacity preserving, but float32 accumulation under a tied adversarial score case produced a column occupancy approximately `1.31e-6` above 1, just outside the frozen `1e-6` tolerance.

The training jobs were correctly skipped.

The repair was numerical only: injective assignment weights/marginals are accumulated in float64 and cast back. The scientific X8 question, topology family, optimizer, seeds, training budget, supervision contract and interpretation thresholds were not changed. The final successful run is therefore evaluated only from `f56faf184cf81f56cbc541535a38b567d0ab9156`.

## Experimental treatments

All treatments use the same explicit recurrent transition architecture, train depth 8, fixed final answer register `0`, no teacher forcing, no intermediate or hidden-state labels, no semantic-operator labels and no binding labels.

- `canonical_sparse`: fixed collision-free positive control.
- `learned_injective`: X7-style learned injective positive control.
- `learned_dense`: decisive treatment; each of the four binding rows is independently softmax-normalized across eight candidate slots. There is no injective constraint, column-capacity constraint, collision penalty, sparsity penalty, entropy penalty, identity prior or matching projection during training.
- `diffuse_dense`: fixed diffuse negative control.

For `learned_dense`, hard evaluation uses each row's independent argmax **without collision repair**. If multiple external variables select the same slot, the collision remains in the evaluated model.

## Result: strong emergent collision-free topology — PASS

The preregistered strong criterion required every seed to pass independently. Cross-seed averaging could not rescue a failed seed.

The `learned_dense` treatment achieved **100% exact capability on every seed and every evaluated suite**, both with its learned soft binding and after unrepaired hard row-argmax discretization.

| Criterion | Required | Worst observed across all seeds/suites |
|---|---:|---:|
| hard answer-final accuracy | >=99% | **100%** |
| deep hard step-state exactness | >=95% | **100%** |
| deep hard hidden-register accuracy | >=99% | **100%** |
| soft answer-final accuracy | >=99% | **100%** |
| deep soft step-state exactness | >=95% | **100%** |
| deep soft hidden-register accuracy | >=99% | **100%** |
| independent argmax unique slots | 4 | **4 on every seed** |
| independent argmax collisions | 0 | **0 on every seed** |
| mean row maximum | >=0.90 | **0.980335** |
| best injective score | >=3.60 / 4 | **3.921340 / 4** |

Training is limited to depth 8. Evaluation includes IID depth 8, held-out family-order composition depth 12 and 24, and stress depth 48 and 96.

## Per-seed learned dense topology

| Seed | Independent argmax | Unique slots | Collisions | Row-max mean | Best injective score | Max soft column occupancy |
|---|---|---:|---:|---:|---:|---:|
| `20261011` | `[5, 4, 7, 3]` | 4 | 0 | 0.980335 | 3.921340 | 1.001780 |
| `20261012` | `[3, 2, 0, 1]` | 4 | 0 | 0.983005 | 3.931995 | 1.000189 |
| `20261013` | `[6, 3, 0, 2]` | 4 | 0 | 0.991078 | 3.964313 | 0.999600 |

The slight soft column occupancies above 1 are allowed in the unconstrained treatment and are handled by the preregistered capacity-normalized transport. The decisive hard topology nevertheless contains zero collisions on every seed without any collision-repair step.

The different slot assignments across seeds are expected gauge freedom: no canonical slot identity is supervised.

## Negative control

The fixed diffuse treatment remains poor at depth 96. Three-seed means are approximately:

- answer-final accuracy: **10.33%**;
- step-state exactness: **2.21%**;
- hidden-register accuracy: **14.65%**.

This strengthens the conclusion that resolving external-variable identity matters in this benchmark. It also rules out the interpretation that the shared transition network can simply ignore binding topology and solve the task from globally mixed state.

## What X8 establishes

Within this supplied external ontology and explicit-state architecture, **the injective/all-different binding prior is not necessary** for this benchmark.

Four independently normalized binding rows, optimized only through one fixed terminal answer channel, spontaneously differentiate into four distinct internal slots on every seed. The resulting soft and discretized state machines recover the complete hidden trajectory exactly through 12x the training execution depth.

A bounded formulation is:

> Given known external variables and a shared explicit transition system, fixed-answer-only learning can self-organize an unconstrained dense variable-to-slot binding into a collision-free executable decomposition without an injective matching prior.

## What X8 does not establish

X8 is still not autonomous ontology discovery. Important structure remains supplied:

1. the external variable count is fixed at four;
2. there is one free learned binding row for each already-known external variable;
3. the candidate internal slot count is fixed at eight;
4. the value domain and `EMPTY` symbol are supplied;
5. command, source-argument and destination external identities are supplied;
6. the model is told that state should be represented through explicit recurrent slots;
7. the local transition architecture is supplied;
8. active-variable cardinality is not inferred from raw observations.

Therefore X8 demonstrates **emergent collision-free topology inside a supplied variable ontology**, not discovery of the ontology itself.

## Decision for X9

The next clean prior to remove is the fixed one-row-per-known-variable parameterization.

X9 should train one **shared binding generator** across worlds with 2, 3 and 4 active external variables and evaluate unseen 5- and 6-variable worlds. The generator must derive each row from a deterministic, non-learned variable descriptor and supplied active cardinality; it must not contain a free embedding or binding-logit row for each possible external variable.

The decisive question becomes:

> Can the learned state-binding rule itself extrapolate to variables and cardinalities never seen during training while preserving answer-only hidden execution?

A positive X9 would still not infer the number or identity of variables from raw input; that would be a subsequent ontology-selection experiment.