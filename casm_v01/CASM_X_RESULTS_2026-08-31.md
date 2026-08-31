# CASM-X results — 2026-08-31

## Status

CASM-X v0 completed three independent CPU training/evaluation seeds from exact unified-harness-v2 base `8b60ddb8887f295b2fd576b02edb3594de9d490f`.

Evaluated seeds:

- train `20260831`, eval `20260911`
- train `20260832`, eval `20260912`
- train `20260833`, eval `20260913`

All contract tests passed. All three training/evaluation jobs passed. The `explicit_oracle_both` integrity ceiling remained exactly 100% on every IID, compositional, and stress suite.

## Final-state exact accuracy

| Suite | Explicit + verifier | Explicit, no verifier | Shared transition | GRU recurrent |
|---|---:|---:|---:|---:|
| IID depth 3 | 100.00% | 100.00% | 100.00% | 6.58% |
| held-out composition depth 4 | 99.93% | 100.00% | 100.00% | 3.06% |
| held-out composition depth 6 | 100.00% | 100.00% | 100.00% | 1.69% |
| stress depth 8 | 99.93% | 100.00% | 100.00% | 1.11% |
| stress depth 12 | 100.00% | 100.00% | 100.00% | 0.85% |

Values are arithmetic means across the three independent seeds. The two 99.93% cells each arise from a single 511/512 result in seed `20260831`; the other two seeds were 512/512.

Parameter counts:

- explicit operator machine: **246,153**
- shared-transition control: **246,160**
- GRU recurrent control: **175,936**

The explicit and shared-transition models are therefore effectively exactly parameter matched (7 parameters difference).

## Efficiency

Mean measured depth-12 CPU transition throughput across the three runners:

| Model | transitions/s |
|---|---:|
| Explicit + verifier | 28,621 |
| Explicit, no verifier | 100,681 |
| Shared transition | 158,906 |
| GRU recurrent | 313,852 |

Verifier reranking made the explicit model about **3.52x slower** on average than the same learned state machine without verifier reranking.

The shared-transition control was about **1.58x faster** than the explicit multi-operator implementation while matching or exceeding its exact accuracy.

## What survives

### 1. Explicit recurrent state transition is a successful mechanism in this controlled world

The state-transition models learned the local transition rule and executed it recursively without the depth collapse seen in the generic GRU configuration. Training was limited to depth 1–3, while both typed state-transition architectures retained 100% final-state exact accuracy at depth 12 and on programs containing withheld ordered operator bigrams.

This is evidence that exposing a persistent typed state and repeatedly applying a learned transition function can produce exact length extrapolation in this controlled Markov program world.

It is **not** yet evidence of general reasoning or autonomous operator discovery.

### 2. The modular operator hypothesis does not survive

The preregistered modularity falsifier is triggered.

A single shared learned transition network, with essentially the same parameter count as the eight-module explicit operator machine, scores 100% on every evaluated suite and is faster. Separate learned operator modules provide no demonstrated accuracy or extrapolation advantage here.

Therefore CASM-X v0 does **not** support the claim that reusable computation requires a bank of separately parameterized learned operators.

The stronger surviving hypothesis is narrower:

> reusable computation in this setting is induced by an explicit typed recurrent state-transition contract, not by operator modularity itself.

### 3. The verifier hypothesis does not survive

The no-verifier explicit model is exactly 100% across all three seeds and all headline suites. Verifier reranking is neutral on almost every cell and slightly harmful in two seed/suite cells, while reducing throughput by roughly 3.5x.

The preregistered verifier criterion is therefore also triggered: verifier-guided selection is unsupported in this version.

This does not show that verification is useless in harder worlds. It shows that verification adds no value when the learned transition is already effectively deterministic and exact.

## What this experiment does not establish

CASM-X v0 is deliberately simple enough to be a mechanism test, and its near-perfect state-transition results expose weaknesses in the benchmark itself.

Most importantly:

1. **Routing is too easy.** Each opaque command alias has a fixed one-to-one semantic meaning. Learning the router is therefore fixed-label classification, not context-dependent operator selection.
2. **The transition is strongly Markovian.** Each step receives the current full state, the command, source registers, and destination register. A universal transition network can learn the step function directly.
3. **The write surface is narrow.** Every instruction mutates one destination register only.
4. **Per-step target states are supplied during training.** This is strong process supervision and may be doing much of the work.
5. **The held-out-bigram split tests sequence recombination, not new operator semantics.** Once a correct local transition rule is learned, unseen ordering is expected to compose.
6. **The GRU control is not parameter matched.** Its poor depth extrapolation is informative but cannot by itself establish that typed state is superior at equal capacity and compute.

Accordingly, this result should be described as a successful controlled proof of learned recurrent state execution, not a proof of abstract operator induction.

## Decision for CASM-X2

The evidence changes the default architecture for the next phase.

Retain:

- explicit typed persistent working state;
- recurrent learned state transitions;
- local/sparse state updates;
- exact execution metrics and depth extrapolation.

Demote to ablations rather than defaults:

- separately parameterized operator modules;
- verifier reranking.

CASM-X2 should make the computational problem itself harder:

1. **Contextual routing:** the same command representation must map to different transitions depending on state/type/context.
2. **Operator induction:** semantics must be inferred from demonstrations or descriptions rather than a fixed alias table.
3. **Control flow:** conditional branches, loops, dynamic HALT, and variable execution length.
4. **Coupled writes:** operators may update multiple typed fields or scratch registers.
5. **State-scale transfer:** evaluate unseen state sizes/register counts where the implementation permits weight sharing.
6. **Supervision ablation:** reduce or remove intermediate target-state supervision.
7. **Equal-capacity baselines:** parameter/FLOP-match generic recurrent and Transformer controls to the ~246k state-transition models.
8. **Longer error-accumulation stress:** depth 24/48/96.
9. **Semantic OOD:** withhold operator semantics or operator families, not only ordered pairs.
10. **Compression reintegration only after the computational kernel is challenged:** use compression for state storage/retrieval/routing, where earlier CASM evidence actually supports it.

The next decisive question is therefore:

> Is explicit typed state itself the source of reusable computation, or can a generic equal-capacity model recover the same algorithm once state, supervision, and compute are properly matched?
