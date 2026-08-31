# CASM-X — Explicit reusable computation

## Why this experiment exists

The unified CASM benchmark v2 materially weakened the earlier compression-attention interpretation. Compression and scoped memory improve probability quality, but the best current architecture solves only a small number of genuinely non-graph hard examples and graph reachability still admits a constant-class shortcut.

CASM-X changes the experimental variable. It does **not** ask whether another attention or routing loss improves latent representations. It asks whether making the computational object explicit improves reusable computation.

## Hypothesis

> A model with typed working state, reusable learned operator modules, explicit recurrent state transitions, operator routing, and an independent transition verifier will generalize to unseen operator compositions and longer execution depths better than equally supervised generic recurrent baselines.

This is an architectural claim, not a claim that the current implementation is already a general reasoning system.

## Controlled world

The world contains four typed registers with values in `0..15` and eight semantic operators:

1. `copy`
2. `add mod 16`
3. `sub mod 16`
4. `max`
5. `min`
6. `xor`
7. `inc mod 16`
8. `dec mod 16`

The model does not receive semantic operator ids. Commands use a fixed opaque alias permutation. Semantic ids are retained privately by the benchmark for supervision, diagnostics, and oracle ablations.

Each instruction also identifies source registers `a`, `b`, and destination register `dst`. Every operator changes only the destination register. This makes the execution state and transition boundary explicit without hand-coding the learned operator implementation.

## Composition split

Training programs have depth 1–3.

Eight ordered operator bigrams are withheld from the training distribution:

`(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,0)`.

The IID evaluation uses fresh depth 1–3 programs that also exclude those bigrams.

The compositional evaluation uses depth 4 and 6 programs and **requires** at least one withheld bigram.

Depth extrapolation stresses the same condition at depth 8 and 12.

The primary metric is exact correctness of the complete final register state. This avoids the class-prior shortcut that contaminated the graph aggregate in the previous benchmark.

## Models

### Full CASM-X

`ExplicitOperatorMachine`

- explicit register/value state;
- opaque command encoder;
- learned operator router;
- eight independently parameterized learned operator modules;
- recurrent discrete state update;
- independent learned transition verifier;
- verifier can rerank operator candidates at execution time.

The model receives per-step state supervision during training.

### Shared-transition ablation

`SharedTransitionModel`

Keeps the same typed state and recurrent execution contract but collapses all eight operator modules into one universal transition network conditioned on the command.

This is the key modularity control. If it matches CASM-X, explicit reusable operator modules are not providing evidence of an advantage.

### Generic recurrent baseline

`GRUProgramBaseline`

Receives the same initial state, opaque commands, arguments, and per-step state supervision but carries computation in an ordinary recurrent hidden state. It has no explicit state-transition bottleneck and no modular operator library.

## Oracle diagnostics

CASM-X records four execution modes in addition to the normal model:

- `explicit_no_verifier`: learned router + learned operators, verifier disabled;
- `explicit_oracle_routing`: true operator route + learned operator execution;
- `explicit_oracle_execution`: learned route + deterministic true operator execution;
- `explicit_oracle_both`: true route + deterministic true execution.

`explicit_oracle_both` must score exactly 1.0. Anything else is an evaluator/data-integrity failure.

The two partial oracles localize failure:

- high oracle-routing performance with low full performance -> routing is the bottleneck;
- high oracle-execution performance with low full performance -> learned operator execution is the bottleneck;
- both partial oracles low -> both routing and execution are failing or the state contract is inadequate.

## Pre-registered interpretation

A strong positive result requires more than IID accuracy.

Across the multi-seed run, CASM-X should:

1. preserve high final-state exactness on withheld depth-4/6 compositions;
2. exceed both the shared-transition and generic recurrent baselines on depth-6 compositional exactness by a meaningful margin;
3. retain non-trivial exactness at depth 8/12 rather than collapsing immediately beyond the training depth;
4. show that any verifier claim is real by outperforming the same model with verifier disabled;
5. keep the oracle-both integrity ceiling at 100%.

The modularity hypothesis is weakened or falsified if the shared-transition model matches or beats CASM-X on withheld compositions and depth extrapolation at comparable parameter scale.

The verifier hypothesis is unsupported if verifier reranking is neutral or harmful.

The reusable-computation claim is unsupported if all learned models collapse as soon as program depth exceeds three, even if training/IID losses are low.

## Metrics

Every suite records:

- `final_state_exact` — **primary**;
- `step_state_exact`;
- per-register accuracy;
- router accuracy for CASM-X;
- parameter counts;
- rollout throughput.

The experiment intentionally does not use token NLL as the main capability metric. The preceding CASM experiments showed that NLL can improve materially without producing reliable algorithmic solves.

## Execution

```bash
cd casm_v01
PYTHONPATH=. python -m casm.run_explicit_compute \
  --steps 1800 \
  --batch-size 128 \
  --eval-n 512 \
  --output casm-x-output/results.json
```

GitHub Actions runs three independent seeds and retains JSON results and checkpoints for cross-seed analysis.
