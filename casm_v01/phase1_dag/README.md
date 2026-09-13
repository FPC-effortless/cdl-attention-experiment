# CASM Phase 1 DAG falsification

This track is deliberately separate from the existing CASM v0.1 language-model experiments.

## Locked setup

- typed Boolean primitives: `AND`, `OR`, `XOR`, `NOT`;
- true-program fan-in cap: 2;
- single-pass topological execution;
- no recurrent settling or spectral-radius calibration;
- structural routing must be value-independent (`∂G/∂x = 0`);
- candidate substrate `A` is a strict superset of episode wiring `E*`;
- node existence is separate from edge selection;
- small instances use exhaustive Boolean truth tables;
- minimality claims require exhaustive subgraph enumeration.

## First falsification control

`g_ij = m_i m_j` is evaluated before router training. Because all generated nodes exist and `A` contains distractor edges, this control sets every admissible edge active. The strict executor rejects that graph when a node receives more than its primitive arity.

If a future benchmark changes execution semantics so that the copy-mask control reaches the task ceiling without learning wiring, the benchmark is non-discriminative and must be repaired before CASM-S is trained.

## Next gate

Only after this preflight is mechanically valid should the CASM-S factorized router be added: structural node embeddings, bilinear source/target compatibility, relation bias, sigmoid gates at `T=2`, relation-indexed `alpha_r = c·softplus(eta_r)`, and the batch-level budget loss.
