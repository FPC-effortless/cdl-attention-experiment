# CASM Phase 1 — typed Boolean DAG falsification

This repository keeps the existing Conditional Description-Length experiment unchanged. CASM is implemented as a separate research track under `casm/`.

## Phase 1 hypothesis

A low-parameter structural router can select the reusable computation needed for a typed Boolean DAG from a fixed admissible structural substrate, without reading runtime values.

The intended factorization is:

`W_t = A ⊙ S(α) ⊙ G_t`

with:

- `A`: fixed candidate substrate;
- `S(α)`: persistent relation-level computational strength;
- `G_t`: dynamic relevance gate;
- router input: structural information only;
- runtime values: consumed only by the executor.

The hard architectural separation is `∂G_t / ∂x = 0`.

## What is implemented first

The first implementation does **not** train the router yet. It establishes whether the benchmark contains a real routing problem:

1. generate a topologically ordered Boolean DAG;
2. generate a genuine candidate-edge superset `A`;
3. generate the episode-specific true edge set `E* ⊂ A`;
4. exhaustively evaluate the oracle on the full Boolean input domain;
5. run the copy-mask control `g_ij = m_i m_j`;
6. exhaustively test small graphs for minimality and same-size alternatives.

This ordering is intentional. If the copy-mask control is already the oracle, CASM has not demonstrated routing; the benchmark must be redesigned before model training.

## Grammar

Operations are `AND`, `OR`, `XOR`, and `NOT`, with maximum true-program fan-in two. Input nodes precede operation nodes, so execution is a single topological pass. Recurrent settling and spectral-radius analysis are explicitly deferred to Phase 2.

## Candidate substrate versus program

Node existence and edge selection are separate variables. Every operation port may receive an edge from an earlier node in the fixed candidate substrate. The sampled program chooses exactly the operation arity. Therefore a trivial existence mask cannot reconstruct the true graph.

The strict executor rejects a graph with the wrong number of selected inputs rather than silently changing primitive semantics.

## Oracle standard

For small candidate substrates, the oracle is evaluated over all Boolean input assignments. The minimality check searches every candidate subgraph smaller than the sampled graph and then every same-size alternative. A claim of `unique_minimal` is emitted only when this exhaustive search finds no alternative.

For larger graphs, the code returns `checked=False`; it does not convert a local edge-removal result into a global minimality claim.

## Next implementation gate

Only after the preflight passes should the repository add the CASM-S router:

- node-wise structural embeddings;
- factorized source/target compatibility;
- syntactic relation bias `b_r`;
- soft sigmoid gates `g_ij = sigmoid(ell_ij / T)`;
- relation-indexed `alpha_r = c * softplus(eta_r)`;
- no edge-specific parameter table;
- batch-level budget loss;
- structural-sensitivity Gate 3 before enabling the budget term.

The next router must therefore be tested against the copy-mask control, not merely against the oracle labels.
