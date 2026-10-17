# CASM Phase 1 — Boolean DAG

This directory is the standalone CASM Phase-1 harness. It is separate from Veritas.

## Frozen execution contract

- single-pass topological execution; no recurrent spectral-radius diagnostic;
- fixed upper-triangular candidate substrate `A` shared across episodes;
- variable-size programs represented by an explicit hard existence mask;
- true wiring is one of many admissible candidate wirings, so `m_i m_j A_ij` is not the oracle;
- the router sees only the **public program representation** (`parent_slots`, node type, depth, position, arity) plus candidate-edge descriptors; runtime values never enter routing;
- CASM-S uses factorized node-wise query/key compatibility, with no op-pair lookup table;
- `alpha_eta.shape == (2,)`: one reusable computation-strength parameter per syntactic argument port; Phase 1 initializes effective alpha exactly at one and freezes it to isolate routing;
- raw operand order is retained for evaluation; commutative canonicalization is bookkeeping-only;
- copy-mask is a mandatory oracle-wiring preflight, not a learned baseline;
- first learned comparison is Static Mask vs CASM-S; other router variants are deferred.

## Important scope

Phase 1 does **not** test whether CASM discovers a hidden program topology from node-local statistics alone. The program topology is part of the public program input, represented independently from runtime values. The experiment tests whether a factorized structural router can translate that program representation into conditional execution over a larger admissible physical substrate, including held-out role combinations.

## Gates

0. finite-depth numerical viability and gate initialization
1. gate intervention changes the forward result
2. router parameters receive task gradient
3. structural program changes routing
4. severing a true edge changes task loss
5. held-out `NOT -> XOR` role-pair routing separates true from distractor edges
6. topology/parameterization integrity

Gates 0, 1, 2 and 6 are implementation invariants. Gates 3–5 are empirical hypothesis tests and are reported rather than silently treated as infrastructure failures.

## Runner

From repository root:

```bash
python -m casm_v01.phase1_dag.runner --train-size 128 --test-size 64
```

The runner first executes the copy-mask oracle preflight, then trains Static Mask and CASM-S on the identical episode split. The OOD split excludes `NOT -> XOR` from training and requires it in test episodes. The training budget is derived from the empirical true-edge density of the training distribution rather than an arbitrary fixed fraction.

GitHub Actions workflow: `.github/workflows/casm-phase1.yml`.
