# CASM Phase 1 — Boolean DAG

This directory is the standalone CASM Phase-1 harness. It is separate from Veritas.

## Frozen execution contract

- single-pass topological execution; no recurrent spectral-radius diagnostic;
- fixed upper-triangular candidate substrate `A` shared across episodes;
- variable-size programs represented by an explicit hard existence mask;
- true wiring is one of many admissible candidate wirings, so `m_i m_j A_ij` is not the oracle;
- routing depends only on structural state; runtime values never enter the router;
- CASM-S uses factorized node-wise query/key compatibility, with no op-pair lookup table;
- `alpha_eta.shape == (2,)`: one learned computation strength per syntactic argument port;
- raw operand order is retained for evaluation; commutative canonicalization is bookkeeping-only;
- copy-mask is a mandatory oracle-wiring preflight, not a learned baseline;
- first learned comparison is Static Mask vs CASM-S; other router variants are deferred.

## Gates

0. finite-depth numerical viability and gate initialization
1. gate intervention changes the forward result
2. router parameters receive task gradient
3. structural inputs change routing
4. severing a true edge changes task loss
5. held-out `NOT -> XOR` role-pair routing separates true from distractor edges
6. topology/parameterization integrity

## Runner

From repository root:

```bash
python -m casm_v01.phase1_dag.runner --train-size 128 --test-size 64
```

The runner first executes the copy-mask falsification control, then trains Static Mask and CASM-S on the identical episode split. The OOD split excludes `NOT -> XOR` from training and requires it in test episodes.

GitHub Actions workflow: `.github/workflows/casm-phase1.yml`.
