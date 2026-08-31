# CASM A4 / state-preserved rerun contract

## Status

This branch is the canonical rerun lane for the compression-attention experiments in this repository. Results produced by the older CASM-specific task harnesses remain historical diagnostics; they are **not** counted as results under this contract.

## Source benchmark

The contract is adapted from the current `FPC-effortless/TAC-transformer` benchmark stack:

- `kaggle/benchmark_chunked_memory.py` / `tac_transformer.training.benchmark_chunked_memory` for chunked recall;
- the corrected same-stream / batcher-state-preserved methodology used by the content-update-frequency matrix;
- `kaggle/benchmark_inference.py` (A4) for prefill, carried-query, decode throughput, peak-memory fields, and vanilla-normalized ratios.

The CASM adapter must preserve the benchmark semantics while using CASM's byte/token interface and memory/state implementation.

## Correctness matrix

Tasks:

1. `single_key`
2. `multi_key`
3. `delayed_query`
4. `noisy_key`
5. `multi_hop`

Seeds: `11, 23, 37`.

Canonical compact settings for the first full matrix:

- 120 training steps per task/seed/variant;
- batch size 8;
- evaluation: 6 batches, eval batch size 8;
- logical sequence length 32;
- CPU, four torch threads.

### State-preservation invariant

Context/prefill and query/decode must be evaluated as one logical stream. A candidate is invalid if it silently resets episodic or persistent state between the context and its query. A separate reset control must deliberately clear the carried state so the causal value of memory can be measured.

Report at least:

- carried-query accuracy;
- reset-query accuracy;
- shuffled-state query accuracy;
- carried-minus-reset delta;
- carried-minus-shuffled delta;
- task and seed breakdowns;
- training/evaluation loss where applicable;
- parameter count and training wall time.

## A4 efficiency profile

For every deployable architecture variant report:

- prefill throughput / latency;
- carried-query throughput / latency;
- single-step decode throughput / latency where the architecture supports token decode;
- prefill peak memory;
- carried-query peak memory;
- decode peak memory;
- ratios against a parameter-matched vanilla/local baseline.

CPU peak memory is a proxy; CUDA runs, when available, should use allocator peak memory.

## Historical experiment families to rerun

The unified matrix will preserve the main hypothesis-changing families from this conversation:

- local-only / no cross-chunk memory;
- ordinary Q/K compressed memory;
- runtime compression-score routing;
- compression-trained Q/K;
- answer-utility Q/K;
- recurrent Q/K (1-step and 3-step shared reasoning);
- set-conditioned utility recurrence;
- scoped-memory recurrence;
- verifier/process-supervised recurrence;
- direct LM-head deep supervision;
- explicit parallel answer-state refinement.

SmolLM Stage-A/Stage-B router-only experiments are reported separately because they are not from-scratch CASM architectures; when rerun, they use the same five-task data contract where technically meaningful rather than being merged into the CASM architecture table.

## Benchmark-validity gates

Before the full matrix can run:

- all five task generators must have deterministic seed behavior;
- `multi_hop` must require at least two relations/edges rather than direct lookup;
- negative/control cases must be represented where the source task defines them;
- carried state must differ causally from reset state on a hand-constructed positive control;
- shuffled-state control must actually permute state between batch elements;
- context and query labels must never leak into the context representation;
- variants compared in a paired cell must use the same seed/data stream and compatible initialization;
- parameter and compute differences must be reported, not hidden.

## Promotion rule

A prior CASM conclusion is not considered reproduced until its variant has completed this state-preserved matrix. Lower language-model NLL alone is insufficient: memory/routing claims require carried-state benefit over reset/shuffled controls, and reasoning claims require task-level correctness on the appropriate benchmark task.
