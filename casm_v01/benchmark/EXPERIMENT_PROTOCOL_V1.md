# CASM-Bench v1 — experimental comparison protocol

This file governs how models are trained and compared against the frozen benchmark. The benchmark cases/metrics are defined in `CASM_BENCH_V1.md`.

## 1. Mechanism ablations must be compute/data matched

When the claim is that mechanism A is better than mechanism B, keep fixed unless the changed mechanism logically requires otherwise:

- training examples and their order;
- training seed family;
- optimizer and optimizer hyperparameters;
- batch size and effective batch size;
- sequence length / episode policy;
- number of optimizer steps and visible training-token budget;
- loss weights not under direct test;
- compatible parameter initialization;
- evaluation decoding and inference precision.

Report parameter counts and wall-clock compute. If parameter count or inference compute changes materially, describe the result as a **capacity/compute tradeoff**, not a clean mechanism win.

## 2. Learning/sample efficiency

For long experiments, save predetermined checkpoints (recommended 25%, 50%, 75%, 100% of the fixed training budget) before looking at DEV results. DEV may be evaluated on those checkpoints to produce a learning curve. Do not choose checkpoint times after seeing performance.

Report score vs training tokens/steps in addition to final score when making a sample-efficiency claim.

## 3. DEV is iterative; HOLDOUT is promotion-only

`DEV-CORE` and `DEV-OOD` are the canonical scoreboards for every architecture interaction. It is expected that research decisions adapt to DEV.

`HOLDOUT-CORE` and `HOLDOUT-OOD` must not be used for ordinary iteration. They are evaluated only after a candidate and its hyperparameters are frozen and the multi-seed DEV promotion gate is met.

If a candidate is modified after seeing HOLDOUT, that HOLDOUT observation becomes part of the development history and cannot be reused as fresh evidence for the modified candidate.

## 4. Public-repository limitation

The repository is public, so the fixed HOLDOUT generator/seeds are **process-protected, not cryptographically secret**. CASM-Bench v1 therefore protects against routine adaptive evaluation and exact train/eval overlap, but it cannot prove absence of deliberate holdout inspection.

For publication-grade or third-party claims, add a genuinely blind evaluation after model freeze: private/external cases or a seed unavailable to the model-development process. Do not describe the public v1 HOLDOUT as a secret benchmark.

## 5. Contamination certification

Every new training loop should use `casm.training_fingerprint` to record SHA256 hashes of visible problem prefixes ending at `answer `. Gold answer bytes are excluded from the hash.

Run exact-overlap checks on DEV as an audit and on HOLDOUT as a promotion requirement. An official HOLDOUT result without the training hash artifact is `uncertified`, even if the training code was believed to use different seeds.

Exact hashes do not detect paraphrase/template contamination; CORE/OOD separation and future blind evaluation address different parts of that risk.

## 6. Test-time compute

The primary score uses the architecture's declared default reasoning depth fixed **before** evaluation. Additional 1/3/5 or other depth sweeps are diagnostics. Do not choose the best depth on HOLDOUT and report it as if it were prespecified.

When extra recurrent steps improve capability, report both capability delta and inference-time delta.

## 7. Statistical evidence

- One training seed: `DIRECTIONAL` at most.
- At least three independent training seeds are required for a replicated DEV claim.
- Use the same fixed DEV cases for paired comparisons.
- Report every seed, the mean, and per-task results; do not report only the best seed.
- Promotion criteria are defined in `CASM_BENCH_V1.md`.

## 8. Historical experiments

Scores produced before CASM-Bench v1 are audit history, not directly comparable canonical benchmark scores. Earlier CASM evaluations used changing generators and contained known validity defects. The first model evaluated under v1 establishes the canonical baseline.
