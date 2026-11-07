# V3/V4 integration review — 2026-11-07

The supplied V4 bundle is integrated on a local review branch, with V3 retained
as the reference inference path and V4 as an experimental historical-memory
path. The user's 12:1 inference-to-training preference is recorded as a ranking
objective after competence, expressivity and memory gates. No architecture is
currently qualified. Nothing in this integration has been merged.

## Provenance and scope

Source: `lrc_v4_inference_first_bundle.zip`, SHA-256
`a38d031f2b9a88ec9cbcbcf6956e0a520f828d346aa920e84f25f0fa721c210f`.
All 41 entries in its reproducibility manifest matched before import. Latest
standalone `Basic.lean`, `Paged.lean` and `paired_results.json` match the bundle.
Original reports/data are preserved in `../inference-v4-import/`, with an import
mapping. They describe the supplied runtime, not new measurements.

Imported cached inference, dependent two-hop models, page indexes, generation,
targeted credit, reference-program composition, cost selection, tests, benchmarks
and Lean sources. Existing `model.py`, `index.py`, and package initialization were
preserved, including LRC-002's explicit oracle diagnostic interface. The retained
original index lacks the bundle's streaming shortcut; measurements of this
integration therefore need their own provenance.

## Review fixes

- V4 focused training previously took the first and last answer positions across
  an entire batch. Different-length rows could receive the wrong intervention
  window. Focus now uses each row's answer window, excludes padding and skips
  unlabeled rows. Regression checks include row permutation.
- Cached LRC-001 metrics now include zero oracle reads when the integrated
  LRC-002 model interface expects that counter. Weights and logits are unchanged.
- Cost selection and mathematical cost helpers reject nonfinite reference inputs
  rather than silently accepting NaN or infinity.
- Page-bit decoding rejects fractional/string digits rather than truncating them.

The CI workflow includes inference tests and their path triggers. The first remote
run exposed an undeclared NumPy dependency in the imported mathematical/program
modules: all 27 original tests passed, but three inference test modules failed to
import. NumPy was preinstalled locally. The integration now declares the locally
tested NumPy 2.3.5 version in `requirements-inference.txt` and installs it in CI.
The failure remains visible in workflow run 42415078770; consult PR #51 for the
latest rerun. No test was skipped or weakened to resolve the failure.

## Checks actually executed

- Unmodified supplied bundle: 27 Python tests passed before integration.
- Integrated inference package: 32 tests passed, including five new regressions.
- Existing relational composition package: 27 tests passed.
- Existing CASM checks: four test functions passed in a direct runner.
- Full V4 stress rerun: 16,384 fixed-token generation steps; all eight sampled
  cached/bulk comparisons have equal argmax and maximum absolute logit error
  4.7684e-7. This is sampled agreement, not verification of every logit at every step.
- A 65,536-token prompt plus 512 continuation steps completed. This leg is a
  completion/throughput check; the script does not bulk-compare the long prompt.
- Lean kernel verification remains pending. The pinned 4.15.0 compiler was
  downloaded, but startup failed with `error: failed to locate application`.
  No theorem is reported as kernel-verified and no build success is inferred.

## Fresh paired timing

Python 3.12.14, PyTorch 2.15.0+cpu, one thread, random untrained weights,
`torch.inference_mode()`, warmup and eight alternating-order trials. Timings are
from a shared host and show considerable variability. Raw trials are in
`paired_results.json`; no trial was discarded.

| Prompt / output | Batch | V3 steps/s | V4 steps/s | Median paired V4/V3 decode latency | Paired range |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 / 256 | 1 | 1,024.56 | 1,085.56 | 0.9279 | 0.5564–1.1907 |
| 512 / 512 | 1 | 868.11 | 728.41 | 1.2099 | 0.8580–1.4436 |
| 256 / 256 | 4 | 3,359.71 | 2,869.13 | 1.1538 | 0.8619–1.6999 |

Median paired ratios need not equal ratios of separate medians. These results do
not precisely reproduce the supplied 7.7% overhead. All ranges cross 1; this run
does not establish a stable performance difference. Retain V3 as the reference
and require an accuracy benefit before promoting V4. There is no optimized
Transformer/SSM baseline and no GPU claim.

Fresh stress throughput was 1,266.04 byte-token steps/s over the 16,384-step run.
The 65,536-token prefill took 3.2502s; its 512-step continuation ran at 1,167.80
steps/s. Allocated historical feature tensors were 24 MiB in the long-prompt leg;
that is not total process or peak memory. Fixed-token continuation does not test
free-running language quality. See `stress_results.json` for raw counts/errors.

## Scientific interpretation

The [LRC-002 result](../lrc-002-2026-11-07.md) is the most consequential learning
evidence: oracle endpoint access with answer-focused training reached 100% IID
exact answers in all three seeds; learned routing averaged 6.25%, versus 5.21%
for a trained local model. Oracle endpoints already solve the relation chain.
They establish decoding capacity, not learned multi-hop reasoning. The unfamiliar
name test changes output characters and cannot isolate identity generalization.

V4's older pages improve what is mathematically accessible under the candidate
cap. They do not establish that a neural router can discover the correct page.
The compiled relation engine assumes supplied materialized maps and cannot count
as learned 256-hop reasoning. Binary page credit is a conditional local surrogate,
not a proven estimator for the complete discrete routing objective.

The [next research design](../../relational_composition/NEXT_RESEARCH.md) separates
copying, learned read selection, page selection, full route discovery and the
coupling comparison. It is a proposal awaiting executable preregistration; no new
learning fit was run after viewing LRC-002 held-out results.

## Publication status

Published: draft PR #49 for LRC-001; draft PR #50 now includes the LRC-002
protocol, code and results. The remote result commit is
`474fba0059deb6f2f054401231d8317e5a896da5`; its tree exactly matches the reviewed
local result commit `af0cad1`. Model checkpoints remain in the recorded archive.
The user explicitly authorized publication of the results/evidence and reviewed
V3/V4 integration after the initial approval-review block. The integration is
being published as a separate stacked draft PR, without merging. This report
records local validation; consult that PR for current remote CI status.

Reproduction commands are in the package README. `review_manifest.json` records
the reviewed source and measurement hashes separately from the original manifest.
