# COG-ATTENTION-KAGGLE-ADAPTER-001

## Purpose and interpretation

Qualify a reusable tokenizer/model/session integration and measure whether a
meaningful model geometry fits the available Kaggle GPU. This is engineering
execution with inert fixtures and untrained weights. It is not a language-model
fit, training-adequacy study, modern head-to-head, or quality/efficiency ranking.
The source-only campaign admission state remains unchanged.

## Inputs and models

- Pinned nanochat `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`, source unchanged.
- Candidate package bytes from comparison proposal commit
  `1a91168b216a49543cc839153fa299a262320730`; its relevant model code is unchanged
  from candidate reference `39be434ff160d4e6507df8d2114979ba5529caff`.
- Exact reviewed adapter, test, tokenizer and runner source hashes.
- Real GPT-2 BPE through the existing upstream compatibility path: 50,257 tokens,
  source bytes independently checked, portable encoding loaded offline.
  This is a declared engineering choice, not reproduction of absent 32K assets
  or a frozen final training-campaign tokenizer decision.
- No corpus, checkpoint, protected evaluation, Kaggle dataset, external notebook,
  model source, user prompt history or credentials attached to workers.
- Small activated-path correctness fixtures plus a capacity geometry of width512,
  12layers, query8/KV2 heads, full-context GQA, 1,024tokens and batch1.
  Each model must be below125M parameters. Actual counts are retained; unequal
  architecture-specific capacity is explicitly not treated as capacity matching.

## Execution and limits

1. Source review and local CPU correctness checks on Torch2.9.1+cpu before GPU.
2. Private Kaggle submission using existing free quota; request `NvidiaTeslaT4`.
3. Worker bootstrap may obtain only pinned rustbpe0.1.0 and tiktoken0.11.0 wheels
   from PyPI in an isolated target directory, with install receipt retained.
   Base-runtime dependency versions and actual GPU details are recorded.
4. All executable research sources and tokenizer inputs are embedded in one
   deterministic allowlisted archive, checked by archive/member hashes before import.
5. Optional Hugging Face kernel imports are explicitly blocked. This FP32 stage
   qualifies the unchanged upstream SDPA fallback, not optimized mixed-precision
   serving performance. Record actual profiler operator names.
6. Worker core deadline720seconds; server timeout900seconds per submission.
   Maximum two GPU adapter submissions in this continuation; a second requires
   a retained defect, reviewed correction, new version and separate receipt.
   Environment probe is separate and capped300seconds, one submission.
7. No optimizer construction/step, no training run, no data selection, no paid
   resource or account change. Only full forward/backward, serving and reset/fork
   engineering checks are authorized by this manifest.

## Required checks

Exact common tensor values and independent storage; causal prefix/history
sensitivity; full-prefill/single-token agreement with activated paths and nonzero
smear; candidate functional chunks; fork/reset isolation; rejection of wrong
model/session/config or unsupported padded/mixed-row and cached multichunk inputs;
gradient connectivity; all retained state storages and metadata counted.

Small activated fixtures are labeled separately from unchanged native
initialization. A zero native gradient caused by upstream zero projections must
remain visible. A separate fixture exercises paths without becoming a trained
checkpoint. All failures are retained and stop entry to the capacity stage.

The capacity stage uses the same inert1x1,025 token schedule, one1x1,024 forward/
backward per arm, and real tokenizer prefill followed by eight owned decode steps.
No optimizer updates occur. All three models are resident, so reported allocator
peaks are capacity observations with that explicit overhead, not per-model
deployment memory. Elapsed times are for resource planning, not benchmark wins.

## Remaining scientific prerequisites

Candidate-specific optimizer groups, licensed frozen TRAIN/CAL/DEV manifests,
adequacy and tuning budgets, capacity-matched contrasts, MLA/KDA integrations,
mixed-precision kernel qualification, statistical validation and independent
fit/prediction/scoring admission remain open. The GPU instruction authorizes
free private compute continuation; it does not convert this engineering result
into scientific evidence or authorize opening reserved data.
