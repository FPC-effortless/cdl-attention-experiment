# Kaggle continuation: integration implemented, GPU jobs queued

Status snapshot: **22 November 2026, America/Los_Angeles**. Exact UTC observations
are in [EXECUTION_STATUS.json](EXECUTION_STATUS.json).

The authorized Kaggle credential authenticated successfully. Two private version 1
jobs were accepted, and their returned source bytes match the reviewed hashes.
Both were still **QUEUED** at the recorded final inspection. No GPU allocation,
GPU model execution, training run, or modern head-to-head result is established.

## Completed work

- A shared loss interface for the original nanoCOG packet decoder, a separately
  parameterized local-only decoder, and unchanged pinned nanochat GPT configured
  with genuine grouped-query attention.
- Explicit independent copies of common embedding, output, MLP and shell tensors
  from upstream initialization. Merely sharing a seed is not the method used.
- Owned packet decoding through the original decoder shell, with model/session
  identity checks, fork/reset handling, invalidation after partial-update failures,
  and complete retained tensor-storage/metadata inventories.
- A real shared 50,257-token GPT-2 BPE through nanochat's existing compatibility
  path. Both raw assets were independently hashed against the expected values;
  portable JSON export was reconstructed with network calls blocked. This is an
  engineering tokenizer choice, not reproduction of a missing trained 32K asset.
- Five CPU engineering check groups passed on isolated Torch 2.9.1+cpu: common
  initialization/loss, causal/cache behavior, ownership/fork/reset, state accounting
  and gradient connectivity. [Complete receipt](CPU_CHECKS_ATTEMPT1.json).
- Separate source/admission and archive-boundary review. The GPU submission embeds
  33 explicitly allowed files/assets, with whole-archive and per-file checksums,
  and retains version/hash-pinned binary dependency installation receipts.

These are completed engineering changes. The local-only model has not been
trained; the common loss facade is not a finished production trainer. Mixed
precision, continuous batching, checkpoint/state resumption, scientific optimizer
groups and corpus integration remain open.

## Private Kaggle jobs

| Stage | Exact remote reference | Version | Requested GPU / runtime limit | Observed state |
| --- | --- | --- | --- | --- |
| Environment | `eweewee2/cog-attention-comparison-environment-001` | 1 | T4 / 300 seconds | QUEUED |
| Adapter and capacity | `eweewee2/cog-attention-adapter-001` | 1 | T4 / 900 seconds | QUEUED |

- [Environment job](https://www.kaggle.com/code/eweewee2/cog-attention-comparison-environment-001)
- [Adapter/capacity job](https://www.kaggle.com/code/eweewee2/cog-attention-adapter-001)

Server metadata confirms both notebooks are private. The environment worker has
internet disabled. The adapter worker has internet enabled for its declared
hash-pinned binary dependency bootstrap; it uses embedded research code/tokenizer
assets and blocks optional unpinned external attention-kernel imports. This is not
a network egress sandbox. Credentials were excluded from staged code and assets.

The account reported 30 GPU hours remaining before submission. That was an observed
quota snapshot, not proof of immediate hardware availability. No paid resource,
account upgrade or protected evaluation was requested.

## What the queued capacity run will establish

After the same checks pass on the actual GPU, the run constructs width 512,
12-layer models with the real 50,257-token vocabulary, and an 8-query/2-KV-head GQA
baseline. Each model must remain below 125M parameters. It records actual counts
and performs one 1 × 1,024 synthetic forward/backward, real tokenizer prefill and
eight owned decode steps per arm. There are **zero optimizer updates**.

Architecture-specific capacities differ and remain visible. All three models are
resident during the probe, so allocator peaks are labeled accordingly. Recorded
times support resource planning only. A successful capacity result would not
establish training adequacy, language quality, binding correctness, long-context
competence or an efficiency winner.

## Concrete defects caught and repaired

Review corrected the upstream [1] versus candidate [] scalar mapping, normalized
noncontiguous training targets for both arms, and added activated-path fixtures
so zero initialization cannot make attention tests vacuous. Gradient fixtures use
three layers to avoid a nearly canceled final-backout scale. Session tests cover
stale buffer/config use, independent forks and failed owned updates.

The unchanged upstream baseline has a cached multi-token continuation limitation
in its smear path. This adapter supports full prefill plus single-token continuation
and rejects unsupported cached chunks. It does not silently fix or rename the baseline.

## Evidence and continuation

- [Bounded execution protocol](RUN_PROTOCOL.md)
- [Adapter design and remaining limits](ADAPTER_DESIGN.md)
- [Independent admission review](INDEPENDENT_ADMISSION_REVIEW.md)
- [Independent bundle review](BUNDLE_REVIEW.md)
- [Tokenizer review](TOKENIZER_REVIEW.md) and [preparation receipt](TOKENIZER_PREPARATION_RECEIPT.json)
- [Submitted source manifest](BUNDLE_RECEIPT.json), [submission](SUBMISSION_ATTEMPT1.json),
  and [server privacy/source/status receipt](REMOTE_RECEIPT_ATTEMPT1.json)
- [Exact continuation instructions](RESUME.md)

The first useful next action is to retrieve these exact versions after Kaggle
finishes queuing. Do not create duplicate jobs because an earlier reply ended.
After successful GPU integration/capacity evidence, the next scientific work is
an admitted TRAIN/CAL adequacy pilot with explicit candidate optimizer groups,
licensed frozen data, matched exposure/tuning, complete resource accounting and
the remaining GQA/MLA/KDA baseline admission work. The prospective comparison's
statistical and quality gates remain in force.
