# Independent review: private Kaggle engineering probe

**Reviewer:** attention_architectures (independent of adapter and runner authors).

**Current state: GO for the first exact private GPU engineering/capacity submission identified below, subject to the binding launch settings.** The root-produced CPU receipt has been independently read and shows all five engineering check groups passing. No model import, model execution, checkpoint load, held-out data access, remote mutation, or credential inspection has been performed by this reviewer. This source/receipt review is not a GPU runtime PASS.

## Authorized scope and applicable instructions

Reviewed the repository AGENTS.md, the FPC Universal skill, nanoCOG QUALIFICATION.md, attention-comparison PROTOCOL.md and READINESS.md. The user has now instructed continuation using their Kaggle credential and free GPU. That direction supersedes the older CPU-only planning restriction for this specifically bounded engineering stage. It does not authorize paid compute, protected evaluation, scientific training, benchmark selection, or publishing a notebook. The root agent handles the credential; the reviewer must not inspect `private_kaggle_materialized` or any secret file.

The proposed next stage is a private, bounded probe using inert synthetic tensors and random weights. Its evidence can establish software behavior on the observed runtime. It cannot establish learned competence, quality equivalence, model ranking, long-context capability, or an architecture-level speed advantage.

## Separate environment-only probe

The independent exact-source environment review is recorded in `kaggle_environment_001/ENVIRONMENT_REVIEW.md`. Probe SHA-256 `70c81e91d05c38a7cd487c2d6a74b639ff2ce1802ce4b27a93071a2ffefc7719` has GO limited to one private free-quota T4 submission with worker internet disabled and an external 300-second timeout. It checks only selected runtime metadata and a fixed 16-value analytic gradient, with no research model or dataset. This separate clearance does not change the model/adapter probe's PENDING status.

## Initial required checks (historical pending state, superseded below)

| Area | Admission requirement | Current evidence |
|---|---|---|
| Shared shell initialization | Explicit copying or isolated RNG streams must make corresponding embedding, FFN, output, and shell parameters equal; differing mixer constructors consume different RNG draws | Awaiting adapters and test |
| Causality | Suffix mutation leaves preceding logits unchanged; cached GQA mask uses absolute query/key positions, including nonzero prefix and multi-token chunk | Awaiting source and execution |
| Actual model path | Compare full-sequence, chunked, and one-token **vocabulary logits** through the real Decoder, not a mock mixer or cached expected outputs | Awaiting source and execution |
| Ownership | State belongs to its model/session; fork duplicates storage, including previous embedding and every layer cache; reset produces a genuinely fresh stream | Awaiting adapters and tests |
| Packet inference | Functional path leaves supplied state unchanged; any owned in-place path requires eval plus inference mode and independently owned storage | Existing packet source declares these conditions; adapter use remains unreviewed |
| Gradients | Audit every claimed trainable path for finite gradients and justified nonzero connectivity; report deliberate zero gates separately | Awaiting tests; known default smear gate issue below |
| Complete state | Count persistent packet/local/GQA state, prior embedding, position counters, padding/capacity and actual underlying allocated storage; distinguish logical bytes, retained allocations, and peak transient allocations | Awaiting adapter and runner |
| Outputs | Label results engineering-only, random-weight synthetic; failed, skipped, unsupported, and unknown remain distinct | Awaiting runner |
| Resource envelope | Explicit bounded model/tensor shapes, CPU threads, GPU count, wall cap, no optimizer/fit unless separately reviewed as an engineering action, no unbounded retries | Awaiting probe manifest and outer watchdog |
| Private delivery | Kaggle notebook private; fixed source allowlist; no credentials, environment dumps, private labels, checkpoints, or broad workspace upload | Root to provide safe manifest/allowlist; reviewer will not inspect secret files |
| Offline execution | No worker networking or package downloads during the probe unless explicitly needed and reviewed; use installed runtime and record actual versions | Awaiting runner |
| Identity | Exact source hashes, runtime receipt, actual accelerator, command, resource actuals, test outcome and failure log retained | Awaiting completed artifacts and runtime |

## Source findings already communicated

1. `Decoder` constructs embedding, then variable mixers, then FFNs and output. Resetting one global seed for each model does not ensure identical later shell weights when mixer constructors differ. Common initialization needs an explicit equality check of the relevant named parameters, excluding intentional mixer differences.
2. `smear_gain` initializes to zero, so the default initial `smear_gate.weight` gradient is structurally zero. Do not mistake this for a disconnected graph or silently mark it as active. A separately labeled gradient-connectivity fixture may set a nonzero smear gain while retaining an untouched default-init receipt. Any such fixture must not be presented as the production initialization or learned evidence.
3. Packet routing has discrete forward choices and a dense training surrogate. Non-null gradients alone do not prove all routing/publish/value paths are healthy. Packet-age connectivity also needs more than one occupied retained entry. Gradient fixture settings and any deterministic stimulus must be disclosed.
4. The decoder state includes `previous_embedding` and a Python `tokens_seen` scalar in addition to layer states. Excluding those from complete-state accounting or fork/reset handling is incomplete. Counts of tensor elements alone may understate retained storage for views or overcount aliases; both logical and storage-aware quantities need explicit interpretation.
5. Existing packet code distinguishes functional copying from `step_inplace_` mutation. The latter is guarded by eval and `torch.inference_mode()`, but caller ownership is still a contractual requirement. No state reuse across independent model arms.
6. An engineering profile on random weights is not a serving-quality comparison. Timing can expose implementation cost or execution feasibility, but its output must not promote an efficiency winner before quality gates.

## Read-only source identity

| File | SHA-256 |
|---|---|
| AGENTS.md | 090c9d0b08a8d2e347cdc0ffbf63f5db5701dab06c7d1ac9106c6906b8c64164 |
| .agents/skills/fpc-universal/SKILL.md | 833b01daff60e91c9b9fb36a630ca0b290d23fe27269e6b3e8579e0951a7a00a |
| research/nanocog-001/QUALIFICATION.md | 7bd1950dbbbac6acedb1944dffd0fea80fd5abad8f6f24824e5f17800b773937 |
| research/nanocog-001/attention-comparison-001/PROTOCOL.md | 06e9534442697147bf321328f30e681cbe6c4bd44dd6fb9f6e77904263bc51c2 |
| research/nanocog-001/nanocog/model.py | b9ae1000992dbd9e103de5f2365b88fb0c2a674e9acc9cfc0c9ac2ba7b599ab3 |
| research/nanocog-001/nanocog/comm/packet.py | 346505fc26654609c8d5f19f59376cb0ecff11c1aa572c51ae7f188d4eeb997b |

## Preliminary adapter review (author revisions in progress)

The reviewer has now read the materialized adapter/tests/design and the pinned upstream `gpt.py`, `engine.py`, `flash_attention.py`, `common.py`, and reachable import declarations. The adapter's explicit common-tensor copies, delegated original decoder shell, model/session ownership, aligned full-prefill plus single-token serving scope, and storage-aware inventory are appropriate for an engineering probe.

Findings communicated for repair or qualification before exact-source clearance:

- **Blocking integration defect in first draft:** upstream smear/backout parameters have shape `[1]` while candidate equivalents are scalar `[]`; the initial strict copy routine rejects them. The author is adding explicit one-element semantic reshaping and a corresponding equality test.
- **Non-vacuous baseline tests:** upstream initializes attention/MLP output projections and smear gain to zero. Native-init cache parity alone does not exercise those paths. The materialized tests include a separately labeled activated fixture; a positive history-sensitivity control has been requested.
- **Gradient geometry:** with two layers, the backout tap is the final layer and its scale is nearly canceled by final RMSNorm. An all-nonzero test on that geometry may rely on numerical epsilon effects. A separately declared three-or-more-layer gradient fixture or explicit classification is required for meaningful coverage.
- **Partial update failure:** owned packet and upstream cache paths mutate state before all work succeeds. The author has added session poisoning; the injected-failure refusal test is being completed.
- **Buffer/config binding:** parameter version/pointer checks do not alone bind rotary buffers and model configuration. The author is extending the signature or must explicitly restrict mutation; ordinary root execution remains bounded.
- **Upstream cached chunk limit:** upstream `T>1` cache continuation omits prior smear at the first continuation token. The adapter correctly rejects multi-token cached continuation instead of claiming that parity. Candidate native chunk behavior is tested separately.
- **Import boundary:** upstream attempts external kernel discovery before resolving FP32 to SDPA. Root plans an explicit optional-kernel import guard, offline worker, and `USE_FA3=false` receipt. This is a qualified FP32 SDPA engineering path, not an optimized attention speed result.
- **Retained baseline view:** upstream previous-embedding storage can retain the complete prefill embedding allocation. Storage-aware reporting should expose this; it must not be erased or interpreted as a universal architectural memory disadvantage.

No native model import or execution was performed by this reviewer. Tests/runner/manifest remain pending exact final-byte verification; this preliminary review is not a model-probe GO.

## Completed adapter source review

The author repaired the scalar-shape copy defect with explicit one-element reshaping, normalized noncontiguous target slices symmetrically with their copy cost disclosed, and added the requested activated-history, buffer-change, partial-failure poisoning, three-layer gradient, and persistent owned-cache storage checks. The reviewer re-read the changes, verified these exact digests independently, and parsed both Python files with `ast.parse` without importing or executing them.

| Reviewed artifact | SHA-256 |
|---|---|
| adapters.py | 41cb92001ef24b0c19968d4d93b1905189ccdeebe4e4aa5a9f9b93c618c8b840 |
| test_adapters.py | 2a2fb0bfd61c3cd721fa6c4e81cd4e791710c4eab5823095d11c80847ef2f38a |
| ADAPTER_DESIGN.md | 165ab0d85db1defb4ababf0a7bce3c5e1a98ee77606dcdd619a1af26ac8583a2 |

Source review found no remaining blocking defect for executing these bounded tests. This finding is deliberately narrower than saying the implementation passes: actual runtime assertions remain unobserved by this reviewer. The test entry point constructs only inert random-weight models with vocabulary 97, width 32, two layers for ordinary checks and three for the separate gradient fixture, fixed short token sequences, and bounded cache capacities. It performs no optimizer updates, accesses no dataset or held-out cohort, and creates no trained checkpoint.

A local CPU check may now run under the root-controlled isolated CPU environment, with `NANOCHAT_DTYPE=float32` set before import, one Torch/BLAS thread, only the pinned source directories and reviewed integration directory on its import path, a hard outer timeout of at most 300 seconds, and retained complete PASS/FAIL output. Do not run Python with optimization flags that disable assertions. Root retains execution ownership. Any failed test remains a failure; repairs receive new hashes and review before a new attempt. This is engineering validation, not a scientific seed or comparative result.

For GPU submission the root runner still needs exact-source review: enforce offline optional-kernel discovery suppression, verify the actual SDPA/FA3 selection, bound device/model dimensions and runtime, and narrowly allowlist source/tokenizer assets while excluding credentials. Any real tokenizer integration must have its separately reviewed provenance. The current adapter deliberately does not support mixed-row lifetimes, padding, serialized state, or cached multi-token GQA continuation; these unsupported paths are not passed gates.

## Completed runner and staged-bundle review

The reviewer read the root-authored `runner_core.py`, `RUN_PROTOCOL.md`, `build_bundle.py`, hash-pinned dependency requirements and lock, and the actual staged `kernel.py` bootstrap/metadata. The embedded base64 archive was decoded **as data only** and every member was checked without executing or extracting research code. All 33 archive members matched the explicit manifest; all 14 upstream Python files matched their declared immutable git commit; all eight candidate Python files matched both declared candidate commits; and all six local integration artifacts matched the reviewed on-disk bytes. The three tokenizer assets matched their declared hashes, with the encoding bound to vocabulary 50,257 and BOS 50,256. Python members were statically parsed. The source-only archive contains 1,307,756 uncompressed bytes, below its 10 MB cap.

| Binding artifact | SHA-256 |
|---|---|
| runner_core.py | 25418c51065a301461ad0e8c468e83b299da569a78b44458e1a42e2ee4be03e8 |
| RUN_PROTOCOL.md | 64c94fe6099d4f99fa2a6dad5fa76d140faf800735f0f8a84d57e0560f52ee2c |
| build_bundle.py | 3f2fe8382be6a33befd28206573fea7b0dc54b276134f148f2fa6c4526857c2f |
| Staged kernel.py | c7cd17d088294cf56f62a372328921ed4e29cf083f8bfc897f7040a374ee4a81 |
| Embedded source archive | 65f19af445dc368e382faa32fd006941969c6df26f5c4b5330bd898472f555f5 |
| Staged kernel-metadata.json | 60f6f59614fb7889326d2fd9db6328bc0f382164d6e571953ca43c4d917fafa6 |
| bootstrap-requirements.txt | e2a15d3b282e2939ab256979eab265a9eaa77166ad56d909a07c55e2ddbfffb3 |
| CPU_CHECKS_ATTEMPT1.json | f91e77368bb296d10a9d0c6bce0007fa4863d7843891d0cb176d9f7d3c7dcec6 |
| TOKENIZER_PREPARATION_RECEIPT.json | efdcfe4e29d79e512a7337216a337e505c95b96ef37668d726519434378bd4b9 |

The CPU receipt binds the reviewed adapter/test hashes, Torch 2.9.1+cpu, zero training updates, and PASS for all five check groups. Its reported 0.411415435 seconds is the engineering-check run duration, not a model-speed benchmark. The tokenizer receipt reports engineering preparation and offline reload; tokenizer provenance also received its separate specialist review. The reviewer did not rerun either native check.

The core now uses a dedicated `CoreDeadline(BaseException)` so per-check `except Exception` handlers cannot swallow the 720-second deadline. Its outer handler preserves deadline failures. It records the scheduled arm roster and current stage/arm before expensive work, leaving later unrun arms explicitly NOT_RUN. The outer remote timeout is still required because Python signal delivery can be delayed by native calls.

### Binding GPU execution limits

- First submission only of the reviewed staged script, private, requested `NvidiaTeslaT4`, GPU execution on device 0, no paid allocation or account change.
- Enforce and retain the actual **900-second server runtime limit** and requested accelerator through the submission API/CLI receipt; the manifest declarations alone do not enforce these settings.
- Staged metadata is private with GPU enabled, TPU disabled, and empty dataset/kernel/competition/model-source lists.
- Unlike the separate internet-disabled environment probe, **this adapter job's internet is enabled** for its explicitly declared dependency bootstrap. Bootstrap installs only the two version/hash-pinned binary packages rustbpe 0.1.0 and tiktoken 0.11.0 from PyPI, with `--no-deps`, an isolated target, a retained installation report and a 120-second subprocess timeout. It does not install source distributions. Do not describe this worker as network-disabled. Subsequent research code uses embedded source/tokenizer bytes, blocks optional external kernels, and sets the HF offline policy.
- The core first reruns the reviewed inert checks on the actual GPU. Any failure prevents entry to the capacity stage.
- The capacity stage is exactly one synthetic batch of 1,024 target positions per arm, width 512, 12 layers, Q8/KV2 for GQA, at most 125M parameters per arm, followed by one fixed tokenizer fixture and eight owned decode steps. It performs backward passes but no optimizer construction/update and saves no trained checkpoint.
- All three untrained models remain resident; reported allocator peaks expressly include this overhead. Unequal parameter counts are explicit, and no quality, efficiency ranking, or capacity-matched claim is admitted.
- Root must complete its planned secret-byte exclusion check before upload. This reviewer verified the explicit member set and reviewed source, and did not inspect any credential material. Only `kernel.py` and its reviewed metadata are present in the staged upload directory.
- Retain failed attempts, actual runtime/dependency/backend receipts and output hashes. The protocol permits at most two adapter submissions in this continuation, but **this review clears only the first exact submission**; a second requires a retained defect, reviewed correction and new exact-version review.

The static machine-readable bundle review is saved as `attention_comparison_work/KAGGLE_BUNDLE_STATIC_REVIEW.json`. No remaining source or packaging defect was found that blocks this bounded engineering submission under the launch conditions above.

## Final disposition

**GO_FIRST_PRIVATE_GPU_ENGINEERING_ONLY** for the exact staged script and binding limits above. Actual GPU allocation, runtime behavior and results remain unobserved by this reviewer until receipts are returned. The independent environment-only review remains a separate stage. Model fitting, protected prediction/scoring, architecture ranking, language quality, deployment superiority, and scientific qualification remain unadmitted.
