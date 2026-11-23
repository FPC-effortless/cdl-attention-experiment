# Kaggle version 1 results: engineering pass, packet timing warning

**Both original private version 1 jobs completed. All five GPU engineering check groups passed, and all three untrained models completed the bounded capacity probe. The packet implementation also recorded a substantial forward/backward cost warning. No scientific comparison or efficiency advantage is established.**

The environment worker started at `2026-11-23T07:15:39.513254+00:00`; the adapter worker started at `2026-11-23T07:52:37.399993+00:00`. Their exact-version outputs were retrieved after the earlier queue observations. Current server receipts were obtained at approximately `2026-11-23T18:31:39+00:00`. The scheduler's reason for the earlier wait remains unknown. No replacement or duplicate Kaggle job was submitted.

## Execution and source identity

| Stage | Exact version | Server status | Worker status | Worker elapsed time |
| --- | --- | --- | --- | ---: |
| Environment | `eweewee2/cog-attention-comparison-environment-001/1` | COMPLETE | PASS_ENVIRONMENT_ONLY | 10.760 s |
| Adapter and capacity | `eweewee2/cog-attention-adapter-001/1` | COMPLETE | PASS_ENGINEERING_ONLY | 56.719 s |

The [environment metadata](raw/environment_remote.json) and [adapter metadata](raw/adapter_remote.json) confirm version 1, private visibility, and exact submitted source SHA-256 values `70c81e91d05c38a7cd487c2d6a74b639ff2ce1802ce4b27a93071a2ffefc7719` and `c7cd17d088294cf56f62a372328921ed4e29cf083f8bfc897f7040a374ee4a81`. The adapter [bootstrap receipt](raw/adapter/bootstrap_result.json) records the reviewed 33-member archive SHA-256 `65f19af445dc368e382faa32fd006941969c6df26f5c4b5330bd898472f555f5`. Its pinned binary dependency installation is retained in [dependency_install.json](raw/adapter/dependency_install.json).

The environment and adapter receipts identify two Tesla T4 devices, each with `15,636,037,632` bytes of device memory and compute capability 7.5. **The model tests executed on `cuda:0` only.** This is not a dual-GPU model run. Python was 3.13.16; Torch was `2.11.0+cu130`, with CUDA 13.0. The model probe used FP32 throughout. Both environment-device analytic gradient checks passed. These statements describe observed worker output, not the requested machine specification.

## Correctness checks

The [full GPU check receipt](raw/adapter/adapter_results/adapter_checks.json) records PASS for each group:

| Check group | Result |
| --- | --- |
| Common initialization and loss interface | PASS |
| Causality and full-forward/cached decoding parity | PASS |
| Fork, reset and session ownership | PASS |
| Retained state accounting | PASS |
| Gradient connectivity | PASS |

Activated fixtures were used where native zero initialization could otherwise make a check vacuous. Native zero gradients are still retained in the evidence. Passing connectivity fixtures does not show that the untrained models can learn a language task. The [model provenance](raw/adapter/adapter_results/model_provenance.json) preserves source identities, exact common-tensor mapping, parameter inventories, and the absence of an optimizer.

## Capacity and the timing warning

The models used width 512, 12 layers, a real 50,257-token GPT-2-compatible engineering tokenizer, and a maximum sequence length of 1,024. GQA used 8 query heads and 2 KV heads. Each arm completed one synthetic `1 × 1,024` forward/backward, a 14-token natural-text prefill, and eight owned decode steps. There were **zero optimizer updates** and no training or evaluation corpus.

| Arm | Observed parameters | Capacity result | One forward/backward observation |
| --- | ---: | --- | ---: |
| GQA | 123,175,106 | PASS_CAPACITY_ONLY | 0.250734 s |
| Packet | 81,076,562 | PASS_CAPACITY_ONLY | 33.074444 s |
| Independently parameterized local-only | 79,841,330 | PASS_CAPACITY_ONLY | 0.164062 s |

Source: [run_result.json](raw/adapter/adapter_results/run_result.json). The timing field is explicitly for resource planning. It includes the adapter's validation/forward/loss, a finite-loss check, backward, and final CUDA synchronization; it excludes model construction and the later prefill/decode checks. These are single observations, without repeated trials, a warmup protocol for this shape, or uncertainty estimates. They are not inference-latency measurements.

The packet observation is a concrete warning about this implementation. Preserve it and profile its cause before committing to a larger training comparison. A separate [timing interpretation](TIMING_INTERPRETATION.md) distinguishes what the code and receipts establish from hypotheses about Python loops, tensor updates and launch overhead. No implementation change or new profiling run is included here.

A separate untimed 16-token GQA call observed `aten::_scaled_dot_product_attention_math` and `aten::scaled_dot_product_attention`. `HAS_FA3` and `USE_FA3` were false. This is not a trace of the timed 1,024-token backward, and these results do not benchmark an optimized FlashAttention baseline.

The common shell has 76,677,170 parameters in every arm, but architecture-specific parameter counts differ. All three models remained resident during the capacity probe. Allocator peaks therefore include the other resident models and must not be relabeled as per-model deployment memory. The retained-state estimates describe a particular session at position 22; GQA includes its allocated 1,024-token cache capacity. They establish neither equal-quality memory savings nor long-context competence.

## Interpretation and next gate

The supported conclusion is that the reviewed code, adapters, tokenizer and bounded FP32 capacity shapes worked on the observed T4 runtime. The GPU checks extend the existing CPU engineering evidence. They do not establish trained language quality, sample efficiency, adequate training budget, long-context retrieval, serving throughput, or superiority over GQA/MLA/KDA and other admitted comparators.

The next useful performance gate is a separately specified, bounded profile that isolates the packet implementation's cost and measures prefill and decode independently. Any optimized candidate must retain the correctness and session checks. A subsequent fair comparison still needs competent trained controls, frozen data/tokenizer, matched exposure and tuning budgets, suitable optimized baselines, repeated measurements, and the existing statistical and quality gates. Scientific campaign admission remains false.

## Evidence preservation and provider switch

[MANIFEST.json](MANIFEST.json) hashes all 11 raw downloads and server metadata receipts, copied byte-for-byte under `raw/`. Logs and warnings are retained. [SUBMISSION_STATUS_SNAPSHOT.json](SUBMISSION_STATUS_SNAPSHOT.json) preserves the earlier queued observation; [current status](../kaggle_integration/EXECUTION_STATUS.json) records completion. [Independent result review](INDEPENDENT_RESULT_REVIEW.md) checks the claims and their limits, including the environment probe's broad BF16 API flag, which is not a native BF16-kernel qualification.

The requested [MoLab notebook](../molab_integration_001/README.md) is prepared and independently reviewed, but remains **not uploaded and not executed on MoLab**. Google verification was not completed, and the current MoLab page still shows Sign In. Its wrapper's local process/export checks do not substitute for marimo reactivity validation or an observed MoLab GPU. The original Kaggle completion is recorded under its actual provider.
