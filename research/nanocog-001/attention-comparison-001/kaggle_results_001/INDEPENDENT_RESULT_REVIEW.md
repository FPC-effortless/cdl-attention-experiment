# Independent review of retrieved Kaggle version 1 results

**The retrieved jobs completed and passed the specified engineering checks. The packet implementation nevertheless took 33.07 seconds for the single resource-planning forward/backward pass, versus 0.25 seconds for GQA. That is a concrete performance concern for this implementation. These receipts do not establish a fair architecture ranking or learned capability.**

This review inspected downloaded JSON, both worker logs, and the exact public source that produced them. It used only standard-library parsing, hashing, and consistency checks. No models were imported or run, no jobs were resubmitted, and no credentials or private directories were accessed. These are Kaggle results; MoLab remains unrun.

## Observed completion and runtime

| Evidence | Recorded result |
| --- | --- |
| [Environment remote metadata](raw/environment_remote.json) | Requested/current version 1; `COMPLETE`; no failure message |
| [Environment receipt](raw/environment/environment.json) | `PASS_ENVIRONMENT_ONLY`; both devices passed the FP32 analytic-gradient check; 10.7602 seconds |
| [Adapter remote metadata](raw/adapter_remote.json) | Requested/current version 1; `COMPLETE`; no failure message |
| [Bootstrap receipt](raw/adapter/bootstrap_result.json) | `BOOTSTRAP_VERIFIED`; expected source archive hash |
| [Adapter checks](raw/adapter/adapter_results/adapter_checks.json) | Five check groups, all `PASS` |
| [Adapter run](raw/adapter/adapter_results/run_result.json) | `PASS_ENGINEERING_ONLY`; GQA, packet, and local all `PASS_CAPACITY_ONLY`; core elapsed 56.7193 seconds |

Both jobs report Python 3.13.16, PyTorch `2.11.0+cu130`, CUDA 13.0, and two Tesla T4 devices with compute capability 7.5 and 15,636,037,632 bytes per device. The adapter explicitly selected **`cuda:0`**, used FP32, and disabled TF32. Its model and allocator observations are single-device observations, not a two-GPU training result or pooled memory budget.

The adapter used the pinned `tiktoken==0.11.0` and `rustbpe==0.1.0` wheels. Both installed wheel hashes match the reviewed requirements. The environment probe's base-image `tiktoken==0.14.0` is therefore not a contradictory adapter dependency report. The worker used the declared GPT-2 vocabulary of 50,257 tokens; it did not reproduce an absent trained 32K tokenizer.

## Source identity and internal consistency

Independent hashing matched the public environment worker to both its output receipt and retrieved remote metadata. The embedded reviewed adapter worker matches the remote adapter source hash. All 33 embedded archive members match the downloaded source manifest; the manifest also exactly matches the bootstrap receipt's copy. All five imported model-class hashes match their corresponding manifest entries. Six local integration files, including runner, adapters, tests, tokenizer loader, protocol, and requirements, match the executed bundle manifest.

| Reviewed source | SHA-256 |
| --- | --- |
| Environment worker | `70c81e91d05c38a7cd487c2d6a74b639ff2ce1802ce4b27a93071a2ffefc7719` |
| Adapter worker | `c7cd17d088294cf56f62a372328921ed4e29cf083f8bfc897f7040a374ee4a81` |
| Adapter archive | `65f19af445dc368e382faa32fd006941969c6df26f5c4b5330bd898472f555f5` |
| `runner_core.py` | `25418c51065a301461ad0e8c468e83b299da569a78b44458e1a42e2ee4be03e8` |
| `adapters.py` | `41cb92001ef24b0c19968d4d93b1905189ccdeebe4e4aa5a9f9b93c618c8b840` |
| `test_adapters.py` | `2a2fb0bfd61c3cd721fa6c4e81cd4e791710c4eab5823095d11c80847ef2f38a` |

The remote source hashes above are read from the supplied retrieval receipts; this review did not make a separate authenticated Kaggle request. Parameter totals agree with the detailed parameter inventories and the common/backend-specific subtotals. Scheduled-arm statuses agree with the completed capacity rows. Gradient and state-accounting summaries agree with their detailed entries.

## What correctness was exercised

The five groups check common-weight copies and loss semantics; causality and full/prefill/step agreement; fork/reset/session ownership; retained-state accounting; and gradient connectivity. The small correctness models use width 32, vocabulary 97, and two layers, with three layers for gradient connectivity. Exact independent common-tensor copies were checked 22 times. Loss comparisons have zero reported error. The largest reported full-versus-prefill/step error is approximately `5.59e-9`; suffix-causality errors are zero in those fixtures.

The tests deliberately exercise nonzero projections and smear in a separate labeled intervention, preventing native zero initializers from trivially hiding some broken paths. Every tested trainable parameter tensor has a present, finite, nonzero gradient in that intervened fixture. This is a connectivity check, not evidence of learning.

Native initialization still has zero gradients, and they must remain visible: the capacity runs report 61 zero-gradient parameter tensors for GQA, 13 for packet, and 13 for local. All gradients are present and finite. No optimizer is constructed or stepped; the runner performs `backward()` and clears gradients. `optimizer_updates=0`, no corpus, untrained provenance, and `scientific_comparison_admitted=false` are consistent with the inspected execution path.

These finite fixtures do not establish exhaustive correctness across lengths, devices, precisions, batching policies, or all cache states. The 1,024-token capacity backward is a separate finite-gradient check; it is not a 1,024-token full/cache-equivalence test. The 14-token natural-text prefill and eight greedy decode steps check finite execution only, not whether the untrained model answered the prompt correctly.

## Capacity observations and the implementation warning

All three models use width 512, 12 layers, batch 1, and a 1,024-token synthetic backward schedule, with identical copied common-shell tensors totaling 76,677,170 parameters. They retain different backend-specific modules and parameter counts. The GQA configuration uses eight query and two KV heads.

| Arm | Total parameters | One forward/backward, seconds | Peak CUDA allocated bytes, all models resident | Retained tensor-storage bytes |
| --- | ---: | ---: | ---: | ---: |
| GQA | 123,175,106 | 0.250734 | 2,858,248,192 | 12,584,964 |
| Packet | 81,076,562 | 33.074444 | 2,787,454,976 | 180,320 |
| Local control | 79,841,330 | 0.164062 | 2,343,901,184 | 51,200 |

The timing field is explicitly a resource-planning measurement. Packet's observed duration is concerning even though this is not a benchmark: its current path is costly at this shape. A single sequential observation, unequal capacities, native untrained initialization, and no dedicated timing repetitions do not support a general speed ratio, confidence interval, or ranking of optimized algorithms. All three models remain resident; none of these allocator peaks is isolated model deployment memory or a complete GPU-process memory total. No optimizer-state memory was exercised.

The final retained state is measured at position **22** with a declared session capacity of **1,024**. GQA preallocates its entire KV capacity; packet and local use bounded state. These values establish allocated state for the implemented sessions, not equivalent information retention or equal-quality memory efficiency. Reporting the resulting storage ratio as a general memory win would overstate the evidence.

`retained_state_bytes_estimate` adds tensor storage to Python wrapper and metadata estimates. That combined number is **not pure GPU VRAM**. The table above deliberately uses `tensor_storage_bytes`. Per-tensor device fields are absent from that inventory; the source constructs the tested session tensors on the selected model device, while Python metadata resides on the host. Temporary activations, returned logits, allocator fragmentation, model weights, and peak process memory are outside retained-state accounting.

## Qualifications that must accompany the result

- **BF16 field:** `bf16_supported:true` is preserved. The probe calls `torch.cuda.is_bf16_supported()` with no arguments, and the [PyTorch 2.11 API](https://docs.pytorch.org/docs/2.11/generated/torch.cuda.is_bf16_supported.html) defaults to `including_emulation=True`. It does not establish native BF16 Tensor Core execution on T4. NVIDIA identifies [T4 as Turing](https://www.nvidia.com/en-us/data-center/tesla-t4/); its [Ampere whitepaper](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-ampere-architecture-whitepaper.pdf), printed page 26, describes BF16 Tensor Core support as an Ampere addition. This ambiguous capability label does not invalidate the FP32 tests.
- **Attention backend:** a separate, untimed 16-token GQA call reports `aten::_scaled_dot_product_attention_math`. `HAS_FA3` and `USE_FA3` are false. This is not a FlashAttention result; the profiler observation is not direct profiling of the timed 1,024-token backward or decode calls.
- **Failures and warnings:** no retrieved stage or capacity arm reports failure. Logs preserve the profiler-cycle warning and later notebook-export syntax warnings; these are not silent test failures. Intentional invalid-session and partial-cache-mutation failures are retained as successful rejection tests. Native zero-gradient lists remain part of the evidence.

The justified conclusion is that the pinned adapter implementations completed their specified FP32 engineering checks and this bounded capacity exercise on one T4. Training adequacy, learned quality, matched-capacity comparisons, optimized mixed precision, other modern attention baselines, and reproducible performance benchmarking remain open. No comparative success claim is admitted by these results.
