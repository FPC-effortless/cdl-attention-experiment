# Interpretation of the first GPU capacity timings

**The current packet implementation has a substantial performance warning at the tested training shape.** Its recorded forward/backward interval was 33.074 seconds, compared with 0.251 seconds for GQA and 0.164 seconds for the local control. These are single resource-planning observations from untrained models, not a qualified inference benchmark or an architecture ranking. The engineering PASS confirms completion and the prescribed correctness checks; it does not establish competitive throughput.

## Observed evidence

Source: [exact version-1 run receipt](raw/adapter/adapter_results/run_result.json), SHA-256 `8323868b4a0d74e15471d4ec4070ce6f88396f0905f5fc80ed6b27c3a2b0f094`.

| Arm | Parameters | Recorded forward/backward seconds | Capacity status |
| --- | ---: | ---: | --- |
| GQA | 123,175,106 | 0.250734 | PASS_CAPACITY_ONLY |
| Packet | 81,076,562 | 33.074444 | PASS_CAPACITY_ONLY |
| Local | 79,841,330 | 0.164062 | PASS_CAPACITY_ONLY |

All three used FP32, batch 1, a 1,024-token input/target shape, width 512, 12 layers, the same generated token schedule and exactly copied common-shell weights. The runner used `cuda:0` on a Tesla T4; the second inventoried T4 was not used for model execution. The models differed in backend-specific parameters and remained resident together. There were zero optimizer updates and no training/evaluation corpus. The observed packet interval is approximately 132 times the GQA interval; that arithmetic describes these two recorded intervals only and is not an estimate with uncertainty or a general speed ratio.

## What the timer includes

The [runner](../kaggle_integration/runner_core.py) synchronizes CUDA before starting a monotonic host timer. The timed interval covers the [common adapter](../kaggle_integration/adapters.py), its input/target validation, model forward and cross-entropy calculation, the finite-loss check, backward, and the final CUDA synchronization. Consequently, it includes host work and any synchronization caused by validation and the finite-loss check.

Model construction, tokenization, `gc.collect()`, `empty_cache()`, and peak-counter reset occur before the timer. The detailed gradient scan, 14-token natural-text prefill, eight decode steps, state accounting and reset occur afterward. The timer therefore measures neither decode latency nor a complete optimizer/training step.

There is one timed capacity pass per arm, in fixed GQA → packet → local order, without a dedicated capacity-shape warmup or repeated sample distribution. Earlier engineering fixtures ran, so the intervals should not be labelled either controlled cold-start measurements or steady-state measurements. No `torch.compile` path is invoked by this runner.

The receipt's math-SDPA operator observation comes from a **separate untimed 16-token inference call** after the capacity probes. It establishes the operator seen in that call; it is not a CUDA trace of the timed 1,024-token forward/backward. Optional external kernels were disabled by protocol. These observations do not constitute a comparison against an optimized FlashAttention serving baseline.

## Source-backed bottleneck hypotheses

The [packet reference](../../nanocog/comm/packet.py), called through [PacketMixer](../../nanocog/interaction.py) in each [decoder layer](../../nanocog/model.py), performs a Python `for i in range(t)` loop. At this shape that is 12 × 1,024 = 12,288 token-loop iterations in the forward pass, before backward traverses the resulting autograd graph.

The timed `model.train()` call selects its dense branch. Each token pools packet buckets, forms hard route one-hot tensors and straight-through soft routing/publish terms, constructs replacement packet and presence tensors, and computes functional updates over the complete bounded topic state. State from one token feeds the next. Projections before this loop and output projections after it are batched, but the state recurrence and local-window construction remain inside the loop. In contrast, the local control applies its convolution to the whole chunk.

These are direct source observations. They make small-operation dispatch, the serial recurrence, repeated state materialization and autograd bookkeeping concrete profiling targets. **Their individual shares of the 33-second interval have not been measured.** The packet token loop contains no explicit `.item()`/Python truth test on its per-token CUDA results, so this review does not attribute a host/device synchronization to every iteration.

The implementation's inference branch is different: it gathers and updates selected buckets and supports owned state. The long training interval cannot be used as its decode latency. Conversely, a compact retained-state report does not establish low end-to-end latency or preserved language quality.

The inspected runner, adapter, packet and decoder hashes match the retrieved [source manifest](raw/adapter/run_source_manifest.json). No model was imported or executed for this interpretation; only source and downloaded receipts were read.

## Next gate before a broad comparison

First run a separately recorded, bounded profiling diagnostic on the exact reference implementation: separate forward from backward, collect CPU and CUDA traces, and measure warmup-controlled repetitions with explicit order, synchronization and compilation settings. Resolve whether dispatch, state updates, backward or another measured component dominates before choosing an optimization. Preserve this first result.

Because inference is the primary objective, measure prefill and owned single-token decode separately across declared context lengths and batch sizes, including adapter checks, state management and any routing/index work. Isolate each arm for deployment-memory measurements. Admit the strongest supported attention backend for the actual hardware and precision, and record its observed operator on the measured shapes. None of that serving evidence is supplied by the present eight-step correctness probe.

If profiling supports a fused or otherwise optimized packet implementation, qualify it against this reference for outputs, causal state transitions and gradients before timing it. Replacing the dense straight-through training recurrence with the indexed inference write path would change backward semantics and cannot be treated as a transparent optimization. Any later efficiency claim still requires the campaign's trained-quality, resource and statistical gates.
