# LRC-V4 engineering and mathematical review

Date: 2026-11-07. Status: **local prototype, not a validated language architecture**.

## Implemented

- Paged, causal, collision-preserving index (`hierarchical_index.py`). The
  old C-most-recent bottleneck is removed *within the page code's capacity*;
  all older records have an exact `(page,slot)` address. No dense prefix
  attention or N×N score matrix is introduced.
- Two dependent, separately typed reads per layer with learned binary page
  policies, a previous-layer occurrence reference and a shared operator
  (`hierarchical_model.py`); bulk and cached implementations both exist.
- One-coordinate counterfactuals over **active** page bits with a conditional
  two-action surrogate (`train_hierarchical.py`). This does not prove learned
  semantic routing and is not an unbiased full-joint gradient.
- Exact function-program compilation for known total graphs, vectorized serial
  baseline, update-aware cached compositions and automatic work-heuristic
  routing (`program_runtime.py`). No learned text-to-graph compiler is claimed.
- Generation API supporting non-PAD tokens, EOS, explicit context budgets and
  grouped variable-length prompts (`generation.py`).
- A hard quality-gated optimization criterion prioritizing inference time
  **12:1 over training time** (`design_selection.py`). No model meets an
  established accuracy gate yet.
- Python operational contracts/tests and written Lean 4 theorem candidates;
  Lean not installed, so theorems are **not kernel-verified**.

## Validation

- **27 Python tests pass**. Included: 10,017-record exhaustive page/slot
  reachability, 262,144-record sampled historical reachability, randomized
  causality and namespace isolation, forced older-page neural read, three-seed
  full-prefix vs cached differential testing, exact common-noise replay,
  finite training gradients, compiled program update invalidation, EOS/PAD and
  ragged generation tests.
- Long output: 16,384 continuation steps, approximately **1,362.69 byte-token
  steps/s** with a 128-token prompt on one CPU thread. At 8 checkpoints up to
  step 16,384, cached logits differ from bulk by less than `5e-7`.
- Long prompt: 65,536 input tokens; prefill approximately **3.0489 s**;
  subsequent 512-token decode approximately **1,334.55 steps/s**. Hidden-state
  cache *allocated* 24 MiB, excluding neural weights, posting list metadata,
  tensors, allocator overhead and Python objects.
- In an eight-repetition interleaved paired benchmark under inference mode,
  512 prompt/512 output, batch 1, V3 achieved approximately **1,365 steps/s**
  and V4 **1,274 steps/s**. The median V4/V3 decode-time ratio was **1.077**
  (7.7% slower for V4). This is a real inference cost of the added page heads.
- At 256 prompt/256 output, batch 4, V3 achieved about **4,198 aggregate
  steps/s** and V4 about **3,773 aggregate steps/s** in the interleaved paired
  benchmark; median paired decode-latency ratio **1.121**. Inter-trial noise
  was greater at batch 4. These are synthetic byte tokens from untrained
  weights, not production text throughput.
- In a separate five-repeat PyTorch benchmark at 512/512 batch 1, V4 decode
  time was **408.43 ms** using ordinary `no_grad` and **386.60 ms** using
  `inference_mode`, a roughly **5.3% reduction** in that environment.
- With known materialized graph maps (`|V|=256`, `H=256`), 65,536 independent
  queries took **14.49 ms** in a fair vectorized serial implementation versus
  **1.38 ms** including parallel program compilation and application: about
  **10.5x** in that specific single-run CPU measurement. Cached reuse took
  about **0.12 ms**. Single queries generally favor serial execution. No
  learned 256-hop ability is implied.

Raw observations, configurations, environment versions and operation counters
are included in `v4_results.json`, `paired_results.json`, `inference_mode_results.json` and
`v4_stress.json`. Short-run CPU timings are noisy and are not cross-hardware
performance conclusions. The three timing benchmark modes run three, eight (interleaved paired), and
five repetitions respectively; long-output/prompt stress is a single run with
sampled correctness checkpoints.

## Findings and remaining blockers

1. **Reachability != discovery.** An integer page address can locate an old
   record, but a randomly initialized neural route will not know the correct
   address, especially after cross-mention paraphrases.
2. **Two dependent reads != arbitrary learned multi-hop.** V4 has two dependent
   reads per layer. Arbitrarily deep known-graph composition is currently a
   separate exact reference backend; the neural model does not automatically
   emit or compile these programs.
3. **No competent comparison yet.** The original LRC-001 pilot had 0% exact
   IID answer accuracy. V4's current tests are engineering/parity tests, not a
   demonstration that this has changed. Do not publish a positive scientific
   result or assert improvement over attention on this evidence.
4. **Python/CPU overhead remains.** Indexes use per-row Python posting lists,
   discrete routing and small PyTorch operators. GPU fusion, asynchronous
   batching and a strong optimized baseline remain future work.
5. **Lean verification pending.** `lean/Relational/Paged.lean` and
   `lean/Relational/Programs.lean` express proof obligations but have not run
   under a Lean kernel. Their correctness has not been independently audited.

## Next controlled experiment

Build an oracle-writer/learned-reader ablation, measure first/second-hop page
and slot accuracy, gold record admission, cross-mention entity identity and
free-running answers on fresh worlds. If decoding with oracle pages works but
learned page retrieval fails, focus the next training iteration on entity
identity and hierarchical routing—not on wider networks. Only after achieving
an established exact-answer competence gate should candidates be compared on
the 12:1 inference/training criterion and measured GPU throughput.

No GitHub changes or PR creation are part of this local deliverable.
