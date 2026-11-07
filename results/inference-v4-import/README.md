# LRC-V4 — Inference-first hierarchical relational composition

**Status: local research prototype, not a trained/qualified model.**
Not pushed to GitHub. Original frozen LRC-001/002 results and V3 artifacts are
untouched. All new V4 files are in this standalone bundle.

## What changed

1. **Page-addressable collision-safe causal memory** (`hierarchical_index.py`):
   every older occurrence can be addressed with a finite-width page code *if*
   its page fits in the code width. Maintains C-sized reads and causal safety.
2. **Learned two-hop page policies** (`hierarchical_model.py`): neural, separate
   relation roles, dependent second read, binary page digits, null ranks, and
   reference continuation. Works with full-prefix and cached streaming forward.
3. **Conditional binary counterfactual credit** (`train_hierarchical.py`):
   targets active page bits only, with a two-trajectory conditional loss
   surrogate. No oracle gold routes enter neural inference.
4. **Versioned compositional execution backend** (`program_runtime.py`): known
   graph functions can be compiled in parallel and reused for large batches;
   updates invalidate only affected programs. This backend is a **formal
   reference/optional oracle program executor**, not an automatically learned
   text-to-graph parser.
5. **Safe generation** (`generation.py`): rejects PAD as an output symbol,
   supports EOS early stopping, an explicit max-context budget and groups
   different prompt lengths without cross-document interference.
6. **Inference-first selection** (`design_selection.py`): hard competence
   gates before a 12:1 inference/training log-cost preference.
7. **Lean 4 proof candidates** (`lean/Relational/*.lean`): causal, paging and
   composition statements; **not compiled or kernel-verified** here.

## Reproduce (CPU)

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python benchmark_v4.py --repeats 3 --output v4_results.json
python benchmark_inference_mode.py --repeats 5 --output inference_mode_results.json
python benchmark_paired.py --repeats 8 --output paired_results.json
python stress_v4.py --generated 16384 --prompt 65536 --output v4_stress.json
lake env lean lean/Relational/Basic.lean
lake env lean lean/Relational/Paged.lean
lake env lean lean/Relational/Programs.lean
```

The final three commands require an installed Lean toolchain. Running Python
successfully does not imply any Lean theorem has been accepted by the kernel.

## Minimal usage

```python
import torch
from relational_composition.model import Config
from relational_composition.hierarchical_model import PagedMultiHopLM, StreamingPagedMultiHop
from relational_composition.generation import greedy_generate

torch.set_num_threads(1)
model = PagedMultiHopLM(Config(width=24, layers=2), page_bits=20).eval()
with torch.inference_mode():
    runner = StreamingPagedMultiHop(model)
    prompt = torch.randint(1, 200, (2, 128))  # non-PAD, equal-length batch
    result = greedy_generate(runner, prompt, max_new_tokens=64,
                             eos_token_id=11, max_context_tokens=20000)
print(result.generated, result.lengths)
```

This is a random untrained model; outputs are not meaningful language.
A 20-bit page code with C=2 can address up to 2,097,152 colliding records
**at one address**. It does not mean a model can learn to retrieve them.

## Results and caution

- Python suite: 27 passed at packaging time (re-run to verify).
- 16,384 fixed continuation steps: 1,362.69 steps/s; full-prefix parity at
  8 sampled checkpoints, max absolute logit error <5e-7.
- 65,536-token prompt, then 512 steps: prefill 3.0489 s, decode 1,334.55
  token steps/s (single CPU thread, random untrained model).
- Compared with V3, V4's learned page heads add overhead. In the interleaved
  eight-repeat batch-one 512/512 comparison (`paired_results.json`), V3 reached
  ≈1,365 tok/s and V4 ≈1,274 tok/s (median paired V4 latency ≈7.7% higher).
  V4 does **not** provide a blanket throughput speedup.
- Parallel composition of known total relation functions benefits large
  repeated batches. This is not evidence the neural model learns programs.
- Running a full generation session in `torch.inference_mode()` typically
  improves PyTorch operator overhead but timings vary. See the separate
  five-repeat benchmark rather than quoting one run.

## Next scientific gates

1. Learn stable cross-mention entity identity and write organization, rather
   than only demonstrating that older pages are addressable.
2. Measure gold fact admission, page/slot accuracy, reference continuation,
   first/second-hop accuracy, and free-running answer accuracy independently.
3. Keep local, oracle-page and learned-page controls. At matched accuracy and
   budgets, compare against V3, an optimized attention baseline and a strong
   local baseline; choose on inference-first latency only **after** quality gates.
4. Verify Lean source in a pin-matched toolchain; establish a stronger
   refinement link from Python indexing to Lean denotational semantics.
5. Replace Python loops/dictionaries with fused indexed kernels for GPU/
   higher batch throughput. The reference here is not production optimized.

## Publication

No GitHub push/merge/PR creation is performed. The bundle can be reviewed
locally or integrated into a new experimental branch after explicit approval.
