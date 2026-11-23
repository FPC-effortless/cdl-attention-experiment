# COG Unified Tiny Shakespeare 10M

A trained, user-requested text-generation prototype from the COG-UNIFIED-001 study.

**9,725,682 parameters.** The validation-selected checkpoint is from update **1,000**. The restored training trajectory stopped at update **2,000** of its 2,000-update budget, with stop reason `validation_early_stopping`.

This is a character-level Shakespeare-style continuation model, not an instruction-following chatbot. It has 6 width-400 causal recurrent blocks and a 16-slot factorized latent VALUE/HOLD board. No token-to-token QK attention or borrowed SSM implementation is used.

## Test the delivered checkpoint

Extract `COG_TINY10M_CODE.zip`, then put the separately supplied `COG_UNIFIED_TINY10M.pt` inside its `COG_TINY10M` folder. Open a terminal in that folder. The checkpoint embeds its model configuration and 65-character tokenizer.

Install PyTorch 2.10.0 if needed. The checked training and inference environment was **Python 3.12 on Linux CPU**; macOS execution was not checked.

**Linux or Windows, CPU installation:**

```sh
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
```

**macOS, using the normal package index:**

```sh
python -m pip install torch==2.10.0
```

These installation commands follow the [official PyTorch 2.10.0 platform instructions](https://pytorch.org/get-started/previous-versions/).

Start the local browser tester:

```sh
python serve.py --checkpoint COG_UNIFIED_TINY10M.pt --threads 4
```

Open **http://127.0.0.1:8765/** on the same computer. Enter a character name or dialogue, choose a length and temperature, and generate a continuation.

For an interactive terminal:

```sh
python sample.py --checkpoint COG_UNIFIED_TINY10M.pt --interactive
```

For a reproducible sample:

```sh
python sample.py --checkpoint COG_UNIFIED_TINY10M.pt --prompt "ROMEO:" --max-new-tokens 500 --temperature 0.8 --seed 42
```

Temperature 0 selects greedy output. Unsupported characters are rejected explicitly. Sampling can repeat, lose coherence, or resemble the small training corpus.

## Observed held-out results

| Model/checkpoint | Validation NLL | Validation character perplexity | Final-test NLL | Final-test character perplexity |
|---|---:|---:|---:|---:|
| Random initialization | 4.268671 | 71.4267 | Not scored | Not scored |
| Training-frequency unigram | 3.332643 | 28.0123 | 3.361968 | 28.8459 |
| Selected COG checkpoint, update 1000 | 1.532512 | 4.6298 | 1.692501 | 5.4331 |

Validation and final test each score **55,769** next-character targets, with state reset for each nonoverlapping 256-character window. Validation selected the weights; the final test was scored after the declared training stop. Character perplexity is not directly comparable to subword perplexity.

See [RESULTS.md](RESULTS.md) for provenance, actual exposure accounting, recovery limits, runtime receipts, and post-fit checks. Generated examples are in [post-fit samples](runs/tiny10m_seed20261122/postfit_audit/samples.txt).

## Reproduce or recover the declared training run

The pinned corpus has 1,115,394 characters. The contiguous training/validation/test splits contain 1,003,854/55,770/55,770 characters. The tokenizer uses training characters only; targets never cross a split boundary.

Fetch and verify the corpus:

```sh
python fetch_data.py
```

The following command derives its thread count and allocator environment from the final successful launcher receipt. For a fresh reproduction, choose a new output directory. If it includes `LD_PRELOAD`, first verify that the named allocator library exists on that Linux machine:

```sh
MALLOC_ARENA_MAX=2 python train.py --device cpu --precision fp32 --threads 2 --batch-size 16 --gradient-accumulation 2 --context-length 256 --max-steps 2000 --eval-every 200 --minimum-steps 1000 --patience 6 --max-wall-seconds 900 --log-every 1 --out-dir runs/tiny10m_seed20261122
```

For continuation from a matching interrupted checkpoint pair:

```sh
MALLOC_ARENA_MAX=2 python train.py --device cpu --precision fp32 --threads 2 --batch-size 16 --gradient-accumulation 2 --context-length 256 --max-steps 2000 --eval-every 200 --minimum-steps 1000 --patience 6 --max-wall-seconds 900 --log-every 1 --out-dir runs/tiny10m_seed20261122 --resume runs/tiny10m_seed20261122/resume.pt
```

`MALLOC_ARENA_MAX=2` records the allocator setting used in successful bounded recovery. It is an observed runtime mitigation, not a guarantee against future out-of-memory termination. The environment prefix is a Linux shell form. A recorded `LD_PRELOAD` library path is machine-specific and should not be copied to an incompatible system. The earlier two-thread arena-limited recovery command was:

```sh
MALLOC_ARENA_MAX=2 python train.py --device cpu --precision fp32 --threads 2 --batch-size 16 --gradient-accumulation 2 --context-length 256 --max-steps 2000 --eval-every 200 --minimum-steps 1000 --patience 6 --max-wall-seconds 120 --log-every 1 --out-dir runs/tiny10m_seed20261122 --resume runs/tiny10m_seed20261122/resume.pt
```

On Windows, ordinary CPU inference does not need these training allocator settings.

The separate `COG_UNIFIED_TINY10M_RESUME.zip` contains matching `best.pt` and `resume.pt`, optimizer/RNG state, and run evidence, under the same `COG_TINY10M` folder. Extract it alongside this source package and fetch the corpus before resuming. The inference checkpoint alone does not contain the optimizer state. A completed resume checkpoint has already reached its declared stop; extending the budget requires a separately declared continuation configuration.

## Scope

The latent board learns routing and VALUE/HOLD updates from next-character prediction. Tiny Shakespeare provides no semantic entity, support, COPY/LINK/DELETE, or transaction labels. The richer typed executor and its abstract proofs are not certified neural capabilities of this checkpoint. See [ARCHITECTURE_IMPLEMENTED.md](ARCHITECTURE_IMPLEMENTED.md).

[NEURAL_GUARANTEES.md](NEURAL_GUARANTEES.md) studies the implemented network's causal dependencies, conditional recurrence bounds, latent update semantics, and bounded logical inference state. These are source-derived structural arguments under stated arithmetic and interface assumptions; they do not prove learned semantic accuracy or verify PyTorch's numerical kernels.

This one-seed run establishes a trained, testable artifact and its measured results on this corpus. It does not establish superiority over attention, broad language competence, or general semantic memory. Optional private online notebook submission was not verified; the supported delivery is the local browser and CLI.

The source/evidence review is tracked in [PR #93](https://github.com/FPC-effortless/cdl-attention-experiment/pull/93). Weights are delivered separately.
