# COG Unified Tiny Shakespeare 10M

A user-requested text-generation prototype from the COG-UNIFIED-001 architecture study.

**9,725,682 parameters.** Six width-400 causal recurrent blocks, a 16-slot factorized latent record board, and a shared character-language output interface. No token-to-token QK attention or borrowed SSM implementation.

This is a Shakespeare-style continuation model. The trained checkpoint and final held-out results will be attached after the current fit completes. The repository contains source and evidence; checkpoint files are delivered separately.

## Test a trained checkpoint

Put the provided `COG_UNIFIED_TINY10M.pt` file in this folder. It contains the weights, model configuration, and 65-character tokenizer.

Install the CPU runtime if it is not already available:

```sh
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
```

Start the local browser tester:

```sh
python serve.py --checkpoint COG_UNIFIED_TINY10M.pt --threads 4
```

Open **http://127.0.0.1:8765/** on the same computer. Enter a prompt such as `ROMEO:`, select the number of new characters, and generate.

For a terminal prompt loop:

```sh
python sample.py --checkpoint COG_UNIFIED_TINY10M.pt --interactive
```

For a reproducible single sample:

```sh
python sample.py --checkpoint COG_UNIFIED_TINY10M.pt --prompt "ROMEO:" --max-new-tokens 500 --temperature 0.8 --seed 42
```

The model continues text rather than following instructions. Unsupported characters are reported explicitly. Temperature 0 selects greedy output; smaller positive temperatures are more conservative.

## Data and training contract

The pinned standard English Tiny Shakespeare corpus has 1,115,394 characters. The tokenizer is fitted on training characters only.

| Split | Characters |
|---|---:|
| Training |1,003,854|
| Validation |55,770|
| Final test |55,770|

No next-character target crosses a split boundary. Training uses 256-character windows, batch 16, two accumulated microbatches, seed 20261122, FP32 AdamW, warmup and cosine decay. The planned budget is 2,000 updates/16,384,000 target presentations. Validation selects the best checkpoint; the final test is evaluated only after the declared training stop.

The free Kaggle submission was rejected because both GPU batch slots were occupied by other queued jobs. The fit proceeds on local CPU with eight threads, with a 3,600-second session bound and checkpoint continuation.

## Reproduce

```sh
python fetch_data.py
python train.py --device cpu --precision fp32 --threads 8 --batch-size 16 --gradient-accumulation 2 --context-length 256 --max-steps 2000 --eval-every 200 --minimum-steps 1000 --patience 6 --max-wall-seconds 3600 --out-dir runs/tiny10m_seed20261122
```

A paused run continues with the identical arguments plus `--resume runs/tiny10m_seed20261122/resume.pt`. Preserve its complete run directory: the best checkpoint and resume state are bound by hashes. The supplied inference checkpoint alone is not a complete optimizer-resume state.

## Scope and checks

See [ARCHITECTURE_IMPLEMENTED.md](ARCHITECTURE_IMPLEMENTED.md) for the enabled-component map. The record board learns latent VALUE/HOLD routing from next-character loss. Raw Shakespeare does not provide semantic entity, support, COPY/LINK/DELETE, or transaction labels. The richer typed executor remains part of the parent architecture study.

[REVIEW.md](REVIEW.md) records independent pre-fit admission: 13 neural-interface checks and 9 pipeline checks passed. These cover causal and incremental behavior, active gradients, data boundaries, full evaluation target coverage, and checkpoint/RNG recovery. The local web tester has a separate constructed functional check.

This single-seed prototype does not establish superiority over attention or general semantic memory competence. Keep character perplexity distinct from subword perplexity in other experiments.

[NEURAL_GUARANTEES.md](NEURAL_GUARANTEES.md) derives the conditional guarantees of the actual frozen neural source and gives concrete counterexamples to stronger claims. It separates exact-real algebra, finite numerical checks, learned behavior, and total resource use. The richer parent semantic executor is a separate implementation.

The post-fit audit and delivery helpers are prepared but run only after the declared training stop and final-test evaluation. Runtime checkpoint continuation preserves the model, training configuration, data, and source identities; allocator/thread settings and discarded interrupted work are recorded separately.

