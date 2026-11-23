# COG Unified Tiny Shakespeare 10M — completed-run results

## Outcome and selection

The run completed with status **completed** and stop reason **`validation_early_stopping`**. The final restored trajectory reached **2,000** successful optimizer updates; validation selected update **1,000**. The delivered inference artifact contains that selected checkpoint, not necessarily the last optimizer state.

- Actual total and trainable parameters: **9,725,682**.
- Model identifier: `cog-unified-tiny-latent-v1`.
- Seed: **20261122**; precision: **fp32**.
- Frozen configuration: 6 layers, width 400, FFN multiplier 2, dropout 0.1, 4×4 latent addresses, payload width 64.
- Selected checkpoint SHA-256: `60a01cc2aab00c96e0186ea211ecab6e83117f2efb86f0ec5f172b051c2cdeb8`.
- Matching final resume checkpoint SHA-256: `10af912effceb16a64a52cbc2a21d2ef6ab619371ca34dba6629828ec0f1175e`.
- Post-fit audit: **17/17 checks passed**.

## Measured language results

| Measurement | Mean NLL, nats/character | Bits/character | Character perplexity | Next-character accuracy | Scored targets |
|---|---:|---:|---:|---:|---:|
| Random initialization, validation | 4.268671 | 6.158391 | 71.4267 | 1.363% | 55,769 |
| Train-frequency unigram, validation | 3.332643 | 4.807988 | 28.0123 | 15.200% | 55,769 |
| Selected COG, validation | 1.532512 | 2.210948 | 4.6298 | 55.203% | 55,769 |
| Train-frequency unigram, final test | 3.361968 | 4.850294 | 28.8459 | 14.596% | 55,769 |
| Selected COG, final test | 1.692501 | 2.441762 | 5.4331 | 52.359% | 55,769 |

Each split's next-character targets were scored once, using reset state in nonoverlapping 256-character windows. The partial final window is masked correctly. Native continuous-state generation has a different context policy from this fixed-window NLL measurement.

The unigram control uses only training-character frequencies. It is a weak sanity baseline, not a competitive neural comparator. No trained attention baseline or architecture ranking is established by this run. The independent audit validates the recorded final-test receipt and denominators; it does not recompute the test or use it to select another checkpoint.

## Data provenance

- Dataset: standard English Tiny Shakespeare from `karpathy/char-rnn`.
- Pinned source commit: `370cbcd448eb7daf32f21a6be560b70e0b33c4e3`.
- [Exact source text](https://raw.githubusercontent.com/karpathy/char-rnn/370cbcd448eb7daf32f21a6be560b70e0b33c4e3/data/tinyshakespeare/input.txt).
- Corpus SHA-256: `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.
- Corpus characters/bytes: **1,115,394 / 1,115,394**.
- Tokenizer: **65** reversible characters, fitted on training text only.
- Tokenizer SHA-256: `ce557fac821f26589a6d1d421f6c111b6c909458d9554c91171cb76be4a5383b`.
- Split policy: **contiguous 90% train / 5% validation / 5% test**.
- Training/validation/test characters: **1,003,854 / 55,770 / 55,770**.
- Training/validation/test next-character targets available: **1,003,853 / 55,769 / 55,769**.
- No normalization, unknown-character substitution, or cross-split prediction target was introduced. Held-out vocabulary inspection was only a coverage check, not tokenizer fitting.

The raw corpus is intentionally absent from the delivery archives. `fetch_data.py` retrieves the pinned bytes and verifies their hash.

## Frozen training and exposure accounting

The optimizer was AdamW with learning rate 0.0003, betas [0.9, 0.95], weight decay 0.1, gradient clipping 1.0, 100 warmup updates, and cosine decay to 0.03 of peak learning rate. Each attempt uses 16 sequences × 2 accumulated microbatches × 256 characters: **8,192 target presentations**.

- Planned maximum successful updates: **2,000**.
- Actual successful updates on the restored trajectory: **2,000**.
- Recorded optimizer attempts on that trajectory: **2,000**.
- Skipped updates on that trajectory: **0**.
- Recorded target presentations on the restored trajectory: **16,384,000**.
- Equivalent training-target passes: **16.321115**.
- Target presentations at the selected checkpoint: **8,192,000**.
- Recorded optimizer-attempt time on the restored trajectory: **12407.718 seconds** (206.795 minutes).

Presentation counts are repeated exposures, not new unique corpus tokens. Recorded optimizer-attempt time excludes validation, checkpoint I/O, initialization, and other overhead. It also does not recover uncheckpointed work discarded by process termination.

Validation is evaluated every 200 successful updates and selects the strict lowest observed NLL. The declared early-stop rule has patience 6, minimum step 1000, and improvement threshold 0.001. The reported stop reason above determines whether the maximum budget or that rule ended this run.

## Interruptions, recovery, and limits of cost accounting

There are **2 recorded exit-137 interruptions**. The preserved evidence does not establish an exact process-specific cause for either original termination. The second receipt records system OOM activity, and the memory diagnostic reports an OOM-kill counter of 2. These observations support investigating memory pressure; they do not prove which process caused or received each OOM event.

- Exit-137 interruption 1: last matching checkpoint **400**. Cause field: not established; no trainer failure/result file written. Uncheckpointed work: unknown; none logged after step400.
- Exit-137 interruption 2: last matching checkpoint **400**. Cause field: exit137 plus system OOM activity; process-specific kill attribution not available. Uncheckpointed work: unknown; no update420 log.

There are **1 separately recorded intentional runtime stops**. These are distinct from the exit-137 events:

- Intentional stop 1: tool exit code **130**, retained checkpoint **419**, last discarded logged update **440**. Reason: 8-thread allocator-limited continuation stalled for several minutes without another10-update log; stopped for allocator recovery. At least **21 completed updates** were abandoned; additional uncheckpointed work is unknown beyond last log. Exact original metrics: `metrics_before_interruption_21959.jsonl`; discarded tail: `discarded_metrics_21959.jsonl`.

There are **1 separately recorded runtime stops without an observed exit status**. These are not assigned an exit-137 or intentional-stop classification:

- Unobserved stop 1: launcher `1795425698-2` has no recorded exit receipt; its exit status and complete runtime cost are **unknown**. Retained checkpoint **1800**; last discarded logged update **1850**. Observations: Previous worker unavailable after a conversation interruption; metrics/checkpoints remained unchanged for over45 minutes and the former trainer resident memory footprint was absent.. At least **50 completed updates** were abandoned; additional uncheckpointed work is unknown beyond the last logged update. Exact original metrics: `metrics_before_interruption_48409.jsonl`; discarded tail: `discarded_metrics_48409.jsonl`.

The archived discarded tails preserve the abandoned logged updates, while canonical metrics were rolled back to the corresponding retained checkpoints and include explicit trajectory-recovery events. Discarded updates are excluded from the restored-trajectory counters above.

The independent archive audit establishes at least **71 discarded completed updates**, or **581,632 discarded target presentations**. Adding these documented discarded presentations to the restored trajectory yields a **lower bound of 16,965,632 physical target presentations**. This is a lower bound, not an exact total; additional unlogged work remains unknown.

Both model/optimizer state and the saved Python, Torch, and batch-sampler RNG trajectory were restored from matching snapshots. Restarts are continuations of one selected seed, not independent fitted seeds. The archive-backed interruption receipts give lower bounds on their discarded completed updates; any additional work after each last observed log is **unknown**. Work discarded around the earlier unlogged kills is also **unknown**. Total physical target presentations and compute time across those discarded attempts are therefore **not established**. The displayed exposure/time counters describe the restored trajectory only.

The initial allocator-bounded recovery used `MALLOC_ARENA_MAX=2` without editing the frozen model, data, or trainer. A two-thread bounded continuation was observed before later launcher continuations; any later jemalloc configuration is reported directly from their environment receipts below. Successful return codes and measured maximum resident sizes belong to the individual receipts below; they are not a general OOM guarantee.

### Trainer session records

| Session | Resume update | CPU threads | Requested wall cap, seconds | Python / Torch |
|---|---:|---:|---:|---|
| 1 | 0 | 8 | 3600 | 3.12.14 / 2.10.0+cpu |
| 2 | 400 | 8 | 1200 | 3.12.14 / 2.10.0+cpu |
| 3 | 400 | 2 | 120 | 3.12.14 / 2.10.0+cpu |
| 4 | 419 | 8 | 1200 | 3.12.14 / 2.10.0+cpu |
| 5 | 419 | 4 | 120 | 3.12.14 / 2.10.0+cpu |
| 6 | 431 | 2 | 120 | 3.12.14 / 2.10.0+cpu |
| 7 | 442 | 2 | 900 | 3.12.14 / 2.10.0+cpu |
| 8 | 575 | 2 | 900 | 3.12.14 / 2.10.0+cpu |
| 9 | 718 | 2 | 900 | 3.12.14 / 2.10.0+cpu |
| 10 | 856 | 2 | 900 | 3.12.14 / 2.10.0+cpu |
| 11 | 986 | 2 | 900 | 3.12.14 / 2.10.0+cpu |
| 12 | 1116 | 2 | 900 | 3.12.14 / 2.10.0+cpu |
| 13 | 1260 | 2 | 900 | 3.12.14 / 2.10.0+cpu |
| 14 | 1405 | 2 | 900 | 3.12.14 / 2.10.0+cpu |
| 15 | 1527 | 2 | 900 | 3.12.14 / 2.10.0+cpu |
| 16 | 1659 | 2 | 900 | 3.12.14 / 2.10.0+cpu |
| 17 | 1785 | 2 | 900 | 3.12.14 / 2.10.0+cpu |
| 18 | 1800 | 2 | 900 | 3.12.14 / 2.10.0+cpu |
| 19 | 1956 | 2 | 900 | 3.12.14 / 2.10.0+cpu |

### Completed wrapper receipts

| Receipt | Return code | Observed wall duration, seconds | Maximum child RSS, GiB | Recorded allocator environment |
|---|---:|---:|---:|---|
| 1: bounded_allocator_recovery | 0 | 131.213 | 3.751 | MALLOC_ARENA_MAX=2 |
| 2: launcher_end: 1795415900-2 | 0 | 132.353 | 2.173 | LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2; MALLOC_ARENA_MAX=2 |
| 3: launcher_end: 1795416061-2 | 0 | 131.980 | 2.173 | LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2; MALLOC_ARENA_MAX=2 |
| 4: launcher_end: 1795416263-2 | 0 | 913.198 | 3.787 | MALLOC_ARENA_MAX=2 |
| 5: launcher_end: 1795417210-2 | 0 | 913.679 | 3.732 | MALLOC_ARENA_MAX=2 |
| 6: launcher_end: 1795418140-2 | 0 | 910.820 | 3.781 | MALLOC_ARENA_MAX=2 |
| 7: launcher_end: 1795419092-2 | 0 | 917.604 | 3.726 | MALLOC_ARENA_MAX=2 |
| 8: launcher_end: 1795420028-2 | 0 | 912.594 | 3.761 | MALLOC_ARENA_MAX=2 |
| 9: launcher_end: 1795421002-2 | 0 | 910.380 | 3.757 | MALLOC_ARENA_MAX=2 |
| 10: launcher_end: 1795421936-2 | 0 | 911.059 | 3.781 | MALLOC_ARENA_MAX=2 |
| 11: launcher_end: 1795422865-2 | 0 | 912.932 | 3.732 | MALLOC_ARENA_MAX=2 |
| 12: launcher_end: 1795423818-2 | 0 | 911.951 | 3.732 | MALLOC_ARENA_MAX=2 |
| 13: launcher_end: 1795424772-2 | 0 | 917.451 | 3.744 | MALLOC_ARENA_MAX=2 |
| 14: launcher_end: 1795429402-5 | 0 | 914.184 | 3.751 | MALLOC_ARENA_MAX=2 |
| 15: launcher_end: 1795430345-5 | 0 | 288.798 | 3.751 | MALLOC_ARENA_MAX=2 |

The wrapper durations include their observed command lifetime. They cover only these recorded invocations and must not be summed into an asserted whole-task or all-attempt compute bill. The final successful launcher receipt is included in the archive. `sessions.jsonl`, `launcher_sessions.jsonl`, `allocator_recovery_receipt.json`, `interruption_events.jsonl`, and `evidence/resume_memory_diagnosis.json` preserve the evidence.

### Launchers without an exit receipt

- Launcher `1795425698-2`: preserved start and log `launcher_1795425698-2.log`; no observed end time, exit code, complete wall duration, or peak child RSS. Its interruption receipt is session `48409`.

These preserved starts are excluded from the completed-wrapper table. No missing end receipt or resource measurement has been synthesized.

The read-only recovery observations are preserved in `evidence/recovery_observation_48409.json`, including the limits of process visibility. `evidence/recovery_lock_receipt_48409.json` records the restart and the checked external process-owner lock. Its status was **running at capture**; it is runtime-guard evidence, not a completed-run receipt. The final successful launcher exit remains the completion evidence.

The optional scalar-replay receipt in `evidence/resume_replay_48409.json` compares logged training quantities across recovery. Its scope is restricted to those logged scalars; it does not establish full weight/RNG tensor identity, a new independent fit, or a benchmark speedup.

The trainer's last-session wall-time field is **269.189 seconds**, recorded before its final test evaluation. Use the final wrapper receipt for that command's full observed lifetime.

## Post-fit review and generation

The completed audit reports **17/17 passing checks**. It validates source/data identities, parameter count, finite weights, the matching best/resume pair, validation selection, declared stopping, restored-trajectory exposure accounting, held-out summaries, and reproducible generation.

The fixed sample prompts were **ROMEO:** and **First Citizen:**, with temperature 0.8 and seeds 42/43. Their full outputs and timing are in `postfit_audit/samples.json` and `samples.txt`. The checked sample hash is `86cd60100cafb79d052f001563c89fe2a0feb883860d92b7954f9078e57f56f2`. Samples demonstrate executable continuation; they do not certify factual or semantic understanding.

## Memory intervention

The optional validation-only inference intervention recorded native NLL **1.532512** and memory-disabled NLL **1.532701**, a disabled-minus-native difference of **+0.000189**. Removing the trained board contribution worsened this checkpoint's validation prediction quality in this replay.

This is an inference intervention on one selected checkpoint. It is not a retrained local-only ablation, an attention comparison, or proof of semantic binding. The final test was not used for this diagnostic. See `runs/tiny10m_seed20261122/postfit_audit/memory_intervention.json`.

## Source identities and reproduction

| Source | SHA-256 |
|---|---|
| `model.py` | `f96bd967b4bc4ece9d2750a5cf1afdf1cc819a6b007e2d638af633376c7eccaa` |
| `train.py` | `52751e1a7d383a8bb809835d1d528ee02fa4abbbe8439a5ef0918186ce3bedf6` |
| `data.py` | `5c53755333460b7fdb35e0f9a5f88b2b3c7e3f37674ef92df506164728dad69d` |

All four core source files were independently fetched from published commit [`155bb47b7421965474acb4d67ac7edb3aad3f40c`](https://github.com/FPC-effortless/cdl-attention-experiment/commit/155bb47b7421965474acb4d67ac7edb3aad3f40c) and matched their local Git blob and SHA-256 identities. `remote_source_verification.json` preserves that read-only verification.

For local interactive use, follow [README.md](README.md). For an interrupted matching run, the recovery command is:

```sh
MALLOC_ARENA_MAX=2 python train.py --device cpu --precision fp32 --threads 2 --batch-size 16 --gradient-accumulation 2 --context-length 256 --max-steps 2000 --eval-every 200 --minimum-steps 1000 --patience 6 --max-wall-seconds 900 --log-every 1 --out-dir runs/tiny10m_seed20261122 --resume runs/tiny10m_seed20261122/resume.pt
```

The finalized resume archive contains the selected `best.pt`, final `resume.pt`, and all whitelisted run metadata/evidence. It contains no source code, credentials, or raw corpus. The source archive contains an explicit whitelist of source, reports, checks, and data metadata; it contains no checkpoint weights, credentials, or raw corpus. Neither archive creates a hosted service.

The separate [implemented neural guarantee study](NEURAL_GUARANTEES.md) analyzes prefix causality, conditional local contraction, hard VALUE/HOLD semantics, and the logical state-size bound. It distinguishes exact-real arguments from floating-point execution and excludes logits, temporary tensors, retained histories, and autograd graphs from the small state count. These structural arguments do not establish learned entity binding, useful board content, or generated-language correctness.

This is a single-seed, single-corpus exploratory language model. The exact typed semantic executor from COG-UNIFIED-001 is not fully instantiated or qualified by unannotated Shakespeare training. Attention superiority, unrestricted no-forgetting, semantic COPY/LINK, and broad language capability remain outside this result.
