# COG-LM-002 research results

**Decision. Hold the parameter count.** At 2,097,152 training targets, the fully trained Mamba-2 plus lookup reached **42.10 Wikipedia perplexity**. COG plus lookup reached **50.8989** and COG plus the new revision bank reached **50.8994**. The bank resolved zero of eight paired paraphrased corrections at each requested delay. The predeclared case for scaling the learned bank failed. The trained COG checkpoint remains a usable autoregressive model.

## Training and held-out prediction

| Optimization | Model | Parameters | WikiText NLL ↓ | Perplexity ↓ | MultiWOZ NLL ↓ | LAMBADA exact ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| Frozen sequence model | Mamba-2 + lookup | 31,356,865 | 4.0863 | 59.52 | 5.9322 | 28/512 |
| Frozen sequence model | COG + lookup | 30,930,625 | 4.5004 | 90.05 | 6.1960 | 31/512 |
| Frozen sequence model | COG + revision + lookup | 31,014,179 | 4.4996 | 89.98 | 6.2015 | 31/512 |
| Frozen sequence model | Same COG, revision bank disabled | 31,014,179 | 4.5001 | 90.02 | 6.1943 | 31/512 |
| All parameters trained | Mamba-2 + lookup | 31,356,865 | 3.7400 | 42.10 | 5.8031 | 30/512 |
| All parameters trained | COG + lookup | 30,930,625 | 3.9298 | 50.90 | 5.8875 | 32/512 |
| All parameters trained | COG + revision + lookup | 31,014,179 | 3.9299 | 50.90 | 5.8854 | 32/512 |
| All parameters trained | Same COG, revision bank disabled | 31,014,179 | 3.9299 | 50.90 | 5.8854 | 32/512 |

The first stage continues the published 001 seed-17 parameters through 524,288 targets. The new stage uses 1,572,864 distinct additional target positions and 1,536 AdamW updates in each fit. Each fitted system has seen 2,097,152 target presentations, covering 1,842,778 distinct training positions in total. The training source holds 3,583,480 positions. This is one English corpus, one seed and 128-token training windows. The frozen regime fits the language adapter, lookup gate and optional bank. The joint regime also updates every pretrained parameter. Each COG regime shares its starting backbone and initial shared readout across the two trained COG alternatives. The disabled row is an inference intervention on the trained revision model. WikiText and MultiWOZ next-token results reuse the known 001 test panels with 15,616 and 12,288 scored targets respectively. The LAMBADA panel is the next 512 passages in the pinned hash ordering, disjoint from the 001 sample. Exact completion requires every target subword to be the highest-probability vocabulary item.

## Closing the original baseline gap

| 524,288-target model | Wikipedia NLL | Dialogue NLL |
| --- | --- | --- |
| Mamba-2 | 4.2954 | 6.4342 |
| Mamba-2 with validation-calibrated lookup | 4.1797 | 5.8979 |
| Original 001 full COG | 4.6377 | 6.1643 |

Both continuations used identical training offsets. WikiText validation selected a 0.35 lookup weight for Mamba-2 from the original grid. Its Wikipedia perplexity was 65.35. The Mamba-plus-lookup minus original-COG loss difference was −0.458 nats/token on Wikipedia and −0.266 on dialogue. Paired 95% intervals were [−0.489, −0.427] and [−0.288, −0.245], respectively. The longer-stage comparisons fit the lookup gate from training tokens in all arms.

## What the additional records contributed

| Regime | Comparator | Full minus comparator, WikiText | Full minus comparator, MultiWOZ |
| --- | --- | --- | --- |
| Frozen | Mamba-2 + lookup | +0.4133 [+0.3846, +0.4413] | +0.2693 [+0.2479, +0.2907] |
| Frozen | COG + lookup | -0.0008 [-0.0016, +0.0000] | +0.0055 [+0.0040, +0.0070] |
| Frozen | Same COG, revision bank disabled | -0.0005 [-0.0011, -0.0000] | +0.0072 [+0.0059, +0.0086] |
| Joint | Mamba-2 + lookup | +0.1899 [+0.1650, +0.2139] | +0.0823 [+0.0598, +0.1045] |
| Joint | COG + lookup | +0.0000 [-0.0003, +0.0003] | -0.0021 [-0.0025, -0.0017] |
| Joint | Same COG, revision bank disabled | +0.0000 [-0.0000, +0.0000] | -0.0000 [-0.0000, +0.0000] |

Negative loss differences favor the revision model. Intervals use 5,000 paired article or dialogue resamples conditional on the fitted weights. The different-architecture comparisons include each backbone and its optimization history. Only the paired COG fits share an initialization. The cost-matched intervention executes memory routing, reads and writes, then masks the read contribution. Its predictions matched disabling the bank with a maximum log-probability discrepancy of 0.0 on the 122 Wikipedia windows. The simpler production models save the cost of those operations.

## Delayed corrections and natural long context

| Fully trained model / query | Immediate | 256 | 1,024 | 4,096 | 16,384 |
| --- | --- | --- | --- | --- | --- |
| Mamba-2 + lookup / literal | 10/16 · 7/8 | 10/16 · 7/8 | 0/16 · 0/8 | 0/16 · 0/8 | 0/16 · 0/8 |
| Mamba-2 + lookup / paraphrase | 0/16 · 0/8 | 0/16 · 0/8 | 0/16 · 0/8 | 0/16 · 0/8 | 0/16 · 0/8 |
| COG + lookup / literal | 14/16 · 7/8 | 14/16 · 7/8 | 0/16 · 0/8 | 0/16 · 0/8 | 0/16 · 0/8 |
| COG + lookup / paraphrase | 0/16 · 0/8 | 0/16 · 0/8 | 0/16 · 0/8 | 0/16 · 0/8 | 0/16 · 0/8 |
| COG + revision + lookup / literal | 14/16 · 7/8 | 14/16 · 7/8 | 0/16 · 0/8 | 0/16 · 0/8 | 0/16 · 0/8 |
| COG + revision + lookup / paraphrase | 0/16 · 0/8 | 0/16 · 0/8 | 0/16 · 0/8 | 0/16 · 0/8 | 0/16 · 0/8 |
| Same COG, revision bank disabled / literal | 14/16 · 7/8 | 14/16 · 7/8 | 0/16 · 0/8 | 0/16 · 0/8 | 0/16 · 0/8 |
| Same COG, revision bank disabled / paraphrase | 0/16 · 0/8 | 0/16 · 0/8 | 0/16 · 0/8 | 0/16 · 0/8 | 0/16 · 0/8 |

Each cell gives full-vocabulary top-1 successes and then counterfactual pairs with correct preference in both directions. Four fixed phrasings, two entity queries and two assignment directions yield 16 trials and eight pairs per distance. No probe labels were used for training. Gold and stale values are single tokens. Delays count intervening natural tokens after the correction. The table is an intervention test with modest template diversity.

For the joint revision model, changing only the two initial assignments altered at least one stored value coordinate by up to 0.000390181, 0.000203476, 0.000202561, 0.000194691, 0.000123858 at these five delays. This measures the residual effect of the changed prefix on stored tensors. The counterfactual answer score measures whether the readout used it appropriately.

| Joint model, same final 128 targets | 256 context | 1,024 | 4,096 | 16,384 |
| --- | --- | --- | --- | --- |
| Mamba-2 + lookup | 3.6917 | 3.6900 | 3.6899 | 3.6899 |
| COG + lookup | 3.8605 | 3.8574 | 3.8574 | 3.8574 |
| COG + revision + lookup | 3.8604 | 3.8574 | 3.8574 | 3.8574 |
| Same COG, revision bank disabled | 3.8604 | 3.8573 | 3.8573 | 3.8573 |

Natural long-context scores use all seven held-out Wikipedia articles long enough for every condition. Each column scores the same 896 targets. This panel measures prediction conditional on more context and does not label which facts each continuation requires.

### What happened inside the bank

A post-hoc trace followed the first 1,024 tokens of each of 58 held-out articles. Across 59,392 token reads, the jointly trained bank's median contribution had **0.011%** of the hidden-state norm. Its median write strength across 3,712 block commits was **0.000596** on a 0–1 scale. The frozen-backbone bank had a median relative read norm of 0.313% and a median write strength of 0.01200. This directly measures weak use of the learned path on natural text. It does not identify whether the limiting factor was the 128-token training horizon, the objective, initialization, routing or optimization.

## Authentic revisions in dialogue

| Joint model | Prefers update after | Switches old → new | Switches new → old | Exact updated answer | Median log-odds shift |
| --- | --- | --- | --- | --- | --- |
| Mamba-2 + lookup | 59 | 20 | 1 | 0 | 0.165 |
| COG + lookup | 52 | 19 | 2 | 0 | 1.119 |
| COG + revision + lookup | 52 | 19 | 2 | 0 | 1.117 |
| Same COG, revision bank disabled | 52 | 19 | 2 | 0 | 1.117 |

The untouched second MultiWOZ test shard yielded 137 eligible changed requests in 110 real dialogues. Each old value had appeared explicitly in user speech and the correction explicitly mentioned its replacement. The scorer compares complete answers including final punctuation. It uses the same old and new candidates before and after the correction, which reduces static answer-length bias in the switch metric. The simple latest-mention heuristic preferred the updated value on 133/137 post-change cases and got both stages right on 122. This heuristic receives the two candidate strings. Exact generation uses the full vocabulary and receives neither answer. The corpus supplies the conversations, and fixed evaluation questions specify the requested attribute.

## CPU inference economics

| Model | Load ms | Encode ms | Prefill ms | Decode tok/s | Live state KiB | Fresh request s | Peak RSS MiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Mamba-2 + lookup | 848.3 | 2.04 | 523.9 | 149.2 | 779.1 | 3.64 | 510 |
| COG + lookup | 735.7 | 1.92 | 573.6 | 120.0 | 72.5 | 2.80 | 481 |
| COG + revision + lookup | 800.3 | 1.45 | 323.1 | 178.6 | 74.5 | 2.60 | 481 |
| Same COG, revision bank disabled | 704.7 | 1.43 | 308.7 | 201.1 | 72.5 | 2.55 | 481 |

Single-stream float32 inference, two PyTorch threads, CPU reference implementations on Intel Xeon Platinum 8573C. Warm-request columns use medians of three timed trials after warm-up at the nominal 1,024-token prompt. Raw-text retokenization lengths are recorded per trial. Prefill includes both memory updates and the final vocabulary projection. Decode includes 32 greedy choices, all state updates and output decoding. Live state combines backing tensor storage and an estimate of the Python table heap. Fresh request time is the median of three new Python processes from launch through exit, including import, weight loading, encoding, prefill, generating 32 tokens and detokenization. Filesystem cache conditions were uncontrolled. Peak RSS is process high-water memory over the full sweep, including the 16,384-token prompt. These figures are specific to this runtime. They do not price GPU service or training.

| Joint model | 16,384 prefill s | 16,384 request s | Weights MiB | 1024+32 bank ms | 1024+32 lookup ms |
| --- | --- | --- | --- | --- | --- |
| Mamba-2 + lookup | 8.56 | 8.90 | 119.6 | 0.00 | 1.37 |
| COG + lookup | 6.35 | 6.54 | 118.0 | 0.00 | 1.09 |
| COG + revision + lookup | 5.53 | 5.74 | 118.3 | 18.41 | 1.13 |
| Same COG, revision bank disabled | 5.07 | 5.26 | 118.3 | 0.00 | 0.98 |

Module timings use a separate instrumented run. Lookup timing measures the Python table operations. Tensor conversion, mixture arithmetic and all remaining computation are included in end-to-end times. All systems maintain the same capacity-256 exact table. The two COG revision conditions share the same additional parameters.

### Order-balanced timing at 1,024 tokens

| Model | Prefill ms | Decode tokens/s | Warm request ms |
| --- | --- | --- | --- |
| Mamba-2 + lookup | 516.0 | 157.3 | 721.6 |
| COG + lookup | 334.8 | 183.5 | 505.4 |
| COG + revision + lookup | 344.1 | 177.3 | 535.3 |
| Same COG, bank disabled | 335.4 | 186.8 | 509.4 |

The serial sweep above showed timing variation between runs. This follow-up loaded all four models, warmed each once and rotated their order across eight repetitions, including reverse order. Use these medians for the 1,024-token speed comparison. Every generated ID agreed with the serial benchmark. Wall times include the lookup and bank updates, input encoding and final text decoding.

## Verification and scope

Thirteen CPU regression checks passed. The audit matched all six fitted model identities, their published ancestors, the data order and the exact number of distinct positions. Fresh loads verified chunked continuation, with maximum log-probability error 7.63e-06. Early concurrent full-training attempts were killed under memory pressure before saving a complete fit. Restarted fits used the original seed, parents and token budget. Only successful complete runs enter the tables. Checkpointed recovery and smaller microbatches reduced concurrency pressure. Data lineage and reproduction commands are in README.md, sources and raw receipts are included in the research package.

## Next research decision

Keep these six fits as references. Before increasing model size, change the training experience at the same scale. Compare natural-text-only continuation with an equal-target, interleaved curriculum containing document-length revisions and delayed references. Include the same curriculum and supervision for both lookup-only backbones. Reserve unseen entities, phrasings and dialogue sources for a new sealed evaluation. Require a paired revision gain over both simpler systems while preserving natural-text loss and measuring the end-to-end inference cost. This cycle established that more language training improved both backbones. It did not establish that a learned revision bank pays for itself.
