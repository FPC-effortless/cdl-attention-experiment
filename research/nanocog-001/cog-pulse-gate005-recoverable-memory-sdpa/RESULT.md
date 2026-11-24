# COG-PULSE Gate 005: recoverable memory with matched local and sliding-window decoders

**An equal-width attention control recovered just as quickly as the local decoder.** In this fixed-token experiment, a fault became irrelevant to future logits after it left the affected layer's receptive window. The local decoder and three-token sliding-window attention matched the same clean outputs without replay for every enrolled restored query. The 128-token attention window retained the fault longer. All arms reproduced their entire clean logit traces exactly after upper-layer replay. This establishes the role of the affected window, not an attention-exclusive or attention-free-exclusive recovery mechanism.

## Method

Used the same versioned Gate 005 memory bank and rollback procedure with the six original frozen Gate 004 width-64 checkpoints (three local and three ALiBi). This is the checkpoint lineage of the historical recovery test, not the smaller width-32 models of the old decoder-only timing study. The original full-attention versus local recovery check was rerun for lineage. In the fresh comparison, only the top layer receives bank reads and the continuation tokens are forced and identical. The three arms are local (current + two previous positions), exact three-token sliding-window ALiBi (equal receptive width control), and 128-token sliding-window ALiBi. No model was trained.

Ten shared constructed sessions each use 192 prefill and 192 incremental tokens. Nine sessions inject a real deletion on one queried record, restore it with its actual receipt, and include queries at different distances from the fault. Cases straddle prefill/decode and both window boundaries. One 128-token case per seed ends before its window can be observed to expire. The bank-only path rolls back the record but leaves earlier decoder state untouched; the repair path restores an upper-layer cache snapshot and replays the affected prefix up to the rollback point. The lower layers remain unchanged. The attention path uses bounded tiles and the fused CPU Flash Attention kernel for both prefill and decode. Its contiguous K/V cache still copies on updates; it is not a production ring or paged cache.

## Recovery and answer correctness

Counts combine three frozen seed pairs running the same cases. They are not independent population samples.

| Measurement | Local | 3-token sliding attention | 128-token sliding attention |
|---|---:|---:|---:|
| Entire clean logit trace reproduced after rollback and upper replay, all sessions | 30/30 | 30/30 | 30/30 |
| Entire trace reproduced on the fault sessions alone | 27/27 | 27/27 | 27/27 |
| Exact at the bank rollback position without decoder replay | 21/27 | 21/27 | 6/27 |
| Exact logits on post-restore queries without replay | 54/54 | 54/54 | 27/54 |
| Exact on the common query subset beyond the 128-token boundary | 27/27 | 27/27 | 27/27 |
| Correct target token, with clean or fully repaired memory | 30/90 | 24/90 | 52/90 |

All 27 fault injections per arm changed logits at the fault position. No earlier logit changed. After a deletion at position p, a single bank injection is no longer visible to this upper layer at p+3 or p+128, respectively, given the same subsequent tokens. On that side of the boundary the parity was exact; three 128-token cases were right-censored, not credited as observed recovery. The no-replay behavior only concerns future fixed-token outputs; earlier affected outputs are not repaired by waiting, and this does not test autonomous continuation. Replaying 491 upper-layer positions per seed per arm (zero lower positions) restored all historical outputs, repaired stale epoch tags, and retained the unrelated record. Duplicate rollback receipts were rejected without changing the bank.

The historical linked test reproduced 144/144 exact later-query local recoveries without replay, versus 0/144 for its unbounded ALiBi control; both fully recovered with replay. In the new implementation, vectorized prefill and independent bounded incremental references agreed within 6e-6 and gave identical greedy tokens on the parity panel. The production decode adapter matched the independently clipped reference at W−1, W and W+1 for both windows; the retained K/V were exact and bounded. Profiler evidence confirmed fused CPU Flash Attention without math fallback.

## Timings and retained memory

Single-core AMD EPYC 9V74 CPU, PyTorch 2.10 CPU. Each of the ten requests was warmed once, then timed four times per seed and measured arm (120 samples per arm). Timed requests include typed input conversion, bank operations, prefill, 192 forced incremental steps, rollback and any upper replay. Reported ranges are the minima and maxima of the three seed summary statistics. Timings exclude model load, parser and tokenizer, open-loop token selection, server I/O, and the separate audit/validation work. The three-token control was used for the recovery mechanism test only, not the timing comparison.

| Timed quantity | Local | 128-token sliding attention |
|---|---:|---:|
| Complete typed request p50 | 49.7–52.0 ms | 192.6–200.0 ms |
| Complete typed request p95 | 62.6–64.0 ms | 232.9–252.9 ms |
| Prefill p50 | 3.0–3.2 ms | 6.3–6.6 ms |
| Fault requests: rollback plus upper replay p50 | 0.45–0.53 ms | 1.36–1.40 ms |
| Fault requests: rollback plus upper replay p95 | 11.3–12.4 ms | 39.3–46.1 ms |

Paired median time ratios by seed are approximately 3.86–3.92 for 128-token attention versus local. These are not equal-dependency-span or equal-accuracy models: 128-token attention answered more of the typed queries correctly. No production-serving or broad architecture speed claim follows.

The live layer caches were 1.5 KiB for local, 4.5 KiB for W3, and 192 KiB for W128. At the sampled peak before rollback, the total accounted retained state excluding model parameters was about 319, 323 and 573 KiB, respectively, after adding snapshot, retained lower activations, logits, tags, and bank/journal storage. Thus the 128× live-cache gap versus W128 was about 1.8× on these measured retained components. These component counts exclude temporary/framework allocations. Combined-process peak RSS is recorded but not attributable by arm.

## Decision

Keep the transaction and upper-only recovery tests. Do not claim that no-replay recovery distinguishes PULSE from an equally local attention decoder. The stronger next test is to compare trained, quality-matched decoders at the same effective dependency span with a bounded rollback buffer, then repeat with generated continuations or ambiguous updates. This study isolates replay mechanics but does not establish durable crash recovery, natural-language correction or learned write correctness.

Every admitted frozen seed role ran once, no protected/paid work occurred, and an independent model-free check recounted the sealed outputs. The reproducible evidence keeps the original checkpoint references separate from the new result package.
