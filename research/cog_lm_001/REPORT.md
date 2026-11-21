# COG-LM-001 · a 30.9M-parameter attention-free language model

The published checkpoint has **30,924,857 parameters** and received **524,288 next-token training targets**. It generates directly from a learned 2,048-token byte-level BPE. A document can be suspended and resumed with recurrent state, slot and link contents, the observed-successor index and sampling RNG intact. On unseen Wikipedia its final token NLL was **4.638**, or perplexity **103.3**. This is an early English base LM with a very short pretraining history.

## Matched language comparisons

All values below are negative log likelihood in natural units per tokenizer token. Lower is better. The primary in-domain panel has 15,616 targets from 61 held-out WikiText articles. Transfer has 12,288 targets from 96 MultiWOZ conversations. Every arm within a panel saw exactly the same ordered training windows. Test windows were absent as exact 129-token strings in the training articles. The tokenizer was fit on the full training split before any short-run sampling.

| 30M arm | Training targets | Wikipedia NLL | MultiWOZ NLL | Wikipedia with same lookup¹ | MultiWOZ with same lookup¹ |
| --- | --- | --- | --- | --- | --- |
| Full COG | 65,536 | 5.866 | 6.489 | built in | built in |
| COG backbone | 65,536 | 6.087 | 7.070 | 5.874 | 6.474 |
| Mamba-2 mixer | 65,536 | 5.562 | 6.800 | 5.378 | 6.248 |
| Transformer | 65,536 | 6.403 | 7.168 | 6.180 | 6.581 |
| Full COG | 524,288 | 4.638 | 6.164 | built in | built in |
| COG backbone | 524,288 | 4.801 | 6.759 | 4.655 | 6.158 |

¹ The same 256-entry latest-successor index is applied to the frozen control. A single mixture weight is calibrated on WikiText validation and frozen for both test panels. COG learns its own mixture gate from training tokens. Only COG and its backbone received the longer training budget. Mamba-2 and Transformer scores from their shorter runs cannot rank the longer checkpoint.

Across the three compact seeds the mean COG-minus-backbone Wikipedia difference was **-0.1902 [-0.2322, -0.1546]**. At short 30M scale it was **-0.2203 [-0.2636, -0.1836]**, and after continuation it was **-0.1630 [-0.1985, -0.1325]**. Brackets are 95% paired article-bootstrap intervals from 5,000 draws. Compact seed values are averaged within article before resampling. These intervals cover test article variation conditional on the fitted models.

Against the backbone with the same lookup, the longer COG difference was **-0.0173 [-0.0220, -0.0127]** on Wikipedia and **+0.0064 [+0.0009, +0.0121]** on transfer. At the original matched 30M four-way budget the lowest Wikipedia loss came from Mamba-2 with the cache (5.378).

## What contributed the improvement?

| Compact arm | Seed 17 Wikipedia | Seed 29 Wikipedia | Seed 43 Wikipedia | Mean MultiWOZ |
| --- | --- | --- | --- | --- |
| Full COG | 5.676 | 5.683 | 5.667 | 6.416 |
| Successor lookup only | 5.656 | 5.657 | 5.642 | 6.399 |
| Slot/link bank only | 5.897 | 5.897 | 5.876 | 7.005 |
| COG backbone | 5.871 | 5.872 | 5.855 | 6.973 |
| Mamba-2 mixer | 5.740 | 5.718 | 5.723 | 6.858 |
| Transformer | 6.233 | 6.100 | 6.154 | 7.097 |

The follow-up retrained component ablations at the same compact budget. The successor-only model beat full COG on Wikipedia in all three fits. The full-minus-successor-only paired difference was **+0.0238 [+0.0207, +0.0270]**. Slot/link-only was worse than the backbone on Wikipedia at every seed. Removing slots after joint training hurts held-out loss, but retraining with no slots improves it. A frozen-removal result alone would have overstated their value.

Swapping complete bank contents between different articles at the same token positions retains the original queries, writes and sequence backbone. At the longer checkpoint, foreign-minus-intact NLL on Wikipedia was **+0.0056**, with a 95% interval of +0.0039 to +0.0076. Transferring exact copied token successors and transferring the meaning of a revised statement are separate outcomes.

## Changed facts and discourse completion

In matched synthetic prompt pairs, the old and new answer directions trade places. Insertions are natural held-out text. Each cell shows full-vocabulary top-1 success on 16 cases at that delay. The model was trained on windows of 128 tokens.

| Longer COG test | 64 intervening tokens | 256 intervening tokens | 1,024 intervening tokens |
| --- | --- | --- | --- |
| Recall unchanged assignment | 16/16 | 16/16 | 0/16 |
| Recall latest assignment | 16/16 | 16/16 | 0/16 |
| Latest assignment with reworded query | 0/16 | 0/16 | 0/16 |

On 24 held-out Wikipedia articles, the same final 128 targets scored 4.665 with a 128-input window and 4.420 when up to 1,024 inputs were supplied. The longer backbone scored 4.804 and 4.797. The natural-text improvement can coexist with failure on exact probes beyond index retention capacity.

On the frozen 512-passage English LAMBADA sample, the longer checkpoint predicted every subword of the final word on **29/512** passages. Of those successes, the exact successor index contained every target subword on **28**. The matched longer backbone got **0/512** unassisted and **39/512** with the calibrated lookup. A modest final-word completion rate does not qualify dialogue, instruction following, or general language comprehension. The samples.json file contains uncurated continuations for three fixed prompts.

## CPU inference and state cost

Single stream, float32, two PyTorch threads, one warm-up, median of three repeats per context. Prefill includes reading the entire prompt and producing its token distributions. Decode includes selection and 32 generated steps with actual cached state. Tokenizer loading, text encoding and checkpoint load are excluded. All arms use the CPU reference paths, including an actual KV cache for the Transformer. Heap sizes are Python object-size estimates. Tensor bytes are backing storage after 32 decode steps.

| Arm at 1,024 prompt tokens | Prefill ms | Cached ms/token | Tokens/s | Tensor + episodic KiB |
| --- | --- | --- | --- | --- |
| Longer COG | 375.0 | 5.19 | 192.8 | 76.7 |
| Longer backbone | 330.7 | 4.60 | 217.3 | 15.0 |
| Transformer | 479.8 | 11.40 | 87.7 | 31680.0 |
| Mamba-2, original reference storage | 577.2 | 5.94 | 168.2 | 31441.5 |
| Mamba-2, compacted recurrent state | 574.7 | 5.84 | 171.3 | 721.5 |

The Mamba-2 reference initially retained the backing storage of a larger prefill tensor through a view. Copying only its two live recurrent tensors per layer after prefill reduced its cache to **738,816 bytes** at all three lengths. All 32 greedy output IDs matched the reference at every measured length and repeat. This control is recorded in mamba_cache_audit.json.

The exact table is capped and may evict older successors. The Transformer KV cache grows with prompt length. The CPU reference materializes full prompt output distributions, so cache size is only one part of total memory use. Training throughput and peak process RSS are in metrics.json. Fits shared the host during training, and the baseline kernels differ. The first concurrent 30M Mamba-2 process was killed. Its original seed and complete budget were rerun successfully after freeing capacity. The interrupted attempt contributes no model or score. These measurements support no hardware-independent speed or economic claim.

## Qualification decision

**Build gate passed.** The released system was initialized, trained, evaluated, saved, loaded and sampled at the requested parameter scale. The preregistered Wikipedia advantage over the recurrent backbone passed the compact three-seed criterion and agreed in sign at 30M. The causal evidence points principally to observed-successor retrieval. The compact learned slot/link bank failed to beat the cheaper successor-only model. A large-scale general reasoning or chat model is still unqualified.

The long fit consumed 524,288 targets from 485,320 distinct training positions. Its data-to-parameter ratio is about 0.017 tokens per parameter. Only one large seed was run, with a narrow English training corpus. The CPU training run cannot resolve asymptotic capacity, scaling laws, multilingual transfer or performance against well-pretrained modern LLMs. The factorial follow-up and foreign-bank test reused the known evaluation panels. Independent reproduction on different training hardware and an independently sealed text evaluation remain open.

**Next decision.** Use the strong sequence baseline plus bounded lookup as the reference for a larger-token campaign. Train it against an otherwise matched model with learned, revision-sensitive memory, retaining the lexical and paraphrased queries as separate endpoints. Advance the extra memory if it improves reworded revision and broad natural-text loss over this reference at acceptable measured inference cost. The evidence supports extending the language model. The current slot/link design still needs a gain over the simpler memory control before a larger training investment is justified.

All 24 checkpoint digests were verified. Fresh reloads reproduced sample-window NLLs on both domains and checked streaming versus vectorized logits for each model. The maximum absolute per-window mean NLL discrepancy was 9.54e-07. The maximum cached log-probability discrepancy was 9.54e-06. This is same-host replay. Versioned receipts, data provenance and intervals are under evidence/. The raw per-window and per-passage records are in the downloadable research archive.
