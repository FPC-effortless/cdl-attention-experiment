# COG-BALANCE-001 · Adaptive Hybrid State Computation

The primary comparison is whether adding conditional updates, competition, reliability weighting and geometric operations to the Native backbone improves held paired retrieval at the same data budget while preserving a useful language quality and latency trade-off. Source base is TRACE-007 commit 13b0b1cac115a5cf4728cf3bb38225a84d50f562. The earlier NumPy event-selection probe is a separate synthetic diagnostic.

| Arm | Mechanism |
| --- | --- |
| A | Native HH |
| B | Full Unified BB, both slow and geometric side channels |
| C | Native plus multiscale slow state |
| D | C plus contextual key/query/value and a plastic delta-rule bank |
| E | D with independently learned binary slow and write permissions |
| F | E with top-one slow, memory or null residual and a selection penalty |
| G | F with a learned sigmoid evidence-weight multiplier on writes |
| H | G with a fourth router destination and four learned reflection axes |
| GQA, GRU | Grouped-query Transformer and two GRU sequence mixers |
| GATED_DELTA, MAMBA2 | Independent CPU PyTorch reference recurrences in the same language-model shell |

All architectures share a byte encoder, local front end, 288-output vocabulary, tied decoder and approximately matched parameter count. The adaptive arms retain Native HH's protected memory. Adaptive context has a causal exponential history. Single-token evaluation dispatches only selected branches in F–H. Training and multi-token prefill compute dense surrogates. F–H use straight-through gates and pay 0.01 for predicted optional activation, averaged over text and question batches. This is an activity proxy. The cost comparisons use measured whole-model timing. The shared routing context is present on every token. G's rate and reliability enter as a product, which cannot identify calibrated reliability from prediction loss alone.

## Data and budgets

WikiText byte windows stay within documents in the pinned TRACE-007 corpus, with a held development subset drawn from the original training partition. The same fixed 32,768 sampled byte targets are evaluated in the GitHub screen. No official test split is opened. Each paired seed fixes the training windows and question examples. Dev sampling uses independent seeds. Synthetic questions alternate structured byte and generated English templates. Neither panel is a benchmark on naturally occurring questions. Number of correctly answered original/revised pairs, out of 128 worlds per panel, is the primary evidence-use measure. For irrelevant English revisions the correct answer stays the same. A held cancellation tests a new operator pairing whose net action is the identity. Randomized-value pairs have different chance properties from single questions. Repeated use of the same dev worlds across seeds precludes counting them as independent worlds.

The engineering pilot runs 96 updates on 4 language windows of 128 bytes plus 4 questions per update, a separate 8-process-CPU-second checkpoint, 16 dev batches and 64 dev worlds per panel. It checks that learning and measurements work across all arms. The main screen uses 384 updates, 8 language windows of 128 bytes and 8 questions per update. Report the first completed optimizer step above 60 training process-CPU-seconds separately as a hardware-specific compute match. Record actual times and digests. Inputs, unweighted language loss, 0.5-weighted answer loss, AdamW learning rate 0.001, clip 1 and three training seeds (9731, 9743, 9769) are shared. Neither pilot nor main screen is a convergence benchmark. The two stages have distinct budgets and measurements.

Per arm and seed the GitHub runner trains the two-layer 1.08M model, then the 22-layer 10M model. The depth-scaling recipe holds the embedding dimension at 128, shares the encoder and decoder, and initializes each added pair independently. It tests this particular scaling strategy. Both scales get identical examples per seed at the fixed-step readout. Comparing cross-runner CPU seconds is approximate.

## Interpretation

Compare each adjacent incremental mechanism and both parent backbones on all three matched seeds. Report individual seed signs, paired accuracy and held English paraphrases, language bits per byte, whole-model decode latency, state bytes and actual measured hardware counters where accessible. For screening, favor models on the non-dominated quality/cost frontier. A stronger claim requires replication on independently held real questions, cost-matched controls and additional training. A zero on changed pairs alongside high unchanged-pair accuracy is consistent with an answer prior. Count null routes and geometric executions before attributing performance to selection. CPU reference timing is specific to these implementations.

Use python check_recurrence.py ARM --scale 1m, then python run_balance.py --arm ARM --seed 9731 --scale 1m --corpus CORPUS_DIR --out OUTPUT_DIR. Corpus artifact trace-007-corpus is in run 46171359964. Runtime is PyTorch 2.10 CPU, Python 3.12 and NumPy 2.3.5. Per-fit artifacts include weights, measured timing and data checksums. Decode energy and DRAM-traffic fields stay null when the host exposes no usable counter.
