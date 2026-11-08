# LRC-TS-001: all implemented LRC language variants on Tiny Shakespeare

Frozen benchmark design, 8 November 2026. This is a new training cohort, not a
relabeling of the earlier Sherlock Holmes experiments. The question is which
implemented LRC architecture or applicable prose-training recipe predicts held-out
Shakespeare bytes most accurately, at what measured CPU cost, and whether its
memory contributes to its predictions.

## Coverage and size

Fourteen backbone fits per seed: V1 coupled, V1 independent, V1 no-reference,
V3 dependent multi-hop, V4 with eight page bits, V4 with sixteen page bits,
local LRC, local LRC plus lexical cache, V4 lexical-address supervision,
V4 lexical-address plus page/slot supervision, shuffled combined supervision,
GRU, GRU plus the same lexical cache, and the inherited two-layer Transformer.
See VERSION_INVENTORY.md for the diagnostics that do not define another raw-text
language model. LRC and Transformer use two layers; the inherited GRU uses one
recurrent layer. All have approximately 43K allocated parameters
(within five percent). Widths: V1 30, V3 26, V4-8 24, V4-16 22. Local kernel 3.
The inherited references are GRU width 68/embedding24 and Transformer width36.
These are size-normalized architecture configurations, not replays of historical
training runs. Local LRC retains inactive memory-policy parameters; allocated size
matching does not imply active-parameter, compute, or memory matching.

Three additional final-layer utility heads per seed (true utility, shuffled utility,
bootstrap-random utility) use the SAME newly fitted V4-8 backbone. First-layer reads
remain enabled. Only the final 50-parameter gate is optimized, from causal features
and train-only paired read-on/read-off losses. The backbone stays frozen. Static
last-layer-off and all-memory-off evaluations are controls, not new fits.

## Dataset and supervision

Canonical karpathy/char-rnn data/tinyshakespeare/input.txt, 1,115,394 bytes,
SHA-256 86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed.
Contiguous train [0,floor(.90*N)), validation [floor(.90*N),floor(.95*N)),
test [floor(.95*N),N). Splits share neither targets nor carried state. Each raw byte
maps to byte+1; token0 is padding; 257 output classes. No fitted tokenizer.
All supervision, including lexical teachers and utility teachers, uses TRAIN only.
No synthetic entity labels, oracle endpoints, test labels, or query-time teachers.
The inherited lexical teacher is a repeated-byte proxy, not semantic supervision.

## Fixed training

Seeds 2201,2202,2203; fresh weights. Exactly 2,048 AdamW updates per backbone;
batch4, context128, 512 target bytes/update, 1,048,576 target-byte exposures/fit.
AdamW lr0.002, betas(0.9,0.999), eps1e-8, weight_decay0.01; gradient clipping1.
Constant learning rate and final checkpoint only: no hyperparameter search,
early stopping, or selection by validation/test loss. All models receive the same
sampled windows in the same order within a seed. Every window starts with empty
state. No shuffled train/test split or text contamination across boundaries.

V1/V3 retain original one-coordinate counterfactual hard-policy credit.
V4/local/cache retain hierarchical credit; lexical variants retain their original
auxiliary recipe and coefficient0.1. The hard policies are not silently frozen by
substituting plain cross-entropy. These methods normally require two forward passes
per update. GRU/Transformer controls require one. Token exposure is matched;
optimizer work, floating-point work, and wall time are NOT matched.

Utility post-training: 600 additional updates, batch4/context128, lr0.002, same
AdamW/clipping. Targets are clamp(NLL_off-NLL_on,-1,1); head regression MSE.
Train three heads from the same teacher features; one shuffled and one resampled
target control. Gate initialization and teacher algorithm match the prior stage.
Report 307,200 additional target positions and1,200 teacher forwards per seed,
plus head-update time, separately. Never count this as an equal-cost backbone fit.

## Evaluation and integrity

Close all backbone and gate fits and hash every checkpoint before TEST evaluation.
Primary: full-test next-byte NLL, bits/byte, perplexity, and next-byte accuracy,
reset at128-byte windows. Score every one of the55,769 valid within-test next-byte
targets exactly once, including the final short block. Secondary context512 scores
the same targets with different resets, to measure distribution shift beyond training.
Report all seeds and mean/sample standard deviation. Descriptive rankings, not
claims of statistical superiority from three seeds. No failed seed exclusions.

Memory intervention: paired memory-disabled evaluation for every full LRC family,
cache-disabled and causal cache-corruption controls, same test targets. Positive
BPB(off)-BPB(normal) means helpful memory. Local trained baselines remain separate
from disabling memory on a full trained checkpoint. Utility includes final-read-off.

Before fitting, test future-suffix noninterference, no state across examples,
bulk/incremental and prefill parity, correct split/target accounting, lexical teacher
causality, finite training, and explicit routing-credit execution. Repeat trained
parity on all checkpoints. Preserve all source versions, protocol/data hashes,
per-update receipts, checkpoint/RNG/optimizer states, per-seed results and anomalies.
Source/protocol are committed locally before the first real fit. Publication timing
must be labeled honestly if remote publication occurs after fitting.

## Efficiency

CPU float32, one PyTorch compute thread per process; record actual software/CPU
and environment. Parallel workers reduce completion time; their elapsed times are
not isolated speed benchmarks. After workers exit, run isolated training-step timing
on disposable copies, and warm up then interleave all inference arms for five
repeats. Workload:512 observed-byte prefill plus128 next-byte prediction steps,
batch1, including normalization, state updates, retrieval and cache overhead.
Timing uses fixed observed validation bytes, not model-dependent generated text.
Report prefill, decode, end-to-end latency/throughput, allocated parameters,
checkpoint bytes, retained stream state and measured peak process RSS where available.
Any profiler FLOP totals are explicitly partial counted operator FLOPs, not complete
algorithmic complexity; Python indexing, lookup, memory traffic and unsupported
operators are outside that count. No GPU or optimized production-speed claim.

## Interpretation limits

This small training benchmark uses approximately one corpus-equivalent of sampled
target exposures (sampling with replacement, not a complete traversal). It measures
Shakespeare next-byte prediction and implementation efficiency. It does not establish
semantic relational reasoning, general language competence, optimal hyperparameters,
frontier scaling, or performance at 1M/125M parameters. A cache win is not evidence
that learned semantic indexing works. Memory benefit is measured separately.
