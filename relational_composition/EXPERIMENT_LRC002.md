# LRC-002: conditional answer information and oracle decoding

Prospective diagnostic protocol, 2026-11-07. Freeze in git before fitting. This
extends the engineering prototype without altering LRC-001's frozen result.
It is not a confirmatory test of a new sequence-model family.

## Motivation from the supplied compression transcript

The useful idea is to distinguish predictable surface redundancy from information
that becomes predictable only given a relevant stored fact. Falling whole-text
loss can mostly reflect the former. Measure answer cross-entropy in **bits per
complete answer**, separately from text NLL and actual exact generation.

The transcript simplifies several points: uniform independent bytes are a
maximum-entropy source, but randomness need not be uniform; finite random samples
do not have exactly equal symbol counts. The counting argument rules out a
lossless compressor shrinking every input; it does not force every particular
random-looking input to grow. Archive headers can add overhead, and recompression
can occasionally exploit remaining redundancy. ZIP/DEFLATE is not just Huffman
coding. These qualifications do not change the proposed diagnostic.

Primary references:
- [Shannon, A Mathematical Theory of Communication (1948)](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x)
- [RFC 1951, DEFLATE](https://www.rfc-editor.org/rfc/rfc1951.html), introduction and coding description.

Use `bits = sum(answer negative log probabilities) / ln(2)`. The difference
between memory-corrupted and intact prediction loss is a model-dependent paired
diagnostic, **not measured mutual information**, net compression or a free-memory
benefit. No compressor or extra recurrent compression module is being added.
Model weights, routing/index costs and explicit memory storage remain real costs.

## Questions and fixed arms

| Arm | Memory access | Optimization targets | Interpretation |
| --- | --- | --- | --- |
| learned_lm | Original learned hard index | Every next byte | End-to-end routing diagnostic |
| local_lm | Disabled memory, trained from scratch | Every next byte | Trained shortcut baseline |
| oracle_lm | Supplied two supporting fact endpoints | Every next byte | Decoder capacity under original objective |
| oracle_answer | Same privileged endpoints | Two answer bytes only | Decoder ceiling with stronger target weighting |
| local_answer | Disabled memory, trained from scratch | Two answer bytes only | Matched privileged-loss baseline |
| oracle_random_answer | Original endpoints, independent uniform random answer | Two answer bytes only | Information/leakage negative control |

**Oracle routes already resolve the relation chain.** They bypass discovery,
candidate-cap misses and variable binding. A success proves only that the neural
operator can decode an answer from the supplied endpoints. It cannot establish
multi-hop reasoning or coupling. Oracle inputs and answer masks are explicitly
privileged and never count as raw-text learned-routing evidence.

The random-label control draws `n` followed by one of 20 equally likely suffixes,
independently of the visible world and original oracle endpoints. Its conditional
entropy is log2(20) = 4.321928 bits/answer in the specified data distribution;
Bayes exact-answer accuracy is 5%. A finite held-out estimate can fall below the
population entropy by chance. The original structured generator does not inherit
this conditional-entropy floor: its correct answer is determined by the world.

## Fixed execution

- CPU only, one thread, deterministic PyTorch; no paid compute.
- All six arms, initialization seeds 201, 202, 203. Identical starting tensors,
  world order and optimizer within each seed; no seed selection.
- Model config: width 24, two layers, kernel 5, four relations, eight addresses,
  two reads with cap two, coupled backbone. All allocate 33,061 parameters;
  effective active paths differ in the local/oracle controls.
- 400 AdamW updates, batch four, learning rate .002, gradient clipping 1.
- Inherited one-coordinate counterfactual rule, weight 1, clipped advantage 1;
  all arms execute base and alternative forward passes. Do not change it here.
- TRAIN worlds `20_000_000 + seed*100_000 + step*4 + row`.
- All 18 fits must finish before any held-out evaluation. No hyperparameter
  search, early stopping or post-result additional fits within this protocol.
- Held-out worlds `90_000_000..90_000_063`, paired across seeds and conditions:
  IID, held-out relation order, unseen names. Random labels use IID only; their
  uniform-20 alphabet would conflict with the unseen-name population.
- Labels use an independent deterministic PRNG keyed by `world_seed XOR
  0x5A71F903`; neither seed nor random label is available in model input before
  prediction. Existing v1 DEV/TEST worlds are not reused.
- Record whole-text NLL, teacher-forced answer bits, two-byte greedy generation,
  per-example losses, memory-corruption deltas, all training receipts,
  model/optimizer/RNG checkpoints, source/protocol/checkpoint hashes and costs.
- Greedy generation feeds back predicted bytes; gold answers are not fed back.
- All index work is still counted even when an oracle bypasses its result.
  Oracle feature reads are counted separately; no inference speed claim follows.

## Prespecified decisions

1. Oracle decoder competence: `oracle_answer` must achieve >=90% IID greedy
   exact answers in **every** seed. Otherwise operator/representation/optimization
   remains a live bottleneck; do not blame routing alone.
2. Random-label leakage alarm: any seed above 20% IID greedy accuracy triggers
   investigation before any positive interpretation. No formal hypothesis test or
   calibrated p-value is claimed for this conservative engineering alarm.
3. Report all six arms and both secondary shifts even when the competence gate
   fails. The answer-masked arms are diagnostics, not language-model results.
4. Differences between oracle_lm and oracle_answer diagnose an objective/training
   limitation only within this setup. Differences between oracle and learned
   routing are not a controlled causal proof that routing is the sole problem.
5. No architecture promotion or speed claim is permitted by LRC-002. The coupling
   versus independent comparison is deferred until learned routing is competent.

Inspection, design, implementation and audit by the same assistant are not an
independent scientific replication. Future changes require a new protocol and
fresh confirmatory populations. The next action follows the observed bottleneck,
not the desire to declare the architecture successful.
