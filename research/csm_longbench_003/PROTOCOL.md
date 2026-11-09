# CSM-LONGBENCH-003 prospective protocol

Parent d63ca288a29998c86301fe213852929fdfe9bb21. Real archived Tiny Shakespeare,
SHA256 86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed.
Train [0,1003854), exposed DEV [1003854,1059624), protected test never scored.

Question: Does persistent regulation help after a larger training budget, and does
a longer gradient window improve multiple-timescale regulation?
Arms instant, single, multi, multi_tb64. All 34,479 learned parameters. Seeds
11,23,37. Same initial learned arrays per seed. AdamW lr=.001, WD=.01,
gradient clip1, batch4, 256 updates ×256 tokens =262,144 targets/model (32× pilot).
No early stop, tuning, or best-checkpoint selection. Final checkpoint primary.
Four independent contiguous 2049-byte excerpts per episode, 32 episodes; seeded
starts sampled from TRAIN only. Carry state through each 2048-token episode,
reset between excerpts; detach across updates. Gradient window256 except tb64:
four 64-token backward calls with loss /4, one optimizer update per256 tokens.
State carry, optimizer steps and token budget remain equal in that comparison.

DEV:16 nonoverlapping 2049-byte excerpts, evenly spaced over DEV. Score all2048
next-byte targets each:32768 targets, with age bins0–31,32–255,256–1023,1024–2047.
Report per-seed/per-excerpt NLL, BPC, perplexity, accuracy, ten-bin ECE, confidence.
Learning curves at updates0,32,64,128,256 on first512 targets in all16 excerpts;
training last-loss, preclip gradient norms, per-trace-group driver/injection norms.
Final train diagnostic on first episode with the same evaluator.
Final interventions: clear trace every token; one-time trace swap across streams
at token256; clear records at256. Report full and post256 NLL plus paired deltas.
These are mechanistic interventions, not alternative training models.
Route diagnostics: destination/operation frequencies, write/link admission rates,
effective write gain, regulator RMS by timescale group. Measure cold and warm age
bins. Single-example CPU step timing separately after all training processes exit:
100 warmup tokens, five256-token runs, median/p95 latency. State bytes explicitly.
CPU process RSS is process memory, not tensor-only allocation. No GPU/paid compute.
Train-only smoothed unigram/bigram baselines fitted on identical sampled tokens
per seed; descriptive controls, not parameter/compute-matched architectures.

All final model/optimizer/RNG/state/schedule checkpoints retained. Independent
cold-process replay checks checkpoint SHA256 and DEV NLL with absolute tolerance
1e-6; report observed differences and fail above tolerance. This tolerance is
registered before new fits. Historical exact-local replay claim remains factual;
previous CI exact comparison failed without logging numeric delta. Preserve that
failure and diagnose it rather than calling historical cross-host replay passed.
Report every seed, mean/range and paired differences; n=3 is not strong statistical
power. Corpus is familiar/exposed DEV, so no generalization or superiority claim.
