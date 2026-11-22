# COG-COMM-001: memory-free communication preflight

COG-CORE is the proposed architecture; nanoCOG is its implementation line. This is the first integrated CPU reference and prospective experiment, not a trained model or speed claim. No cross-document memory, exact executor, learned entity IDs or protected evaluation is in the candidate.

- **A:** token-conditioned depthwise causal taps plus common decoder shell.
- **B:** A plus strictly exclusive completed-block prefix summaries.
- **C:** B plus per-token bounded packet routing over prior positions. This is an independent scratch path, not routing over the summary packets.
- **C-blind:** same packet writer, scratch and four-candidate limit as C, but the subscriber chooses topics from position rather than content. Its subscriber parameters are inactive even though they are registered. A trained C vs B comparison changes mechanisms and capacity; C vs C-blind provides a narrower comparison.

Shared parameter tensors begin exactly equal; do not assert equal capacity or work between A, B and C. Local, summary and packet residuals are added in the existing nanoCOG decoder; no typed board is attached. The four-topic, two-slots-per-topic writer retains at most eight payloads per layer and stream. The reader admits up to two topics of two candidate positions each. Per-token writes can evict earlier evidence long before the age limit. Content-aware routing remains an attention-like operation, not dense all-pairs attention.

## First finding: age coverage precedes fit

[Fixture geometry](FIXTURE_GEOMETRY.json) uses 40 inert TRAIN-form families, not the experiment panel. With 16 eight-token blocks, direct literal support was age-eligible in only 16/80 worlds, and in none at delays 4, 8 or 16. A 64-block age envelope removed that hard exclusion for this fixture. It did not increase the eight-slot storage capacity or establish actual candidate recall. Delay denotes distractor statements, not tokenizer tokens. A literal support position miss is also not proof there was no information: later hidden states can carry earlier context.

[Fixture work](FIXTURE_WORK.json) records same-host inert forward/backward, not serving cost. At 448 positions, the learned parameter-matched-with-C-blind router took about 0.81s median for eight worlds versus 0.61s for C-blind and 0.035s for summary-only. This is motivation to audit publication, retention and physical work before optimizing or scaling; it is not a system-level benchmark.

## Train, predict and verify boundaries

The [prospective protocol](PROTOCOL.md) governs the upcoming test. Same grammar, held alias strings only; no broad compositional or natural-language claim. Capitalized terms are represented by byte pieces consistently, and the tokenizer is trained from the TRAIN projection. Actor input is raw prompt text. Pair IDs are used only to join predictions, and verifier-side locations and labels are not model inputs. Use the same next-token, outcome CE and paired foil loss in every arm. Rehearsal on authentic prose and full language-model comparisons are separate stages, not part of this fixture.

Run software checks from this directory with the parent nanoCOG project on PYTHONPATH, using the pinned CPU Torch runtime. For example, in the research repository checkout:
```sh
cd research/nanocog-001/comm-001
PYTHONPATH=.. OMP_NUM_THREADS=1 python -m pytest -q cog_comm
PYTHONPATH=.. OMP_NUM_THREADS=1 python -m cog_comm.geometry --fixture
```

Both compact test receipts are preflight only. An optimizer fit needs independently reviewed actual group/role custody, enough supported fit and calibration exposure, an easy learnable control, real-shape work limits and exact TrainLab admission. Neither software parity nor the age calculation can authorize a model fit.
