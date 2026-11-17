# OUTPUT-MEMORY-021: completed three-seed comparison

17 November 2026. **The tested system failed its factual-output qualification. Neither cached autoregressive output nor genuine masked-diffusion output made the retained CSM a reliable persistent-memory language system.** All 12 planned head fits completed, with independently audited saved evidence and exact recorded checkpoint replay. No paid compute or protected TEST was used.

## What was tested

Three retained 33,699-parameter literary CSM checkpoints were frozen. Each supplied a 392-dimensional prefix-state representation to two separately trained 162,818-parameter output heads: cached autoregressive (AR) and absorbing-mask diffusion. A second pair of heads received an engineered representation of all resolved records, original record descriptors and the supplied query identity. This control did not receive the selected answer, but did receive privileged structured information. It was not learned CSM memory.

Each head received 4,096 updates of 128 answers, with matched sampled-example schedules and initialization across paired conditions. The first 2,048 complete TRAIN pairs were prospectively selected before outcomes: 4,096 sessions and 24,576 question features, repeatedly sampled rather than 524,288 independent examples per fit. The original 8,192-session archive is retained. TRAIN caches include earlier gold answer sentences; evaluation instead feeds back each head's own attempted output. The frozen CSM was not retrained or established as competent on these tasks.

Outputs were full but fixed-template sentences, such as `The current code is ABC.` or `No code is currently assigned.` Capacity was 64 byte positions including EOS/PAD, without gold answer length. This is a constructed factual-output study, not open-ended language qualification. Diffusion used a true masked-denoising objective and iterative absorbing-mask reverse process; primary inference used eight rounds, not corruption-only training. AR used its tested KV cache and stopped at generated EOS or malformed PAD.

All six exposed development axes were retained: familiar, renamed identities, alternate wording, held-out link composition, longer delay and combined changes. Each axis contained 256 sessions/128 paired groups and six questions per session. These are exposed, correlated constructed DEV panels, not protected final TEST or broad natural-language generalization. Zero/swapped state and diffusion four/sixteen-round diagnostics used the prespecified first 16 pairs of each axis. They are frozen interventions, not separately trained alternatives.

## Primary results

Counts pool the three seeds and six axes for descriptive accounting; they do not turn repeated panels into independent datasets. Every arm has 27,648 primary answers, 4,608 sessions and 2,304 paired first-answer checks.

| Condition | Exact sentence answers | Correct non-absence payload answers | Complete six-answer sessions | Both paired first answers correct |
|---|---:|---:|---:|---:|
| Frozen CSM + cached AR | 2,449 / 27,648 (8.86%) | 2 / 19,968 | 0 / 4,608 | 0 / 2,304 |
| Frozen CSM + diffusion, K=8 | 346 / 27,648 (1.25%) | 0 / 19,968 | 0 / 4,608 | 0 / 2,304 |
| Supplied resolved state + cached AR | 7,184 / 27,648 (25.98%) | 0 / 19,968 | 0 / 4,608 | 0 / 2,304 |
| Supplied resolved state + diffusion, K=8 | 6,831 / 27,648 (24.71%) | 1 / 19,968 | 0 / 4,608 | 0 / 2,304 |

**Almost all correct answers were absence sentences.** The approximately 25% aggregate accuracy of the supplied-state conditions is not successful factual copying. Across the four arms only three non-absence answers were correct. Literal-field and exact-sentence counts happened to agree in these primary records; literal presence is not generally a semantic correctness guarantee.

Saved pair-bootstrap intervals of [0,0] are degenerate resampling of all-zero observed outcomes, not certainty about unseen performance.

All 72 seed × arm × axis panels failed the prespecified 95% complete-session and paired-first-answer gates. No failed seed, operation or payload was removed, and no threshold or training recipe was changed after observing outcomes. The diagnosis is broader than a lack of distant CSM state: even heads given resolved records failed to turn available payload information into dependable output. This does not isolate whether representation, learned selection, byte reconstruction, objective, optimization or exposure distribution is the principal cause.

## Cost, without a misleading speed victory

Whole-primary-batch execution totals across the same 4,608 sessions per arm were:

| Condition | Aggregate primary execution time |
|---|---:|
| Frozen CSM + cached AR | 1,561.545 s |
| Frozen CSM + diffusion, K=8 | 1,261.397 s |
| Supplied resolved state + cached AR | 438.630 s |
| Supplied resolved state + diffusion, K=8 | 143.366 s |

These are batched workload totals, not per-user latency. They include prefix ingestion, structured-state work, generation/sampling and generated-answer feedback where applicable; model loading and output-file serialization are excluded from this specific subtotal. Full stage costs below include broader orchestration. Different generated errors and termination behavior also affect work. All conditions failed acceptable quality, so these timings **do not establish an inference-efficiency win or satisfy the 12:1 quality-constrained cost rule**. Batch-one trained-model latency was not measured by this scientific run.

- Three scientific stages: 10,516.877 s in total.
- Native outer execution: 10,614.421 s.
- Included TRAIN cache construction: 1,916.580 s.
- Included head training: 525.715 s.
- Successful scientific attempts consumed 10,796.503 seconds of provider-reported free quota, including provider overhead; this excludes earlier software calibration and retained pre-fit failures.
- Final provider snapshot: 12,377.240 / 21,600 seconds used, 9,222.760 seconds remaining, no reserved time, paid scaling disabled. The recovery reserve was preserved.

## Evidence and limitations

The full roster is 12 fits, 49,152 optimizer updates and 6,291,456 sampled answer exposures. There are 18,432 primary sessions, the same number of replay sessions and 6,912 diagnostic sessions. Primary capability accounting uses 110,592 answers; replay is repetition evidence, not additional independent capability evidence. All three archives retain checkpoints, optimizer/RNG state, sampled-ID traces, raw output tokens, costs and native receipts.

Independent review verified archive CRC/content hashes, exact source/runtime/review/admission bindings, all checkpoint/update/exposure counts, paired schedules, raw primary answer rescoring and identical recorded primary/replay rows for all 72 panels. The producer reran the exact saved checkpoints on the same host. Reviewers independently inspected those retained artifacts and recomputed metrics; they did not retrain or independently execute GPU inference. Model implementation correctness is supported by bounded tests and review, not an end-to-end mathematical proof of learned behavior.

Two pre-fit failures remain in the history: unsupported deterministic rank-3 CUDA cross-entropy was replaced by an equivalent flattened formulation without disabling determinism; an oversized genuine review text was replaced by an independently authored concise declaration without weakening its schema. A global pair-ID validation gap was also corrected before scientific execution; actual frozen data were valid. Old admissions remain historical, not reusable approval for changed code.

The protocol mentioned a deterministic interpreter/copy reference. The executed scientific comparison contains the four learned-head conditions above; it does **not** contain a separately timed deterministic copy-output arm. Do not claim that such an output reference was measured or that it has a demonstrated runtime advantage in this study.

## Decision and next useful target

Do not scale or promote this frozen-CSM-plus-output-head recipe. Do not select diffusion simply because it used fewer sequential output steps, or interpret the supplied-state scores as successful memory use. This is a completed negative study of these implementations, conditioning representations and training budget, not a proof that diffusion, AR generation, or CSM in general cannot work.

The next justified engineering target is exact factual readout: a shared representation that preserves arbitrary payload bytes, a learned choice of the correct record, and an explicit, inspectable output-copy path. Keep exact copying structurally separate from optional free text or diffusion refinement, so a fluent decoder cannot silently alter authoritative facts. Any such repair should be a new prospectively reviewed comparison against an equally structured simpler controller/store. It is a proposal, not an implemented or validated repair, and no further fits were launched by this study.

Public source and recovery instructions are in draft PR #68 of `FPC-effortless/cdl-attention-experiment`; the three verified private seed archives are account-scoped in ChatGPT Library. The final combined package and research Space index are being reconciled for delivery. GitHub source alone does not grant another account access to private datasets/checkpoints. Use the latest private recovery guide and exact hashes; do not rerun completed seeds merely because their private archives are absent from a public checkout.
