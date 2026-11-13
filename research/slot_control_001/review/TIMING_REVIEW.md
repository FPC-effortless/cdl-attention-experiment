# Independent timing and report review

Fresh review session `/root/review_changing_state`, 2026-11-13.

**PASS for the descriptive fresh-query CPU latency comparison.** No timing rerun
was performed. This audit validates saved measurements and their aggregation; it
does not remove shared-host timing noise or establish a production cost frontier.

## Record and source checks

All **864 durations / 432 model pairs** are present, unique and correctly keyed
by seed, block, regime, history depth and repetition. Every duration is a positive
finite integer number of nanoseconds. Declared alternating order matches both the
recipe and serialized row order. Every query matches its predeclared block-based
selection, and every timed answer matches an independent current-value calculation.
The timing file binds the reviewed source freeze and exact timing source bytes.
The authored `timing_summary.json` matches the independent computation.

Source inspection confirms the measured region includes event tokenization,
tensor construction, fresh state, every token step and answer argmax. Gold answer
and control-target calculation, checkpoint loading, imports and reference forward
checks are outside the timer. CSM uses the uninstrumented original class. One
warmup per model/cell and a shared token-step loop are present. This is a fresh
one-query replay of an already structured event stream; it does not time learned
natural-language extraction or sustained multiquery reuse.

## Independently recomputed numbers

| Additional SET updates | CSM pooled median ms | Slot pooled median ms | Median paired CSM/slot | Seed/regime median-ratio range |
|---|---:|---:|---:|---:|
| 1 | 10.978287 | 4.042158 | 2.564761 | 2.182470–2.721514 |
| 16 | 35.638620 | 13.643033 | 2.577914 | 2.541605–2.638737 |
| 256 | 419.925981 | 166.544371 | 2.530941 | 2.373292–2.665882 |

Each depth has 144 paired measurements. Pooled medians separately aggregate each
model's durations. The ratio is computed per matched pair first, then its median
is taken; it is **not** the quotient of pooled medians. Each seed/regime median
uses 24 pairs; the displayed range spans six such cells per depth. These cells
share base streams and are not independent population replications.

There are **two** reversals where the slot duration exceeds CSM, both at depth16:

- Seed101, block4, all_changed, repetition2: CSM50.148141ms versus
  slot116.663671ms; paired CSM/slot0.429852.
- Seed303, block3, all_changed, repetition0: CSM45.327349ms versus
  slot57.826985ms; paired CSM/slot0.783844.

No measurements were removed. The raw reversals and wide individual-pair range
are consistent with the need to retain the single-host, descriptive limitation;
their causes were not diagnosed. No latency confidence interval is claimed.

Logical retained tensor counts,1,568bytes CSM and1,280bytes slot, agree with the
model state definitions. They exclude weights, Python objects and temporaries and
must not be described as measured peak memory or process RSS.

## Report review

The report's reliability table, historical raw-CSM/Transformer means, timing
table and current two-reversal statement match the audited evidence. The host
inventory supports Xeon Platinum8573C. The report distinguishes shared supervision
from answer-only historical controls, public development data from confirmation,
sampled neural replay from complete metric reconstruction, and interpreter
semantics from learned capability.

Use **lower median session latency** for the measured advantage. Broader financial,
energy or production inference-cost savings are unmeasured. The report properly
leaves the well-trained-alternatives question open and does not claim a CSM win.

Machine-readable exact values, definitions, source/freeze/timing hashes, all18
seed/regime cells and both reversals are in `timing_checks.json`. The read-only
checker is `audit_timing.py`.
