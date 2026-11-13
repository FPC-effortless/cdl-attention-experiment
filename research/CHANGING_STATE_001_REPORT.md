# Changing-state research: CSM robustness survives; a simpler control matches it

Woyengi Intelligence Labs · 13 November 2026 (Africa/Lagos)

**CSM's supervised SET/recall result survived much longer overwrite histories,
but an equally supervised, simpler learned slot store matched it perfectly.**
The new result supports reliable learned keyed storage in this bounded grammar.
It does not establish a need for CSM's operators, pointers or content-dependent
controller on this task. Extra machinery now needs to earn its place on a task
that requires more than local parsing and keyed replacement.

## What actually ran

The exact public CSM-ADDRESS-002 archive was recovered and hash-verified from
`FPC-effortless/cdl-attention-experiment` parent
`3edc4f1a541a3221390e4c0a21fe0c12004f1ca5`. Original model/checkpoint bytes remained
unchanged. The archive SHA-256 is
`e4a3d3e39e6e8aa3e6f4f3c6c76b705dacf310be4adb8fc82f068712e678b9f2`.

1. CHANGING-STATE-001 froze a new public development diagnostic: four identities,
   eight values, four distinct initial values, followed by 1/4/16/64/256 genuine
   value-changing overwrites. Either all identities cycle through changes, or
   one repeatedly changes while the other three must remain intact.
2. SLOT-CONTROL-001 trained three new conventional learned slot-store controls:
   18,254 parameters versus CSM's 18,294. The control retains learned admission,
   destination, gain, value proposal and readout, while removing CSM's operator,
   link and content-dependent routing machinery. This is a simplification
   control, not a claim to reproduce an optimized published MemNet.
3. Both received identical public TRAIN examples, exactly the same sampled
   schedules, the same destination/admission/gain labels and loss weights,
   256 AdamW updates per seed and 8,192 answer exposures per fit. Seeds were
   101/202/303. Initial weights differ by architecture. No hyperparameter tuning,
   early stopping, extra exposure or failed-seed removal occurred.
4. All three new fits closed before either development evaluation. Total new
   training: 768 updates and 24,576 answer exposures. No GPU or paid compute.
   PyTorch 2.5.1+cpu, NumPy 2.3.5 and Python 3.12.14 match the historical versions.
5. The original supervised/raw/supplied-controller CSM checkpoints and historical
   Transformer were evaluated without training. The Transformer was deliberately
   scored only at <=128 tokens; its implementation supports longer sliding-window
   execution, so unscored cases are not architectural failures.

Each panel contains 64 worlds and all four identity queries, independently replayed
from a fresh state. The longest input is 1,048 tokens. There are 640 panel-worlds,
but only **eight independent base patterns**, each shifted through eight correlated
value assignments. Depths/regimes are paired; depth-one regimes duplicate histories.
No confidence claim treats the thousands of queries as independent observations.
Complete canonical histories are excluded from the original TRAIN bank. These are
public development cases, not sealed confirmation data or protected TEST.

## Reliability result

| Measure, every depth/regime/seed | Supervised CSM | Equally supervised slot store |
|---|---:|---:|
| Correct current-value answers | 100% | 100% |
| All four queries correct for a world | 100% | 100% |
| Untouched-identity answers, where applicable | 100% | 100% |
| Immediate stale-value errors | 0 | 0 |
| Parameters | 18,294 | 18,254 |
| Learned inference decisions | Yes | Yes |
| Destination/admission/gain labels at inference | None | None |

Each learned arm supplied 7,680 predictions across three seeds and ten panels.
The supplied-controller diagnostic also remained perfect, but uses privileged
routing at inference. The raw CSM averaged 12.70% answer accuracy over its 30
seed/panel cells (range 12.50–14.84%). The historical Transformer averaged 25.11%
over its 18 shorter seed/panel cells (range 12.50–38.67%). Those answer-only controls
received less training information; neither supports an architecture ranking
against the two process-supervised models.

The input grammar makes the destination and write timing identifiable from a
three-token local window. A learned local parser plus keyed replacement is
sufficient here. Extending overwrite count adds persistence/interference pressure,
but does not add a requirement for multi-record reasoning.

## Lifecycle benchmark added, with a strict capability boundary

An executable semantic contract now distinguishes:

- SET: replace a record's value.
- DELETE: remove a record.
- COPY: resolve the source now and store an independent snapshot.
- LINK: retain a live reference whose answer can change with its target.
- QUERY: follow live references; missing targets and cycles yield UNKNOWN.
- NOP: preserve state.

A dangling live link can resolve after target recreation. A copied UNKNOWN remains
absent. Self-copy resolves before mutation. Querying does not change semantic
state. These explicit choices prevent ambiguous gold labels in future experiments.

A map interpreter was checked against an independently implemented reverse-history
interpreter. Independent review checked 50,625 exhaustive small traces and 607,500
prefix/query comparisons in normal and optimized Python. **These are semantic
engine checks. Neither neural model was trained or scored on deletion, COPY,
LINK, UNKNOWN, or composed answers in this increment.**

## Independent review and evidence integrity

A fresh reviewer session checked the protocols and implementations before scoring.
For the new slot control it identified a closure-validation gap. The first draft
and freeze are retained; the revised scorer requires exact seeds, budgets, frozen
source, checkpoint/log hashes, complete step sequences and historical schedule
agreement. No fit or model score occurred before this repair and renewed review.

Verification includes:

- 10 author lifecycle/stress tests; five author slot-model tests.
- Six independent model/recipe checks, including exact CSM instrumented versus
  original logits on synthetic input and schedule/objective equivalence.
- Valid closure accepted and ten adversarial closures rejected in normal and
  optimized Python.
- Independent reconstruction of every answer/current-state label and all 35,328
  saved predictions into 138 scored metric cells plus 30 deterministic-control cells.
- Native frozen replay of 1,728 sampled predictions across all five arms and three
  seeds; every prediction matched, and replay weights stayed unchanged.

The metrics audit is complete; native replay is sampled. Aggregate route counters
were not independently replayed in full. Review occurred in a separate agent
session on the same machine, not external replication or independent retraining.
No new Lean compilation or learned-routing theorem is claimed.

## Exploratory complete-session CPU timing

Both learned models passed the predeclared reliability screen before timing.
The same loop measured event-to-token encoding, tensor construction, fresh state,
every token step and answer argmax. The original uninstrumented CSM avoids hook
overhead. Checkpoint loading/imports are outside the timer; no fit or reviewer
model replay ran concurrently. One CPU thread, batch one, FP32, Xeon Platinum 8573C.

| Additional overwrites | CSM median session | Slot median session | Median paired CSM / slot |
|---|---:|---:|---:|
| 1 | 10.98 ms | 4.04 ms | 2.56× |
| 16 | 35.64 ms | 13.64 ms | 2.58× |
| 256 | 419.93 ms | 166.54 ms | 2.53× |

There are 864 measured durations: three seeds, two regimes, three history depths,
eight base-block representatives, three repetitions and two models. Every timed
answer was correct. One warmup per model/cell preceded measurement; model order
alternated. The ratio column is the median of individually paired duration ratios,
not the quotient of the two pooled medians. Timing noise is visible: individual
pairs do not uniformly favor the slot store, including two reversals among 144 pairs at depth 16.

The slot implementation therefore has lower median session latency in this
measured fresh-query workload at equal observed quality. This is descriptive single-host timing, not
a production benchmark or a comparison against tuned Transformer/MemNet/SSM
systems. It does not measure sustained multiquery state reuse, restore costs,
energy, GPU kernels or confidence intervals. Each session ends in a 19-class
answer selection; it is not free-form language generation.

Logical persistent FP32 tensors total 1,568 bytes for CSM and 1,280 bytes for the
slot store. These counts exclude weights, Python metadata, inputs and peak
intermediates; no peak-memory or system-memory advantage is asserted.

## Research decision

Keep CSM-ADDRESS-002 as a working reference, and keep this simpler learned slot
store as a mandatory equally supervised control. The result shifts the burden
of proof onto CSM's extra mechanisms; reliable four-identity SET behavior alone
cannot justify them.

The next discriminating capability experiment should compare deletion/recreation,
snapshot COPY versus live LINK, and dependent answers after an upstream update.
Use the same semantic process information for CSM, the slot store and a competent
memory network. Separately retain an answer-only track. Introduce more identities
independently of longer histories, and later test learned extraction from language.
A conventional current-state store with the same extraction interface remains a
necessary control. Do not scale architecture merely because both toy models pass.

The broader question—greater reliability or lower full-session inference cost
than well-trained alternatives—remains open. The present work materially improves
its starting point: a stricter repeatable SET benchmark, a functioning equally
informed control, and exact lifecycle semantics for the next capability test.
