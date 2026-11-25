# COG-JOINT-LOCAL-001: joint local composition under paired role edits

**Prospective source/capacity draft. No cohort, fit, prediction, score,
publication, or protected MAIN/TEST is admitted.** This study follows the
negative frozen final-value residual, not another adapter or seed rescue.
Value-path V3 was 98/128 strict second-clause triples in both fitted arms
for all three seeds, with 30 wrong distractor insertions, ordinary
853–855/960 versus 858 native, and 70/120 complete ordinary families.
INFO002 previously showed that a value change can propagate without reliable
selectivity. Neither finding isolates whether the offset-gated architecture
could learn selective local composition jointly under an adequate regime.

## Three matched arms and two attributable contrasts

| Arm | Causal local mixer | Mixed-step answer aggregation |
|---|---|---|
| G-mean | Original ROLE gated32, native offset-gate architecture | Ordinary mean |
| G-tail | Exact same gated32 | Preregistered mean/role-tail mix |
| A-tail | Original rank-10 four-lag attention32 | Same mean/role-tail mix |

All 37,414 weights train jointly in every arm: embeddings, both causal
mixers, channel MLPs and output. Normalization operations remain fixed. No frozen backbone, residual adapter,
oracle, injected role/answer metadata, retrieval path, hand-coded fallback,
extra inferred prefix, or packed memory. Two width-32 layers, four causal
positions including current per layer, 287 literal vocabulary and fixed
norm/residual/readout. Gated architecture unchanged, including its native
offset gate, but its weights are trainable. Branching and raw-token inference
ABI unchanged. Use fresh paired seed IDs **2371, 2471, 2571**, not sought-for
successful checkpoints; equal initial full gated tensors within each matched
pair, exact same non-mixer initialization in attention32 from the existing
`build_arm`. Audit trainable scalar count, input span and initialization
hash before any real update. Attention is matched in parameter count and
causal lag reach, not op count, numerical mixing, optimization difficulty,
or a cached engine. Retain unchanged old frozen weak1371 as no-fit
preservation reference only; never substitute old fits for primary controls.

- **G-tail minus G-mean** isolates the answer aggregation under the stated
  joint regime, not frozen-vs-trainable scope, more data or new supervision.
- **A-tail minus G-tail** contrasts the local mixers with the same objective
  and exposure. With no A-mean, do not claim a mixer-by-objective interaction
  or universal gated/attention ranking.

## Single locked objective candidate

One mixed batch is six jointly arranged rows from one history and two
ordinary rows. Its six roles are both literal queries with (1) no edit,
(2) the first physical clause edited, (3) the second physical clause edited.
Use unreduced full-287 answer-token CE ( CE_i ) from the same logits and
label wire in all arms. For the six rows let

```text
M6 = sum(CE[0:6]) / 6
T6 = (logsumexp(CE[0:6]/1.0) - ln(6)) * 1.0
R6 = 0.5*M6 + 0.5*T6
answer_mean = (6*M6 + CE[6] + CE[7])/8
answer_tail = (6*R6 + CE[6] + CE[7])/8
```

Temperature **1.0 nat**, weighting **50/50**, fixed before outcomes. Logmeanexp
is across the same history's six conditional roles, not across six unrelated
rows or four family averages. Equal losses return their original common
value; the increment reallocates existing gradient weight toward painful rows.
The ordinary-only step always uses the mean answer CE across its eight rows.
Both mixed arms retain the exact same four edit penalties, two per physical
edit: `softplus(1 - (logit[correct] - logit[foil]))`, foil old target on
edited-name query and inserted value on opposite query. The edit margin
term is .25 times the mean of those four penalties, zero on ordinary steps.
No new label class or forward. Foils are stored in the order
[old_first,inserted_first,inserted_second,old_second] aligned to rows
2, 3, 4, 5; corresponding gold is gathered from those exact rows. Do not add a discretionary assignment-foil term
in one arm; existing legacy paired loss must be factored into only these
specified terms, not silently included. Gold and all bundle relationships
are TRAIN-only objective labels, never embedded, routed, or given to actors.

Locked common prefix term: hold ROLE-002 quarter-prefix **.25 ×**
mean next-visible-token CE on exactly positions 0..5, with row labels
equal to visible input tokens 1..6. There are exactly 48 prefix targets
per step. Position 6 predicts the answer only. No discretionary prefix
weight, loss mode or auxiliary sweep; all three arms use this exact common
term. This study does not identify prefix interference or demonstrate
open-language retention.

## Fixed exposure, runtime and TRAIN-fit adequacy

- Exactly 32 proposed TRAIN name-pair/color-triangle groups and four
  assignment × order base histories each. Both queries work through each
  history. 128 complete six-row TRAIN bundles, exactly five presentations
  each. 80 exact ordinary TRAIN factorial families, 640 distinct raw rows,
  each exactly ten presentations. No missing, duplicated or malformed
  counterfactual side. Keep immutable within-bundle order: two base queries,
  both after the first physical edit, both after the second physical edit;
  within each pair first rendered query then second rendered query.
- 1,280 committed updates per trajectory, exactly 640 mixed steps with
  one history bundle + two ordinary rows, 640 ordinary-only ×8; 3,840 edit
  rows + 6,400 ordinary = 10,240 answer rows, 71,680 input tokens, 61,440
  common prefix targets, 2,560 edit penalty terms. One mixed
  and one ordinary in every two-step block, paired PRNG schedule. At update
  128: 64 mixed, 64 ordinary, 384 edit and 640 ordinary rows. Save only
  exact updates **128 and 1,280** and retain unsuccessful runs. No sweep,
  selected seed/endpoint, retries, hidden optimizer resets, budget extension,
  or reordering after score. Undertraining remains inconclusive.
- Reuse ROLE native Forge/TrainLab/Veritas and Research Lab orchestration and
  audits, deterministic PyTorch 2.10 CPU, one thread, prefitted source/runtime
  and dataset hashes, adapter-free executor, separate FIT and redacted
  held actor/judge. Fixed AdamW peak .001, warmup8, betas (.9,.999), eps 1e-8,
  weight decay .01, clip1. One run per arm-seed, at most 180s process wall CPU
  configuration with runner watchdog/no paid acceleration; 9 serial runs,
  cap 27 minutes, practical runtime to measure rather than claim. Abort full
  roster on structural/cap failure; don't replace a seed. Total gradient
  update ceiling 11,520; exposure 92,160 TRAIN answer rows across the 9 fits.
- Predeclare TRAIN exposure-adequacy audit on same **1,408** distinct TRAIN
  actor rows in reset eval mode at update1,152 and1,280; no gradient, selection,
  new label or saved model at 1,152. One main forward per update + 176 audit
  batches at each = **1,632** forwards/run, 352 not learning presentations.
  Joint evaluator code proves causality, shape, strict scoring, and native
  stage receipts, with inert checks before admission. Report raw gradient
  norm/clipping and per-term values. Bounded evidence of fit plausibility:
  final strict ≥122/128 train triples per query stratum, ≥608/640 ordinary
  TRAIN worlds and ≥72/80 all-eight families, and ≤.02 nat absolute change
  over the six mean per-column full-vocabulary answer CE values on
  identical TRAIN bundles across the final 128 updates (the maximum absolute
  paired column drift). Show per-stratum curves and ceilings. This predeclared stability
  criterion is not mathematical convergence; failure keeps the result
  undertrained/inconclusive for mechanism attribution. If an arm's adequate-fit
  status differs across seeds, do not hide that with aggregate averages.
- Cost counts: equal forwards, data and updates are not equal FLOPs.
  Measure actual training wall and CPU, tensor dimensions, profiled operations,
  and declared analytic FLOPs per mixer (including masked dense attention work
  as implemented) on prespecified identical B=8, T=7 exposures; record
  profiled non-FLOP kernel/memory overhead separately. No rank/parameter
  proxy or process launch time can imply serving advantage.

## Fresh cohorts: analytical capacity is adequate but geometry is unproved

The existing 16 name × 8 color inventory contains
`C(16,2) * C(8,2) = 120 * 28 = 3,360` exact unordered name-pair/color-pair
joint keys. The last independently verified custody counted **1,432** prior
keys and **392** then-new value-study keys (all three edges of each triangle
reserved plus ordinary keys), hence 1,824 already counted and 1,536 residual
before any evidence of later usage. The proposed replication/data challenge
consumes 32*3 TRAIN edit edges + 32*3 held edit edges +80 TRAIN ordinary +120
held ordinary = **392**, leaving at most 1,144 keys by this ledger. This
does not guarantee even one fresh triangle in any particular name pair:
a triangle-free graph on 8 colors can still contain 16 edges; residual
average 12.8 edges/pair cannot establish triangle support. Do **not** reuse
older private held bytes or count the same edges twice to fill holes.

Next step only after separate admission: reconcile all subsequent exact
key-exclusion receipts and run a **count/feasibility-only** solver. Do not
produce assignments, held members, prompt bytes, or sample holdout before a
later separate DATA freeze. Treat each name-pair residual color graph as
eligible triangle options and ordinary edges. Need 32 distinct TRAIN edit
name-pair triangles (with 80 additional ordinary family keys on those
same TRAIN name pairs); 32 separate held edit name-pair triangles; 120
ordinary held families on a third disjoint lexical-pair pool. Each cohort
must cover all 16 names/eight colors, both baseline and inserted role cover
on both edit cohorts, all four assignment/order histories and both queries.
Original keys reserved jointly across all grammar permutations. Report
marginal frequency, capacity, eligible-pair counts, coverage and solver
status only; no selected solver vector. Structural floor: at least 64
triangle-capable name pairs, plus at least `ceil(120/28)=5` ordinary-held
pairs before coverage, and at least eight pairs necessary to cover 16
names. These floors are necessary, not sufficient.

If infeasible, stop this exact study rather than reduce the original
denominator, silently overlap name pairs, or relabel old held as fresh.
An independently preregistered **different experiment** may add a symmetric
fresh role vocabulary, e.g. 16 new name tokens have another 120 within-set
pair graphs and 256 old–new pair graphs, all with distinct joint keys.
Keep names randomized and their only meaning referential (not positional).
If intending structural generalization, use genuine alternate grammar,
e.g. a new clause linker, explicit subject/object wording or noninformative
interleaved tokens, crossed with edits and nuisance-only changes; precheck
actual causal receptive reach and pick an equally scoped attention control.
Re-tokenization changes padding/embedding and possibly the receptive field;
all three arms must be re-matched in active parameter count and causal
information, and exact historical weak1371 comparison/gates must live on
a separate eligible legacy-compatible panel rather than comparing an
unsupported vocabulary or sequence to old native. A new template alone
with the same consumed underlying key is not fresh. No fallback is approved
here, and no new language claims are licensed by this one grammar.

## Prospective held scoring and advancement

Freeze new panel bytes and complete judge joins **before any fit**. Learner
gets TRAIN only; held actors get only opaque raw prompts, not pairing or
answers. Separate weight-free verifier uses strict unique argmax of all
287 logits, ties fail. Independent audited parser verifies actual edit
positions/polarity and the same-history/opposite-query joins without scoring
any weights. Budget 18 fixed fit endpoints (two per 9 trajectories), plus
one exact old native prediction, 19 closures; two panels per closure,
768 edit + 960 ordinary rows, 216 B8 fresh forwards each, 45s per closure
only after separately reviewed native engineering dry test and FIT closure.
Never choose the best endpoint. Score only after the complete prediction
closure and independent score GO. No protected test, language benchmark
reclaim, or inference-cost promotion.

**Carry all existing quality gates unchanged.** At step 1,280 in every
seed, G-tail needs ≥7 additional complete rendered-second query triplets
out of 128 versus its exact G-mean pair; each triple is base strict gold,
correct inserted value after target edit, preserved unchanged strict gold
after distractor edit. No second-query increase in strict distractor
insertions or loss in strict unchanged gold versus **either** matched
G-mean or exact old native. Ordinary must have ≥912/960 strict worlds and
≥108/120 complete families, and lose ≤5 strict first-query correct worlds
vs exact old native on the same panel. Also prescribe a first-role guard
against the matched trained G-mean: no loss in ordinary first correctness
greater than 5/480, no increase in first-query edit distractor insertions
and no decrease in preserved original strict gold. Report exact paired
gain, loss, failure intersections, first and second strata, and interim
128 purely descriptive. Tightening these holds does not rewrite past gates.

**Prospective history safeguards, not another outcome-selected patch.**
For each physical edit and both opposite literal queries on the same base
history, require both target adoption and strict distractor preservation.
For the four-answer metric, require this under both separate edits of the
same base history. For any incremental G-tail promotion, no decline
relative to either G-mean or native on each of first-edit pair, second-edit
pair or four-answer complete counts (each /128); report paired losses and
rescues even if totals tie. No aggregate means over seeds masking a failure.
As descriptive continuous evidence, paired resample whole name-pair blocks
within panel: predeclare 2,000 draws with a fixed seed, retain pairing,
ordinary unequal-size groups use a ratio of summed correct counts to
summed rows. Uncertainty conditional on these seeds and fixture panels.

**Absolute competence independent of improvement.** Prespecify the same
practical absolute screen for all three arms per seed: ≥116/128 *complete
strict triplets in both query-role strata*, ordinary ≥912/960,
≥108/120, loss ≤5 first ordinary versus native; per-stratum no distractor
insertion increase or strict preservation decrease versus native; both
physical-edit pair and four-answer counts each at least native's.
Report absolute pass independently of all incremental assertions.
This added planning floor (at least 90% on both edit roles) is a proposed
practical requirement, not an old threshold or result. Use the simple
G-mean candidate if it clears this absolute screen unless G-tail also
clears it and earns the entire registered incremental gate. If G-mean has
>121/128 primary successes for a seed, +7 cannot be shown on this panel:
record ceiling-limited *incremental* evidence, never lower the bar or call
an absolute pass a failure. If G-tail qualifies and G-mean does not,
attribute only this matched objective contrast to improvement in this
regime, not exposure, causality of old errors, or architecture necessity.

**Architecture discriminator.** A-tail uses identical absolute screen.
Only both gated arms failing the all-seed absolute screen with every
relevant TRAIN-fit audit adequate,
while attention passes absolute in all seeds, merits a bounded
scaffold/mixer gap under this width, span and training budget.
If all fail or relevant arms are undertrained the comparison is
inconclusive for architecture; if multiple arms pass, preserve every result
and select no architecture without quality-matched cost. A-tail outcome
does not test mean objective interaction. Emphasize this is a local,
seven-token, exact identity fixture, not memory, natural-language
compositionality or general attention replacement.

Serving economics remains separate and uses the established 12:1 weight:
for quality-matched operational implementations and identical service
workloads measure fresh prefill, streamed/cached decode at batch1 and
predeclared larger batch, end-to-end logs, peak/state bytes and traffic.
For equal inference graph G-mean/G-tail no added algorithmic serving term,
but verify whole-policy cost rather than infer precision timing.
The old rank-10 attention implementation is batch-only; do not compare a
streaming gated model to forced attention recomputation as a matched serving
claim. Implement and validate a matched causal bounded cache under separate
approval before making this comparison. Publish no unreviewed outcomes.

## Reuse and lineage

- [nanoCOG draft PR #88](https://github.com/FPC-effortless/cdl-attention-experiment/pull/88),
  head 323865ca986eabd92a1cd9a6d78f6218577943ce: CI completed; the public
  text puts main/protected COG-COMM fitting on hold. Reuse exact ROLE gated
  and attention controls and native reproducibility checks, not old private
  actors or old weights as new training arms.
- Public ROLE-001 results at commit d8ffad99d860abfed680e354739813bf53e2bec1;
  ROLE-002 inherited a fixed quarter-prefix variant but did not qualify.
  ROLE-REPAIR-005 family-tail source-only draft influenced the test,
  not an already running matched fit. V3 did not promote the frozen
  final-value modification.
- Forge/Veritas invariant: run capability-adequacy and independent exact
  review at separate capacity, DATA, FIT, PREDICT, SCORE and optional
  cost stages. No atomic pipeline that skips a gate.
