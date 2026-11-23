# COG-FUSION-002 — prospective multi-turn training-regime repair

23 November 2026. This protocol is fixed before the six fits and before scoring
the new held panels. COG-FUSION-001, including its failures, remains unchanged.

## Question and scope

Does training on the actual neural continuation produced by generated answers
repair interpretation of subsequent observations? The primary causal contrast
is continuous-state training versus resetting all neural state after each answer
during training, with both evaluated continuously. Improvement over frozen 001
parents measures the entire added training package, including added paraphrases
and supervision; it cannot by itself isolate state-carry training.

The earlier chest/Bela intervention is an exposed diagnostic motivating this
study, not a new held test. Failure of a finite fit, especially an inadequately
fitted training regime, is not an architecture impossibility result.

## Fixed models and initialization

Use the unchanged 001 FusionModel, strict typed executor and original
65-character head. No new trainable module, tokenizer, role token, vocabulary,
archive interface, generator restriction or optional numeric wrapper is added.
Each model stores 1,077,834 parameters, of which 997,674 are active; 80,160
retained native-board parameters are disabled. Neural continuation is 10,592
tensor bytes for one world. Fixed neural state does not bound archive storage.

Each of three pairs begins from its corresponding 001 memory endpoint:

| Seed | Parent SHA256 |
|---|---|
| 17 | `69180f8e9d05574eff19c78cf003f7e4aa21e5442e45a90ec2c09bffff553421` |
| 29 | `5f9699aa14f44c2bbb30f176d9a1886767e971a48235d7e5dab435a13f6d7522` |
| 41 | `5640ccce618b25c434b664934fc7e63986629cd6ec7bc5a790f785cfb7ee4332` |

The arms are `continuous` and `reset_training`. Within each seed, initial tensors,
model configuration, new optimizer initialization, family order, language-rehearsal
draws, supervision, learning rates and update count match. The three frozen
parents are also evaluated. These are three inherited language-model endpoints,
not three new independent language pretraining runs.

## Data admission and fixed partitions

The data use sixteen known entities, two relations and at most two typed reads.
Targets must be representable by these interfaces, fit the twelve-character cap,
and contain only inherited characters. Intended ledgers, supports and answer
labels are independently replayed before fitting. Candidate inference receives
input text and its own predicted operations and reads, never reference labels.

| Partition | Paired families | Worlds | Status |
|---|---:|---:|---|
| TRAIN | 2,048 | 4,096 | Only fitting corpus |
| TRAIN adequacy probe | 128 | 256 | First 128 TRAIN families; 16 of each kind |
| DEV | 64 | 128 | Separate diagnostic; no checkpoint or budget selection |
| Familiar TEST | 128 | 256 | New semantic histories; familiar training wording |
| New held wording | 128 | 256 | New lexical and syntactic templates |
| LONG | 32 | 64 | Neutral-delay extrapolation |

Eight balanced kinds are direct, twohop, correction, irrelevant, retraction,
bridge, missing and cancelled. Each ordinary world has eight observations and
three questions, after observations three, five and eight. Actual generated
answers are consumed before the next observation; no extra unmodeled separator
is inserted. LONG adds 32 neutral observations immediately after each of the
first two answers, giving 72 observations and three questions.

Training wording includes the old familiar bank and new training paraphrases.
Neither the old 001 held templates nor the new 002 held templates enter TRAIN.
New held wording changes vocabulary and syntax but retains the entity names.
Reject identical text-world pairs and duplicate canonical accepted histories
across partitions. Within cancellation pairs, accepted state can intentionally
agree while proposed edits differ. Record changing/invariant denominators at
each question and over the complete answer sequence.

The fixed schedule is a scope limit: query types, target keys, answer lengths and
postanswer event kinds must vary, but this remains a structured distribution.
A separate runtime scenario shifts question timing, including a first question
after two observations. Its result is a schedule-shift diagnostic. Old 001
familiar/fresh/long panels and the exposed live scenario are regression panels,
not newly held generalization evidence.

Canonical TRAIN data are stored separately at `evidence/data/train.json`.
The trainer reads no DEV, TEST, held-wording or LONG file. The manifest binds
every split, the combined corpus, wording banks and probe indices by hash.

## Fixed training regime

| Setting | Fixed value |
|---|---|
| Pairs | Seeds 17, 29, 41 × two arms |
| Updates | 1,200 per fit; fixed final checkpoint |
| Batch | Four paired families / eight worlds |
| Typed warmup | First 100 updates use gold typed writes and query execution |
| Remaining execution | 1,100 updates use predicted writes and queries |
| Answer rollouts | Actual unrestricted greedy generation throughout |
| Native learning rate | 0.000075 |
| Interface learning rate | 0.0005 |
| Optimizer | Fresh AdamW, default betas, weight decay 0.01 |
| Gradient clipping | Global norm 1.0 |
| Schedule | Cosine factor `0.2 + 0.8*(1+cos(pi*(step-1)/1200))/2` |
| Rehearsal | Four windows of 128 TRAIN character targets every update |
| Rehearsal loss weight | 0.5 |
| CPU policy | One Torch thread per fit; six fits may run concurrently |
| Hard process CPU cap | 3,600 seconds per fit |

There is no adaptive extension, early stopping, checkpoint sweep or held-result
selection. A cap failure retains an explicitly incomplete checkpoint and does
not silently change the budget. Canonical TRAIN-only forward/backward timing
before this lock measured warm continuous and reset batches at 1.141 and 1.317
CPU seconds, respectively; first compiled batches took 3.144 and 1.682 seconds.
Peak process RSS was 844,376 KiB. No optimizer update occurred. These resource
admission measurements are recorded in `evidence/CALIBRATION_TIMING.json` and
are not model-quality measurements.

Observation/query semantic losses, strict-route supervision, teacher-forced
character loss and paired-answer loss retain the 001 forms and relative weights:
semantic + 0.5 route + 2 answer + 0.2 paired + 0.5 rehearsal. Observation and
question contributions are averaged within their respective event groups;
answer/route/paired losses are averaged over the three questions.

For each answer, the teacher-forced loss branch begins at the same preanswer
state and uses the candidate's selected evidence, except declared typed warmup.
Its final state is discarded. The independently computed greedy rollout consumes
the answer prefix and actual emitted characters, including a generated period,
once. Only that state can enter the next observation. Gold answer spelling and
length never choose the carried state or stopping point.

The continuous arm preserves the full graph through the actual generated
continuation. Argmax token selection is discrete; subsequent recurrent operations
on the selected tokens remain differentiable. The reset arm still computes the
actual answer but replaces all neural continuation tensors with `init_state`
afterward. Both retain the factual ledger. At a new independent world, both
initialize neural state and an empty ledger. All active inherited weights and
interfaces are trainable; disabled-board parameters remain inactive.

## Implementation and provenance admission

Before fitting, require a passing independent corpus replay and engineering
checks demonstrating: teacher-target changes cannot change the carried rollout;
later losses reach earlier generated backbone state in the continuous arm and
not across a reset boundary; resets preserve the ledger; variable-length answers
consume the terminator without hidden padded characters; and token counters
match actual calls. Verify paired initial tensors, parent identities, parameter
counts, unchanged 001 dependencies, and finite model states.

Write `evidence/PRETRAIN_REVIEW.json` before `TRAINING_LOCK.json`. The training
lock binds this protocol, the fitting/evaluation sources, data manifest and TRAIN
file, engineering/data review receipt, parent checkpoints and immutable imported
001 sources. No new held performance is inspected before locking the campaign.
Evaluation runs only after the fixed fitting endpoints or explicit failed-run
receipts exist. Evaluation source and new data hashes remain visible.

## Metrics and prespecified decisions

Primary held evaluation uses **continuous** state for every candidate and frozen
parent. A complete world requires every proposed operation, every postobservation
ledger, every query interpretation, every support and every generated answer to
be correct. A complete pair requires both complete worlds. Merely producing two
correct final strings is insufficient. No-op operations ignore irrelevant key
and value labels; deletion ignores unused value labels.

For a 128-family ordinary panel, denominators are 768 answers, 2,048 observations,
512 observations immediately following an answer, and 1,280 observations after
at least one answer. Report meaningful-operation subsets separately, using
reference action != HOLD to define meaningful, never the model's predicted action.
Also report question number, task kind, answer-changing/invariant outcomes,
query-support joint correctness and each world's complete-history outcome.

### Training adequacy admission

Evaluate the fixed first-128 TRAIN probe with fully predicted execution and actual
greedy answers under **each arm's own training state policy**. Require all of:

- At least 99% exact meaningful postanswer operations.
- At least 99% jointly correct query interpretation and supporting reads.
- At least 95% exact generated answers.
- At least 90% complete world histories.
- All 1,200 updates completed, finite losses/gradients, and implementation admission passed.

Score this after fitting, without modifying any endpoint. A missed adequacy gate
means learning adequacy was not established at this finite budget. It does not
falsify the architecture. A reset arm that fits only under its reset policy may
still fail continuous deployment; that is the intended comparison.

### Familiar multi-turn repair

Each continuous endpoint must achieve at least 95% exact answers and 95% complete
paired histories on familiar TEST, and at least a 10-percentage-point complete-pair
gain over its own frozen parent. Report these conditions separately and require
training adequacy before calling the endpoint qualified for this familiar regime.
All three endpoints must pass for a three-seed repair claim.

### Specific causal benefit of continuous training

The primary contrast is complete-pair accuracy of continuous-trained minus
reset-trained endpoints on the same familiar TEST families, evaluated continuously.
Require a mean gain across seeds of at least 10 percentage points, no negative
seed-level gain, and a paired-family 95% percentile-bootstrap interval whose
lower bound is strictly greater than zero. Use 20,000 resamples and NumPy RNG
seed 20029017. One bootstrap sample resamples 128 family indices with replacement
and retains all three seed comparisons within each selected family. Seeds are
not counted as independent additional test families. This interval is conditional
on these three starting endpoints, not a population guarantee over pretraining.
Require training adequacy in both arms before a qualified regime-attribution claim.
If this contrast fails, an absolute repair can still exist without evidence that
carrying state during training was specifically necessary.

### Novel-wording qualification

For each continuous endpoint, require at least 95% exact answers and 95% complete
paired histories on the new held-wording panel. This is separate from familiar
regime repair. A failure must remain visible and prevents a broad wording claim.
LONG is a separately reported extrapolation panel; it does not qualify arbitrary
reasoning length. Report the same absolute 95% answer/pair reference on LONG,
without converting it into a new architecture-wide claim.

### Regression and ordinary-language preservation

For every continuous endpoint require: old familiar answers >=99% and complete
pairs >=95%; old LONG answers and complete pairs >=95%; and old FRESH answer and
complete-pair accuracy no more than 0.05 below the matching frozen 001 parent.
These already-exposed panels remain regression tests.

On the same fixed 001 language windows, require NLL <=1.05 times the corresponding
001 memory parent's NLL. Also recompute the original full native reference and
the admitted local-only parent. Separately report whether NLL <=1.10 times the
original full native NLL; passing the paired-parent gate does not establish that
the original full capability has been recovered. These fixed windows are exposed
regression evidence, not a broad new language benchmark.

All numerical thresholds are inclusive except the bootstrap lower bound >0.
Use exact counts/fractions for discrete gates, not rounded percentages. No failed
primary decision is rescued by a successful diagnostic or a best-seed selection.

## Diagnostics, runtime and costs

On a preselected balanced first-32 familiar TEST families, probe both first
observations immediately after the first two answers with natural state, all
neural state reset, all backbone-layer states reset, backbone-hidden-state-only
reset, and decoder-controller reset. Preserve the ledger and identical next
text. These interventions compare operation correctness and
key margins; they are not candidate successes. Any supplied-correct-evidence
decode is explicitly diagnostic. Report actual Session continuation/restart,
scoped pending, source retraction, request replay and token-accounting checks
separately from semantic accuracy. The shifted-schedule runtime scenario has no
post-hoc pass threshold.

Count observation/query characters, greedy-prefix characters, generated
characters, teacher-loss input characters/targets, rehearsal inputs/targets,
padded physical positions and actual decoder calls separately. Realized generated
lengths may differ between arms. Equal examples, supervision and updates do not
imply identical realized token counts or FLOPs. Retain process CPU/wall time,
batch-order/LM-draw digests, endpoint/optimizer states and source/data hashes.
No efficiency-superiority or attention-replacement claim follows from this study.
