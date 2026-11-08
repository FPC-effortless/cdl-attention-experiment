# VB-002: V4 circuit on the frozen VB-001 task

Prospective local protocol, 2026-11-08. Commit source and protocol before fitting.
VB-001 baseline outcomes are already known; this is a follow-up, not a blinded
architecture competition or a replication of the unavailable original LRC task.

## Question

Can an indexed V4 circuit learn the same versioned two-hop task as the already
executed attention baselines, using identical label information? How frequently
do single-decision counterfactuals provide nonzero loss differences? Can a
controlled path require multiple repairs before any useful loss signal appears?

## Model and fairness boundaries

Reuse PagedMultiHopLM's actual routing/index/composition executor, extracted into
`_forward_states` without changing token-forward/cached semantics. The VB-001
sentence encoder design and learned occurrence-position embeddings replace its
byte embedding/local convolution frontend. Fixed-template word inputs and supplied
sentence boundaries match VB-001. No canonical entity keys or parsed graph enter
forward. Two V4 layers, width 20, four relation codes, eight addresses, two
candidates/read, three page bits; 19,888 nominal trainable parameters versus
19,076 Transformer and 19,129 memory network (about 4% more).

The layers each perform two dependent reads (four total at the query); baselines
have two full-set attention reads. This is NOT equal read count or identical
candidate access. Parameter counts include the final continuation head, whose
selected reference cannot affect a later layer. No padding parameters are added
for artificial matching. Compare complete cost, not a nominal interaction bound.

## Conditions: three seeds each, nine fits

1. `lrc_answer_cf`: answer CE plus one-coordinate counterfactual policy surrogate.
2. `lrc_support_cf`: same, plus support-based auxiliary loss, weight 1.
3. `lrc_oracle_control`: a separately labeled privileged competence ceiling.
   Training AND its primary evaluation force the two gold support occurrences
   through the actual capped index at both layers. This is NOT learned routing.
   Also report its ordinary learned-routing evaluation separately.

All use VB-001's exact generator, train/eval seeds, 8,192 training examples,
1,024 examples per test, seeds 301/302/303, AdamW .002, zero weight decay, batch 32,
2,000 updates, grad clipping 1, no schedule/early stopping/tuning. Same minibatch
order per seed as VB-001. Complete all nine fits before LRC held-out evaluation.
All reported model seeds remain in the results. Existing held-out baseline
outcomes were known when designing this follow-up.

## Credit and supervision details

The execution path is hard/greedy at both training and evaluation. Each training
step proposes a different action for each row at one uniformly selected
(layer, policy-field). Writer coordinates are uniformly sampled among the 16
records; other fields target the query. Multi-axis fields choose one coordinate.
Each row gets one alternative. Other fields recompute naturally downstream;
other coordinates of the changed field are frozen to the baseline action tensor.
The detached per-row CE improvement, clipped to [-1,1], multiplies
`log p(chosen) - log p(alternative)`; average over rows. This is a biased local
heuristic, not REINFORCE or an unbiased gradient estimator. Record every field's
trial count, nonzero delta count (absolute difference >1e-7), operand changes and
sum absolute delta. Two complete forwards per update, all charged to runtime.

Support labels are the SAME two occurrence indices as VB-001, repeated across
layers. A training-only soft address-agreement contrast over all 16 records
provides writer/query gradients: multiply relation-distribution agreement by
address-distribution agreement, normalize over records, penalize the gold
occurrence. Page/slot CE targets are derived from the support index and CURRENT
detached write assignments; no semantic address targets are supplied. Average
across layers/hops: agreement CE + (rank CE + mean binary page CE)/2.

This is an architecture-specific surrogate, not the baselines' identical loss.
It may itself cause failures, and soft agreement need not imply hard selection.
All-record comparison happens in the AUXILIARY TRAINING LOSS only; inference
retains the bounded index. High page bits get zero targets even when inactive.
No teacher-forced first read is used for the learned conditions. The oracle
control is explicitly separate and never credited as learned competence.

## Evaluation and diagnostics

Reproduce every VB-001 train/test tensor hash. Report all seeds, answer and
counterfactual-quartet accuracy, NLL, historical/latest accuracy, actual operand
accuracy and candidate admission for each layer/hop. Keep admission distinct from
selection. Report the denominator and answer accuracy conditional on both final
candidates being admitted; this is observational, not a causal estimate.

Every fit is evaluated with ordinary learned routing, memory disabled, and forced
support routing. Oracle-fit primary results use forced routing and are labeled.
Expected memory-disabled accuracy is exactly 25% by counterfactual construction.
Replay every metric/prediction from saved checkpoints, optimizer and RNG states.

Credit diagnostics use the first 32 IID examples, four deterministic repeats per
natural intervention size. Natural interventions vary 1/2/3 policy fields; these
counts do NOT establish the number of decisions required for a useful path.

A separate constructed conjunction holds other actions fixed and arranges a gold
second operand plus one newer distractor in a bucket. At the final layer, corrupt
its writer relation, writer address, and/or query slot to require 1/2/3 repairs.
A partial repair must leave the second operand NULL; only a joint repair selects
the gold record. Writer changes do not directly change the operator's relation
embedding, avoiding that shortcut. Exclude cases with no suitable newer
distractor, and report the count. Measure single-repair versus joint-repair loss
changes. This demonstrates a possible mechanism under a deliberate construction;
it DOES NOT estimate its prevalence or prove the cause of natural routing failure.

After fitting/auditing, measure complete batch-one CPU inference: 30 interleaved
blocks ×32 calls, 32-call warmup, one thread, including sentence encoding/index
work. Learned routing is timed even for oracle-trained checkpoints. Do not call
this generation throughput or an optimized hardware comparison. Training cost
includes both forwards and dense support-loss work. Do not combine 12:1 costs
without an amortization convention and competence gate.

## Integrity gates

Before fitting: original token/cached parity tests, strict causality, capped
candidate counts, exact oracle-to-support mapping through index, nonzero writer
gradients from the support loss, exact memory-off chance and controlled-conjunction
non-vacuity. Preserve original baseline source/checkpoints. All evidence, failures,
source hashes and deviations must be retained. No paid compute, merge or deploy.
