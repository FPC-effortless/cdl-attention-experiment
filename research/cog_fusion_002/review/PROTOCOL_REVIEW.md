# Independent prospective review — COG-FUSION-002

23 November 2026; before the six new fits and new held evaluation.

## Design judgment

The matched experiment can test whether exposure to postanswer continuation
during fitting improves the same continuous deployment process. Both arms gain
the same additional multi-turn examples and paraphrases. Therefore the contrast
between the two fitted arms, not improvement over frozen parents alone, is the
specific test of continuous-state training. The frozen parents are necessary to
measure practical change from the previous endpoints.

The experiment keeps the 001 architecture and factual executor unchanged. It
does not train or qualify the optional numerical composition attachment. Its
domain remains a finite-identity, synthetic linguistic task. Model/data/regime
admission and finite training adequacy are explicit prerequisites; no architecture
impossibility conclusion is licensed by an underfit endpoint.

## Trainer source review before locking

The reviewed `train.py` uses a fresh optimizer, identical family/rehearsal draw
streams for a paired seed, a separately evaluated teacher-forced loss branch,
and a greedy generated continuation. The teacher-forced branch's final state is
discarded. The continuous arm has no detach/no_grad boundary before a later
observation, while the reset arm replaces neural tensors after generation and
retains its ledger. Gold typed execution is confined to the first 100 updates.
These choices address the actual state-distribution hypothesis instead of merely
adding a second teacher-forced target after a fabricated continuation.

One configuration discrepancy was found before locking: the trainer's initial
CLI CPU-cap default was 3,000 seconds while the agreed protocol specified 3,600.
The owner aligned the default and launcher argument to 3,600 before fitting.
The cosine formula is recorded exactly; its final applied multiplier is
approximately 0.20 because the formula uses `(step-1)/1200`.

Source inspection is not sufficient on its own. Pretraining admission requires
executable checks for teacher-branch isolation, actual cross-answer gradients,
reset boundaries, ledger retention, variable-length generation and exact input
accounting, plus independent label/partition replay. Those receipts will be
referenced by `evidence/PRETRAIN_REVIEW.json`; no passed result is asserted here
before it exists.

## Required interpretation of the outcomes

Training-probe adequacy is evaluated with predicted execution and each arm's
own state policy, not with gold typed warmup. It is a fit diagnostic using
training examples and must not be represented as generalization. There is no
extension after seeing the probe or any held result. The fixed final checkpoint
is scored even if disappointing; cap or numerical failures stay explicit.

Absolute familiar repair, a causal carry-versus-reset advantage, new wording,
old-panel regression and language preservation are separate decisions. The
bootstrap resamples families while preserving their correlated seed comparisons.
The same 128 families across three seeds are not 384 independent examples.

The fixed eight-observation/three-question schedule remains a possible source
of shortcuts. Data admission verifies variable query plans, object keys, answer
lengths and event kinds, while a separate Session scenario shifts question
timing. LONG adds neutral delay; it does not add deeper reasoning structure.

The language gate is stricter and correctly anchored to the corresponding 001
fused parent. Recovery toward the original full native reference is reported
separately. Reused 001 language windows and old semantic panels are exposed
regression data. Their scores must not be relabeled new held evidence.

## Ownership and integrity

This review owns the protocol, the independent pretraining audit and the final
statistical summarizer. It does not modify 001 sources or train an endpoint.
The protocol, data, applicable source hashes and passing pretraining evidence
must be locked before the six fits. Later fixes to a locked fitting dependency
would require a new version rather than silently reusing the original lock.

## Pretraining admission completed

The independent `audit_pretrain.py` replay passed over 2,400 families, 4,800
worlds, 42,496 observations and 14,400 query answers. All constructed postevent
ledgers, query supports, evidence values, answers, partition hashes and changing
versus invariant denominators reconcile. There are no identical input-world
pairs or cross-family/split duplicate accepted histories. The first 128 TRAIN
families contain sixteen of each kind. Each TRAIN question position covers all
three query plans, all eight object identities and answer lengths four, five,
six and eight. The parent checkpoint hashes and unchanged 001 locked sources
also verify; independent state-dictionary accounting confirms 1,077,834 stored
and 997,674 active parameters for each parent.

The combined pretraining engineering receipt records **15 passed**, with zero
failures, errors or skips. It includes teacher-branch isolation, genuine
cross-answer gradients, explicit ledger retention across neural reset,
variable-length terminator handling, label-free candidate inputs and independent
whole-history scoring fixtures. `evidence/PRETRAIN_REVIEW.json` was written
before the 22:19:43 UTC training lock. No new held model was scored in this work.

The later statistical summarizer was tested separately before held evaluation:
seven fixture tests passed. These cover inclusive integer thresholds, inclusive
decimal NLL boundaries, family-level resampling without treating correlated
seeds as extra examples, rejection of a negative-seed exception, all-condition
training adequacy, actual evaluator-schema reconciliation and forged-read
rejection, and rejection of incomplete/failed endpoint evaluation despite
existing panel files. They add no training examples and change no locked fitting source.

One arithmetic issue in the already-locked language scorer was documented
prospectively in `LANGUAGE_GATE_ARITHMETIC_NOTE.json`: computing a relative
change by division then subtraction can round an exact boundary just above
0.05. Locked files remain unchanged. `summarize.py` independently applies the
protocol's inclusive comparison to the saved decimal NLL values and explicitly
reports any discrepancy. This is a numerical implementation qualification,
not permission to change thresholds after seeing outcomes.
