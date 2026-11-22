# ROLE-NAME-CONSIST-001 prospective outcome lock

Hypothesis: a bounded training loss aligning otherwise identical
name-substitution outcomes improves local binding on known weak frozen
ROLE-002 seed1371 without degrading first-clause decisions. Compare
exact same 552-parameter tap adapter, seeds 1771/1871/1971 and frozen
backbone, with coefficient 0 or 0.25 on companion margin agreement.
Only coefficient differs. Both arms see the same novel **schedule** of exact
old TRAIN rows and same optimizer/forward budget, not the old separate
batch schedule. Both still optimize answer CE and assignment foil, so
equal wrong answer preferences are not success. No remap metadata is
given to inference. Original native frozen checkpoint is read-only
reference, not refitted.

## Panel pins fixed before fits

Ordinary 120-family joint-combination panel: seed20261129, 960 rows
with full eight-cell each; 109 familiar and 11 unseen name-pair
families. Excludes all 928 prior family keys.
- Actor: `d74d4659d487806e74dce30e519d57a53fb39129c0d7ba46874e41fad404a9e9`
- Judge: `4b36c29d4dc1966163bddffde98d4e3e88b504455630c66532833c06cbb01459`

Secondary target-specific panel: seed20261130, 24 matched blocks/96
families, 768 rows; excludes all 928 previous and 120 new ordinary
families. Each block has two prior-fragile and two TRAIN-count-matched
controls, fixed values, same distractor, all query/slot/assignment
cells. This is an enriched contrast, not a representative uplift estimate.
- Actor: `a0369773833feb97349dac595bd176c2eb6f6fd6cb132b3251e15d329bd2c0c1`
- Judge: `e21a40fe1c67071caa6981ca646349b3829847495478a338dd57026b37e550f7`

## Primary and decision

Final 1,280 endpoint only for qualification. Ordinary must meet
**≥912/960 strict all-vocabulary words and ≥108/120 strict all-eight
families in every seed**, with ties treated as incorrect. Report
paired query/order/assignment successes and 109/11 name-pair strata.
Require first-clause strict count no more than five worlds below the
native frozen reference in every seed. The prespecified trained-arm
preference additionally requires, on the secondary panel, a strict
focal-second gain on fragile names in every paired seed and no losses
in robust focal-second count. Report actual deltas rather than implying
a practical improvement from only one newly correct row. If this
preference does not clear, no repair promotion regardless of whether
equal margins are easier to learn. No post-hoc alternate seed, loss,
threshold, or panel selection. Report zero-weight and native results
even if better than the consistency loss.

Secondary reports per group and role, 24-block paired bootstrap as
descriptive conditional on selected names and common backbone, not
row-independence or backbone generalization. Early 128 checkpoints
are descriptive on ordinary only; there is no early secondary selection.
Prediction outputs sealed and independently scored; no score given
to an actor, no panel opened by the fitting worker. No protected TEST,
MAIN, paid compute, broad language-quality, retention or speed claim.
