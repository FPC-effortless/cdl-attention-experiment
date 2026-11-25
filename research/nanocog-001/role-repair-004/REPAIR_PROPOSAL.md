# ROLE-REPAIR-004 proposal: learn to select lag weights, leave backbone frozen

ROLE-GATE-003 demonstrated finite gate authority on individual known errors,
but showed no fixed setting is a repair. Test whether a content-dependent
increment to the gate can be learned. Keep the full original learned pathway.
This proposal does not authorize training or define a qualification run.

Both arms add the same 552-parameter MLP (68→8→1, no biases) at the last
query token of L2 and center the four bounded gate offsets before adding
to the native scores. Zero-init output gives exact native parity. Both arms
consume the same literal token-query embedding before smear and four
fixed lag indicators. Candidate additionally sees the frozen current/
historical normalized lag vectors; control sees one of four fixed invertible, lag-specific signed permutations
of the literal query in place of each content vector.
The original contextual native scores are unchanged in both arms.
No supplied entity role, correct gate, answer, or candidate branch is
available to the inference adapter. No cross-checkpoint tensor grafting.

Train on the original TRAIN only if admitted, never on oracle grid labels,
with matched schedule/batch budget/loss and frozen original parameters.
Answer CE and paired foil establish learnability; prefix loss is irrelevant
to these new weights if prior logits are exactly unchanged. Verify frozen
hashes, no-op and earlier-logit parity, finite nonzero gradient after the
normal zero-output initialization, actual active parameter/compute counts,
and standalone replay. Distinguish learned mechanism from original native
and parameter-matched query-only negatives. Use a prospectively frozen
fresh panel with no overlap and no old/protected TEST, and report full 287
strict ranks, all-eight families, both query/order pairs, second-clause gains
and first-clause damage. No per-world oracle policy.

A test confined to known seed 1371 with fresh adapter RNG seeds would ask
whether that frozen model is repairable, not whether unseen backbone seeds
generalize. If the existing qualification screen is invoked later, preserve
≥912/960 and ≥108/120 in every named seed; get new backbone/panel evidence
before an across-seed robustness claim or resuming retention/main work.

## Before fitting

An optional exact seven-token wrapper collects the actual frozen input
embedding before smear and patches the second local mixer scores only at
position 6. It bypasses untouched native computation when disabled.
Validate on no-fit, label-free exact TRAIN prompt projections; retain
full-vocabulary same-checkpoint parity at initialization and positions
0..5 with a fixed nonzero inert output weight. Report matched *nominal*
552 weights; do not assume equal independently useful feature rank.
Freeze actual rank across both arms at fixed relative 1e-6 and 1e-4
thresholds, reporting common literal+lag feature rank and the additional
source block rank. Rank is not hypothesis-class equivalence.
