# Frozen final local gate probe

Descriptive audit found a repeated first-statement default in weak
ROLE-002 quarter-prefix, clip1, gated32 seed 1371, update 1280. Strong
reference is seed 1171 (preselected lower of the two passing seeds), same
recipe/endpoint. Test one component only: the four second-layer offset
gate scores at the last token. The ordinary query token and all value,
residual, and readout paths remain unchanged.

Use 32 fresh synthetic joint combinations, sampled with seeded draw
20261125 from the lexicographically sorted remainder after excluding
all 680 exact prior family keys (ROLE-001 FIT/CONTROL, ROLE-002/003
confirmation). Exact eight assignment/order/query worlds per family.
No protected/historical split. Query donor for each row is the world
with identical history and opposite final query. Actors see prompts,
not answer annotations. Freeze hashes before inference. Judging uses
all 287 logits and conditional margins for the two actual colors.

Per world run native capture; same-score rewrite (exact no-op); true
opposite-query score swap; centered-dose sham; native hook removal
(exact). Sham uses a deterministic nontrivial cyclic permutation of
the centered donor shift, choosing minimal absolute directional
cosine among three cyclic rolls, stable tie order. Report actual score,
probability shift norms and degeneracy (zero or collinear) without
excluding rows. Count all conditions and never choose only large donors.
No changes at other token outputs or to frozen weights; assert exact
baseline/parity with original unhooked execution.

Primary descriptive metric on weak checkpoint: family-paired difference
between true swap and sham in mean loss of originally aligned query
response on all 128 fixed-history pairs. For each pair, let r_i be the
margin for the correct color when that world's query is honored; native
response = (r_0+r_1)/2, computed again after intervention. A true swap
reducing that response more than sham supports a contribution of this
gate in hybrid states, not a complete mechanism; report the effect size
and paired family bootstrap percentile interval (32 clusters, 2,000
samples, seed 20261125), using no world-independence assumption. Report
both interventions, first/second target, standalone exact/paired scores,
rescue/damage and strong comparator descriptively. Treat an interval
including zero or substantially dose-degenerate sham as inconclusive.
Any native replay that does not reproduce a substantially larger
weak first-vs-second gap than the strong reference is a diagnosis
limitation, regardless of intervention effect. This is not Gate 1
qualification, a fit, or restoration of language-model capability.
