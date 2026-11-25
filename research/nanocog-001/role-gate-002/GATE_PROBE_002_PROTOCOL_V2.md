# ROLE-GATE-002 candidate: equal-dose probability-space direction test

Engineering candidate only. No scientific forward pass, fit, or scoring
has been invoked in this workspace. Do not dispatch until an independent
reviewer approves exact native executable, checkpoint, and input hashes.

The previous probe matched centered gate-score norm, but the actual
L2 displacement in softmax probabilities differed materially. This
candidate uses the **same** two frozen ROLE-002 quarter-prefix, clip1,
gated32 checkpoint bytes and same 32 synthetic families as ROLE-GATE-001.
That panel has already been exposed. Treat every result as descriptive
mechanism evidence, not fresh evaluation or model qualification.

Frozen input SHA-256:
- Strong1171 b2cbd5aa6cb1acd345f54bd562ece11e6d85d696b86cd87adc4cb13f8d3c810c.
- Weak1371 517c990e5834a1648524b59bd287f97f472583fa642ce0e5a8dc95270536e890.
- Label-free actor 44f380f0e934450e3c809162578381e670683602a4e52a4487c89fd2197a8116.
- Separate judge d5160ddcee95efcc28e6a9e042485cd9469ddae2827802e81efbdeea3cd1b848.
- Original receipt seal 430460c8260d09537940a1c5efbd034df411baf3f0d801400121e9d67483d766.

## Exact geometry

Intervene at the second layer's final query-position offset gate. Let
p be the native gate probability vector and q the same-history donor
probabilities under the opposite final query. Validity: four finite
nonnegative coordinates and normalized total. Let u be the unit vector
of q−p after centering to remove floating-point sum error. Choose a fixed
sham among nonidentity permutations of u with absolute cosine ≤.5.
Choose maximum common feasible radius by the formula below; resolve ties
by smallest absolute cosine, then lexicographic permutation. Geometry only;
no optimization over evidence labels, predicted answers, or effects. Define boundary b(p,d)=min over i with d_i<0 of
p_i/(−d_i). Use r=min(||q−p||₂, .99 min(b(p,u),b(p,v))). Apply the same
r in both directions, p+r*u and p+r*v, preserving strict support.
The .99 joint safety margin also applies if the requested donor dose
is near the boundary; this needn't reproduce a full transplant.

Null statuses are donor norm ≤1e−12, native probability zero,
no sham with absolute cosine ≤.5, and common feasible radius <1e−9. Keep all
such rows in the all-row contrast and set both branches to exact native
(no operation). Count them, plus radius-cap frequency and near-boundary
p_min ≤1e−6 separately. Report realized doses, requested doses,
r/requested ratio, cosine and signed rescue/damage by slot, with
quantiles both including and excluding geometric no-ops only as marked.
No selective re-run or post-hoc dose increase.

## Stage 1: native geometry only

One serial CPU FP32 attempt per seed with 45s cap, exactly 32 natural model
forwards in batches of 8 (256 rows); no intervention forwards, no optimizer,
no emitted answer logits, no truth file. At exactly the final token all four
local taps are eligible. Capture four natural pre-softmax scores and matching
native-dtype softmax probabilities per row. Re-normalize those probabilities
in f64 for ideal geometry alone. Retain row index/donor, raw values, caps and
ideal dose/status in geometry.json. Weights immutable. Independent reviewer
verifies frozen source, staging and receipts before opening geometry.

**Prospective practical-dose rule (v2; supersedes individual-row rule):**
For each checkpoint separately, a same-history pair qualifies only if BOTH
opposite-query legs admit ideal common radius ≥0.10 L2. Require at least
103 of 128 qualifying histories and at least 26 of 32 families with at
least three qualifying histories (four possible). BOTH checkpoints must
satisfy both requirements before considering any directional run. The
independent geometry judge recomputes this from retained natural p and the
fixed radius calculation, without model or task outcome access. Individual
row threshold counts remain descriptive. Falling short holds the
intervention and is not evidence that the gate has no effect. A later
separately reviewed intervention worker must meet these same thresholds using
realized FP32 doses, not merely ideal geometry. No post hoc relaxation.

## Separate later-stage safeguards before interventions

Use existing no-op/self/unhooked parity and no changes before final
position. Centered-score hooks alone are insufficient. A native
worker must match the **realized** probabilities in the actual inference
dtype, using directly validated normalized-weight replacement, or log(p)
scores with roundtrip checks. Measure actual runtime replacement norms
to same native p; do not rely on f64 nominal norms. Before dispatch,
independent review must define an exact dtype mismatch tolerance and
row-wise equal-treatment fail-closed policy. Keep runtime tiny/unmatched
rows as paired no-ops and count them, or fail the study on mismatch;
never silently score one branch with greater dose. Verify source and
checkpoint hashes, FP32 CPU one thread, no writes to frozen weights,
five branch forwards per registered batch only, ≤45 seconds per seed
and no fit. Seal both native outputs before the separate judge.

Primary proposed readout unchanged: weak1371 paired signed margin loss
under true direction compared with sham, averaged over 128 fixed-history
pairs and bootstrapped over 32 families (2,000 resamples; seed 20261125).
Strong1171 descriptive; report native/true/sham first vs second hits,
rescues/damages and the signed pair response. The independent runtime
review must bind equivalence tolerances and apply the practical dose guard; equal L2
does not equate every contract (coordinate mass, local curvature,
projection, or perturbation suitability). Do not claim a weak-checkpoint
root cause, general learned binding, speed or model promotion.
