# COG-INFO-001: frozen stagewise accessible-information audit

Use the exact native ROLE-002 quarter-prefix/clip-1 final checkpoints
(weak seed 1371 and stronger reference seed 1171), no model update.
The interpretation is for two known checkpoints of one architecture;
the graded probe is itself a separately trained diagnostic, not an
improvement to the underlying network. Do not start INFO-002 intervention
until the observational trace and probe have been independently reviewed.

## Data and targets

Fresh synthetic seven-token factorial worlds, grouped by unordered literal
name-pair. Fit/held probe groups must not overlap; all lexical names and
values are the original vocabulary and occur in both where possible.
Exclude the 1,144 exact previously used research combinations. Keep actor
raw-token-only; opaque row order and exact receipts are sealed before
joining scorer annotations. No protected TEST or MAIN, no adaptation.

Score two targets separately:
- **Rendered binding role:** which of the two clauses assigns the entity
  denoted by the final literal query.
- **Assigned value:** which of the eight color vocabulary entries is
  correct. The probe trains on those global 8 classes, receives no
  candidates. At scoring only, report conditional performance by
  renormalizing against the two actual visible values. Eight-class
  log loss is not comparable to 287-class native LM loss.
For native LM logits retain strict full-287 and two-color conditional
readouts, only if emitted separately by the frozen actor.

A third, explicitly trivial identity sanity check verifies that distinct
literal token features are present; that is not binding competence.
No ordered names or support positions, candidate colors, oracle address,
solution bits or input labels are supplied to a probe. The raw literal
query representation may be concatenated to each stage's query-time
feature to estimate *additional accessible signal given the query*.
Distinguish this from a mathematical decomposition of all inputs.

## Prospective probe recipe

Use only row-aligned fixed numerical activations from each native query
time t=6. For each checkpoint and target, compare an intercept-only prior,
the literal-query-only feature, query + each registered stage feature,
and identical within-family TRAIN label-permutation negative controls.
Raw-literal and normalized-literal are standalone sanity controls, not a duplicated query feature. Only the three nonliteral stage tensors are appended to the normalized literal query. For real probes standardize each column using TRAIN mean/scale only
(scales ≤1e-8 use 1). One deterministic multinomial logistic solver with
fixed L2 coefficient 0.01 on feature weights only, zero initialization,
single CPU thread, L-BFGS-B max 300 steps and tolerance 1e-9 (report exact
convergence, no fallback on held evidence). Same solver for all heads,
no tuning or learned feature extraction; report actual dimension and
numeric feature rank separately. Do not pass padded zero columns off
as equal expressive capacity. Permute TRAIN labels independently within
each eight-world family with fixed RNG 20261131 and preserve family class
counts. Held labels stay exact; no training on held predictions.

Report held strict accuracy with exact ties counted separately, cross-entropy in nats, entropy of a
balanced target when known, and both-world/counterfactual consistency
by assignment, clause order and query. Use held name-pair groups as the
resampling units for paired stage-vs-query-only deltas (2,000 draws,
seed 20261131); report every stage, no post-hoc winner score. Preserve
raw outputs privately, aggregate externally. Never claim direct
mutual-information estimation from these probes.

For a normalized probe distribution, expected CE is conditional entropy
plus expected KL on the specified distribution. Hence population probe
CE is an upper bound on the corresponding conditional entropy; a held
sample estimates CE, not exact entropy. Restricted two-color CE is
conditioned on that candidate set; compare only like distributions and
label spaces. Failure of this fixed probe class is insufficient evidence
that the original representation lacks the relevant information. A good
probe or stage difference does not prove native causal use. Only a
separately reviewed INFO-002 intervention on fresh families may test that.

Information-seeking/active-inference mechanisms, memory size sweeps,
and complete inference comparisons remain later work, gated by the
local result. No model fits, activation capture or probe runs are
authorized by this text alone.
