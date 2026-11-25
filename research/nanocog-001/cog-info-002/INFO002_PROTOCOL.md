# INFO-002: localized causal gate/value-tap contrast

INFO-001 showed the fixed final probe did not repair weak-decoder
decisions, despite modest lower matched NLL. INFO-002 tests content
transfer at one exact site, not a model repair or a language result.
No dispatch, fit, scoring, retention/MAIN, or paid activity authorized
by this staging packet. One exact reviewed attempt per seed is required.

## Fresh frozen setting

Use only the same independently sealed ROLE-002 weak1371 and strong1171
gated32 checkpoints, 1280, quarter-prefix/clip1. Seed20261202 produced
32 distinct name-pair groups, each with two baseline colors x,y and a
third color z. The three joint families x/y, x/z, y/z are all fresh
against 1,336 exact old family keys. Each group fully crosses both
value assignments, rendered order and queried entity. For each base
row either change only first-clause value token to z or only
second-clause value token to the same z. The query and every other
token stay identical. Both registered edits are given to the actor by
row index, without truth, target/distractor distinction, or extra
features entering the model. Truth and role are recovered separately.

Actor SHA 41e1c19a805ac0061252113303d4bf4e5b0e2a3a3c582df4ba67f5763d858761;
judge SHA 3c403237f504fc4a40baa1fea144e00263a1d81ef67c03c6a99eee0e1c9a38f1.
No protected TEST or MAIN.

## Exact intervention

Keep the unedited input as receiver. On the second and final
LocalMix at t=6, capture G (four original gate scores) and U (four
existing per-lag contextual tap vectors before original depthwise
scaling). Use each independently captured donor to evaluate
G1/U0, G0/U1 and G1/U1 versus receiver G0/U0. Use native FP32 mask,
softmax, depthwise product, reduction and full-shape pointwise
operator, keeping original weights. Only t=6 local output delta is
replaced; unchanged receiver residual up to the site and original
cache stay intact. Subsequent MLP activations may respond, as they
should; their weights and readout stay frozen. The hybrid is not the
natural donor forward and can be off-manifold.

For every batch, feature capture must equal unhooked native on the
receiver, exact recompute of native LocalMix delta must agree, every
self-donor leg must be bitwise native, and all earlier local deltas
and full model logits must stay bitwise native. Require exact hook
removal replay and all weights unchanged. Per seed: 32 batches of
eight receiver rows, 416 one-thread native/instrumented forwards
(96 natural captures,192 hybrid,96 self,32 unhooked repeats), zero
updates. 45s cap, no retry. Retain 9 × 256 × 287 float32 final
outcome logit vectors under a native seal, plus receipt; no label or
score is emitted by the actor. This is not serving latency.

## Separate prospective scorer

Primary is weak1371, rendered-second queries, U-only tap transfer.
For ordered old edited value v and inserted z compare M=logit(z)−logit(v).
For target edit, use M01−M00. For distractor use **crossed assignment
sibling** (same ordered old→new IDs, rendered order and literal query,
query still targets second clause) with first clause changed:
its own M01−M00. Average target-minus-matched-distractor for both
assignments within row unit and then equal-weight each of 32 groups.
Predeclare 2,000 percentile bootstrap draws over name-pair groups with
Python Random(20261202), sorted endpoints 50 and 1949. Shared 16 names
and common backbone limit generalization despite disjoint blocks.
Evidence for this registered contrast requires lower bound >0 AND
positive net strict donor-gold rescues on U-only versus the same
receiver outcome scored against donor gold. Ties fail. The crossed
sibling holds the exact token substitution fixed, without implying
target and distractor originally have the same answer.

All nine condition strict full-287 outcomes, old/new pair margins,
target donor natural and distractor native changes, rescues/damages,
second-vs-first and strong-vs-weak, G-only, G+U, and
interaction M11−M10−M01+M00 are descriptive. Report paired
distractor harm separately. No adaptive cutoff or chosen subgroup
after outcomes. Failure of the registered probe does not prove
absence of binding information or inability of every causal edit.
