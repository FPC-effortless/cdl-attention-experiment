# ROLE-GATE-001: the final local gate carries query-dependent selection

## How the failure repeats

On two disjoint 120-family panels, the **same** weak frozen quarter-prefix
checkpoint (seed 1371, clip1) answered 475/480 and 469/480 when the target
was in the first clause, versus 376/480 and 373/480 when the second
clause held the target. With identical history the model picked the
first clause's value for **both** questions in 102/480 and 105/480 pairs.
The two strong seeds did not show this large gap. This is a recurrent
shortcut under these fixtures, not universal first-position logic.

The independently reconciled 24-run observational analysis found that
doubling clipping switched the weak seed's preference toward the second
clause, yet 56 failed families overlapped across clip1 and clip2 on the
same 120-family panel. This is reported as descriptive structure; shared
names, colors, and paired trials preclude simple independence claims.
Weak late failures grouped more by repeated name pair than color pair,
which motivates more identity-sensitive instrumentation, not a causal
conclusion about the names.

## Forward-only causal intervention

On a fourth disjoint, prospectively frozen 32-family panel, hold input
tokens, value/residual paths, and model weights fixed. Replace only the
four layer-2 gate scores at the final token with the scores produced by
its opposite query on the exact same history. Separate controls use a
same-row no-op and a score-dose-matched lag permutation (sham).
Native no-op and hook-removal logits matched exactly; earlier
positions were unchanged and weights remained frozen.

| Frozen seed | Native strict worlds | Opposite gate | Sham | Native both queries |
|---|---:|---:|---:|---:|
| 1171 (strong reference) | 255/256 | 2/256 | 179/256 | 127/128 |
| 1371 (weak) | 223/256 | 33/256 | 143/256 | 95/128 |

Weak native first/second target: 128/128 and 95/128, confirming the
direction on this panel. Swapping the gate reversed the mean correct
two-color margin from +3.946 to -3.709 in the weak checkpoint and
+6.062 to -5.922 in the strong one. The sham left the mean margins
positive (+2.359, +4.077, respectively). Relative to sham, the
opposite gate reduced weak query response by 6.067 logit units
(descriptive 32-family bootstrap interval 5.264–6.935); strong 9.999
(9.670–10.309).

This intervention shows that the final gate is a potent causal lever
on query-dependent value choice in these frozen hybrid states. The
original question was never changed by the intervention. It does **not**
show why the weak gate sometimes defaults, that gate-only training is
sufficient, or that the directional difference is independent of
intervention strength. In particular the median true/sham probability
change was 1.157/0.260 (weak) and 1.274/0.477 (strong), even though
centered score doses match. The sham itself disrupted some outputs.
A preference for the true direction over equal *probability* dose has
not been tested here.

## Boundaries and custody

TrainLab native forward-only receipts: `20261122T172251-17ba163ea26e`
and `20261122T172255-28854fa81b51`. 256 prompts and 160 small model
forwards per checkpoint, 0 updates. Three readout conditions from
each job were sealed before scoring; all 287 vocabulary IDs scored.
No protected TEST, no MAIN, no paid compute, no retraining. This
mechanistic test is not Gate 1 qualification.

Prediction seal SHA256:
`430460c8260d09537940a1c5efbd034df411baf3f0d801400121e9d67483d766`.
Results SHA256:
`e6de817c126163de2340a74b202a759430a350828f4e39aa9f7357d95dc2772d`.

Next engineering target: test whether gate modulation for the queried
identity fails specifically on default-to-first cases. Equal-dose
probability controls and all-family dose/behavior traces can distinguish
direction from potency before choosing a narrowly targeted learning
intervention. Keep the original qualification thresholds.
