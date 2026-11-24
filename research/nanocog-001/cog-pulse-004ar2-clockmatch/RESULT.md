# COG-PULSE-004A-R2: clock-twin consistency for memory applicability

**A training-only clock consistency penalty made gate scores less sensitive to timestamp swaps, but did not make the reader reliable.** It reduced mean absolute score movement in all six seed-by-composition panels, yet increased the number of categorical flips in two panels and failed every prespecified clock test. No candidate qualified.

## Comparison

The same 768 constructed training rows from the preceding composition-matched study supplied 384 reciprocal clock twins. A new open panel had 384 rows, balanced between repeated-owner and three-distinct-owner fully occupied banks. A clock twin preserves the query, records, their physical order, and the presence or absence of a matching owner; it exchanges exactly two timestamps. In 24 repeated-owner pairs, the exchange changes which record and value are current. In the other 168 clock pairs, the correct answer is unchanged. The new panel has no exact bank or owner–value overlap with either immediately prior composition-matched split. Older individual records recur (79 of 113 distinct open records), so this is not a test on entirely new language or vocabulary.

The controls were fitted maximum cosine and the original per-record gate. The third arm used the identical per-record gate with a fixed mean squared difference penalty on the two training logits of each clock pair. All three shared the original training rows, 128 Adam steps and a fixed zero logit cutoff. The two unpenalized controls exactly reproduced the prior fitted coefficients, training normalization and traces. The memory encoder, router and full 48-token answer head were frozen for seeds 173, 271 and 907. A positive gate passed the original router's highest occupied record to the frozen head at unit strength. A negative gate used the frozen zero-read logits, not a supplied NONE answer. Latest-record labels and answer labels were scoring-only, not inference inputs.

## Open results

Counts separated by seed (173 / 271 / 907); each cell has 96 opportunities. Clock pairs compare two timestamp variants. Same-bank hit/no-match pairs compare one factual and one no-match question on the same bank.

| Metric | Fitted cosine | Original per-record | Clock-penalized |
|---|---:|---:|---:|
| Clock-only gate flips, repeated-owner | 6/10/6 | 8/20/16 | 12/18/8 |
| Clock-only gate flips, distinct-owner | 30/18/34 | 20/22/10 | 14/16/20 |
| Both same-bank gate decisions correct, repeated-owner | 54/38/44 | 50/44/48 | 58/48/42 |
| Both same-bank gate decisions correct, distinct-owner | 48/42/36 | 40/36/42 | 38/42/46 |
| Both full-vocabulary answers correct, repeated-owner | 36/22/18 | 28/24/14 | 34/24/10 |
| Both full-vocabulary answers correct, distinct-owner | 32/24/20 | 28/18/20 | 24/14/20 |

Mean absolute clock-induced logit movement fell by 38–78% relative to the original per-record gate, but the total number of changed decisions moved from 96 to 88 out of 576 seed-panel pair evaluations. The clock-penalized arm correctly classified both sides of 348 of those clock pairs, versus 342 for the untreated gate and 344 for cosine. Clock stability alone is not competence: always accepting a bank never flips, but fails every no-match example.

Across the matching hit/no-match queries, the penalty improved exact paired gate decisions (260 to 274 of 576) while complete full-vocabulary answers fell (132 to 126; fitted cosine: 152). On the 24 repeated-owner timestamp exchanges that should change the answer, the penalty got both answers right in 2/6/2 pairs versus 2/6/6 for the untreated gate and 4/6/4 for cosine. Even supplying the actual current record at full strength gave only 74/80/56 factual answers per 96 on repeated-owner banks and 64/76/56 on distinct-owner banks. An oracle source locates a downstream limitation but is not a deployable policy.

## Decision

Do not promote this penalty or retune its threshold on this panel. No panel reached the maximum of five clock flips or the absolute matching requirement. This result rules out the tested soft logit penalty as a sufficient repair, not clock-insensitive identity representations in general. The next diagnostic should test a separate learned identity representation that does not use timestamps, while preserving chronology in the selector and isolating answer readout with oracle controls. Keep cosine and the unmodified reader as references. Autonomous writer training remains on hold.

All six single-core roles ran once. Only small applicability calibrators were fit; no publisher, router or answer-head weights changed. No protected cohort or paid run was used. An independent post-run calculation matched the entire locked score. Counts span shared constructed roots and repeated seed evaluations, not independent samples. No inference-speed or natural-language claim is made.
