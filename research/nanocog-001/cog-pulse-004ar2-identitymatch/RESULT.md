# COG-PULSE-004A-R2: clock-free identity matching

**Removing variable timestamps from the matching branch eliminated clock-only gate flips on this diagnostic, but did not repair owner matching or answer accuracy.** No seed-by-composition panel qualified. Separating the inputs was sufficient for timestamp invariance, not for learning which records refer to the same owner.

## Comparison

The study reused 768 constructed training rows from the preceding composition-matched study. A fresh open panel had 384 rows with equal repeated-owner and three-distinct-owner bank compositions. Its 192 clock pairs changed only two timestamp tokens. Twenty-four repeated-owner pairs changed the actual latest record and answer; the other 168 kept their answers. The full banks and prompts were new to the checked lineage. No owner–value pair overlapped the immediate TRAIN or earlier open comparator panels; earlier studies contained all 56 constituent owner–value pairs, so this is not novel language or wholly novel knowledge.

The four matched fits crossed maximum cosine versus a learned per-record metric with the presence versus removal of variable timestamps from the identity input. The clock-free branch replaced the timestamp position with one fixed in-vocabulary token on a separate copy of the record. The encoder itself was frozen; this is a newly fitted match metric over existing learned keys, not a newly trained clock-free encoder. Each fit received only bank-level existence labels and the same optimization budget, without selecting an open threshold. The original record, clocks, router and tied 48-token answer head were unchanged. A positive primary gate chose the native top occupied record at unit strength; a negative gate used the original zero-read logits. Both timestamp-visible control fits reproduced the previous fitted arrays exactly.

## Open results

Counts are for seeds 173 / 271 / 907, each out of 96 per composition. Clock flips compare timestamp variants; complete gate or answer pairs compare a true hit and a true absence on the same bank.

| Exact metric | Cosine, clock-visible | Learned metric, clock-visible | Cosine, clock-free | Learned metric, clock-free |
|---|---:|---:|---:|---:|
| Clock-only gate flips, repeated owner | 4/22/18 | 6/20/14 | 0/0/0 | 0/0/0 |
| Clock-only gate flips, distinct owners | 18/12/24 | 20/28/18 | 0/0/0 | 0/0/0 |
| Both gate decisions correct, repeated owner | 36/50/26 | 24/36/40 | 24/24/28 | 36/36/24 |
| Both gate decisions correct, distinct owners | 46/36/16 | 44/30/32 | 36/24/36 | 40/20/28 |
| Both full answers correct, repeated owner | 22/32/10 | 18/24/20 | 16/16/14 | 22/14/10 |
| Both full answers correct, distinct owners | 22/24/12 | 22/12/12 | 14/12/10 | 12/2/16 |

The saved identity interaction features and fitted scores for every clock twin were bitwise identical in both clock-free arms. That is invariance by construction, not learned separation of time and identity. For the learned metric, removing variable time reduced complete hit/absence gate pairs from 206 to 184 and complete full-answer pairs from 108 to 76 of 576 seed-panel evaluations. It improved gate pairs relative to clock-free cosine (172) but not exact answer pairs (82). Clock-visible cosine produced 122 exact answer pairs. The clock-free metric's best owner retrieval on repeated-owner factual queries was 88/96, but the same evaluation retrieved a matching owner on just 48/96 for another seed. Matching any record for an owner is not the same as finding that owner's current record.

## Separate chronology diagnostic

Using the same clock-free learned metric and same gate, choosing the newest distinct visible timestamp among positive-scoring slots improved the 24 genuine opposite-latest answer pairs from **0/0/0 to 4/2/4**. The ordering rule was engineered, not learned. A false-positive shortlist can still admit a newer unrelated record, and a bank-level existence label does not identify every correct member of the shortlist.

As a separate grammar-only ceiling, exact owner-token equality plus visible chronology attained **20/22/16** opposite-latest pairs with the same answer head. It located the actual current source, but the head still emitted the exact factual answer in only **72/70/54** repeated-owner and **50/72/48** distinct-owner cases of 96. The oracle supplies a record, not an answer. Source selection and answer decoding therefore both remain material bottlenecks on this panel.

## Decision

Keep clock-free identity as a testable design constraint, but do not promote this metric or selector. Next, test whether training on **record-level identity contrasts** improves the shortlist: same owner across changed values and times versus different owners with matched value and time. Compare with the same bank-only fit and cosine, without passing literal IDs or correct slots at inference. Independently audit the frozen head before changing writer or long-context scale. The result does not establish that all clock-free encoders must fail.

All six bounded CPU roles ran once; only the small match metrics changed, no protected or paid work occurred, and an independent calculation reproduced the entire score. The examples share roots across interventions and seeds. No natural-language, serving-speed, or deployment claim is made.
