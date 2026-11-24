# COG-COMPUTE-001 Stage 1e: familiar versus held selection

**Frozen diagnosis: the paired loss helped selection on familiar maps, but did not make it reliable there, and most of the gain did not carry to the held panel.** Confusing the two visible facts remains a major error even when the scorer suppresses every output not in the prompt. No additional training was performed.

## Exact scope

The twelve original [Stage 1d](https://github.com/FPC-effortless/cdl-attention-experiment/blob/75c791beada26bdf804c6a66fd5ded541f789e53/research/nanocog-001/cog-compute-001-stage1d/RESULT.md) synthetic checkpoints were evaluated once on all 1,536 original raw TRAIN prompts. No weights, model code, autoregressive policy, optimizer state or test budget changed. The original 384 held output tapes were reused by hash, not regenerated. The strict native unit is a complete family: both questions correct in both fact orders, with the model generating ANSWER, the correct node, and EOS. This is an after-the-fact diagnosis of those exact checkpoints, not a fresh qualification test.

The prespecified familiar subset keeps only original training maps with the same relation token set {13, 14} as the held panel: 512 rows across 128 families. Per seed those are 192 subject-change, 192 subject-invariant, 64 relation-change and 64 relation-invariant rows. The held set has 96 rows in each case. Physical support positions are balanced but joint query–answer distributions differ. Only 56/96 held rows in each subject case, 20/96 changed-relation rows and 0/96 invariant-relation rows have a familiar row with the same case, query key and answer. Thus the gap does not identify a causal effect of novelty alone. The changed-family totals below weight subject and relation 48:16 in matched familiar and 24:24 in held; the separate pairs of counts preserve that distinction.

## Native complete answer-changing families

Correct subject-change and relation-change families, respectively, per seed:

| Model and loss | Seed | Familiar subject /48 | Familiar relation /16 | Held subject /24 | Held relation /24 |
|---|---:|---:|---:|---:|---:|
| COG CE | 173 | 13 | 6 | 1 | 5 |
| COG CE | 271 | 5 | 7 | 0 | 4 |
| COG CE | 907 | 9 | 11 | 1 | 4 |
| COG paired | 173 | 20 | 11 | 3 | 5 |
| COG paired | 271 | 39 | 11 | 1 | 5 |
| COG paired | 907 | 14 | 11 | 3 | 4 |
| Tied attention CE | 173 | 8 | 3 | 0 | 0 |
| Tied attention CE | 271 | 11 | 3 | 1 | 0 |
| Tied attention CE | 907 | 4 | 0 | 0 | 0 |
| Tied attention paired | 173 | 21 | 10 | 1 | 5 |
| Tied attention paired | 271 | 22 | 11 | 1 | 8 |
| Tied attention paired | 907 | 17 | 8 | 2 | 1 |

Across the three seeds, the paired loss raised complete changed families in the familiar subset from **51 to 106 of 192** for COG and **29 to 89 of 192** for tied attention. Held changed families increased from **15 to 21 of 144** and from **1 to 18 of 144**, respectively. On the full TRAIN panel, COG reached 296/576 complete changed families with paired loss, tied attention 249/576 (ordinary CE: 150/576 and 82/576). No endpoint reaches the prior 90% changed-family competence standard even on its familiar maps. Counts across seeds reuse the test units and are descriptive, not independent trials.

## What restricting the answers diagnoses

The independent scorer also ranks the two literal fact values using the same full-vocabulary answer-step logits, with all other logits excluded **only in detached scoring**. This is an oracle constraint, not a native model capability. The one-value invariant cases are tautological under this constraint and are excluded from this diagnostic. Bit-exact fixed-marker agreement holds for ordinary free paths reaching the same marker, so changing format does not explain the result.

For paired-loss COG the detached restriction recovered only four complete changed held families, from **21 to 25 of 144**; for tied attention it recovered eight, from **18 to 26 of 144**. The actual right value had a logit no higher than the other visible value on **287/576** held changed rows for COG and **264/576** for attention, so preventing outside answers does not by itself fix selection. The familiar matched counts were 119/768 and 74/768 for the paired arms. Conversely, attention's constrained held individual changed results rose from 240/576 native to 312/576; this is compatible with outside outputs masking some genuine preference on particular rows, not with fixing whole families.

The experiment localizes two limitations in this regimen: incomplete fit of the complete selection behavior and poorer transfer to the particular held distribution. It does not separate optimization, representation, data coverage and distribution effects into unique causes. Before adding memory, depth, or another output penalty, establish that a learned control can master truly paired two-fact cases in calibration; then compare both models on a larger, more diverse set with explicit exposure controls, keeping the final multihop gate closed.

## Verification

12 native no-fit familiar evaluations ran once in 24.8893 serial role seconds on one core. The detached score reuses identical earlier held tapes; no held inference rerun. Independent exact recount agreed. There are no newly trained checkpoints, inference speed conclusions or promotion. The owner package links the original Stage 1d weight archive rather than duplicating it.
- DATA: `97ed099730e3ac291a077fce89493e1259510130fd4098469f3f785a67d8dcc6`
- LOCK: `87097638c08d758836ba5869e180472109755484bd37e7830d3a66d99c1b4f2f`
- WIRE: `92c6c99b7f2565ed0c1bd52f7a265b00fbdfe71147c054dd364cea6b80df3b63`
- Reused held WIRE: `cbd7d3d3c9fa53f167fd363247542c9894687816fac5d1762caa8f5041e0cba8`
- SCORE: `3bc201974c827bee08b641f24b51afbc0a3d02c172aaff23ad912efe296cc804`
- Independent receipt: `1a87dc8dce9e972f5d1ca7dd0435c354229a653a59ebfa3457fec8c8e1505346`
