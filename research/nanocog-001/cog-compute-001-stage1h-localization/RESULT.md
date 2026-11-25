# COG-COMPUTE-001 Stage 1h frozen localization

**Result: choosing the right span mattered; merely isolating intact spans did not repair binding.** On the previously exposed Stage 1h cases where the query changes the answer, shifting existing gate weight toward the correct intact span raised exact free answers from **808/1,152 to 1,065/1,152** for the sender-conditioned model. It also raised complete four-example families from **127/288 to 239/288**. It rescued 257 answers and harmed none. This was an oracle intervention on unchanged weights, not a learned repair.

## Frozen workload and controls

Six exact Stage 1h endpoints, three seeds per architecture, 768 existing constructed prompts per endpoint, no optimizer updates. Each panel contains 384 prompts (192 changed-answer, 192 same-answer). Known and unseen gap layouts are the same underlying maps. Descriptive totals repeat cases across seeds and layouts; they are not independent sample sizes. A held panel in Stage 1h is exposed diagnostic material here.

We intervened at one answer decision after the model's naturally emitted marker. All later tokens are generated normally from the emitted prefix. Exact means the full free sequence has the correct marker, answer and stop, without a vocabulary mask. All original prompts emitted the marker. Native full-step logit tapes matched the original exactly. The no-op sham was exact, and jointly swapping gate and payload preserved emitted tokens (maximum answer-site logit difference 5.73e-6). Swapping just the two gate vectors is algebraically equivalent to swapping just their whole payload vectors; the two are not independent evidence.

At this decision site, every prior contiguous three-token window is normally included (10 windows). The correct-favored and rival-favored arms preserve the total gate weight of the two intact spans per channel and keep all payloads, all other gates, the receiver and the divisor fixed. Each moves that fixed mass as far as possible toward one span without exceeding the gate range. These two edits need not be equal distance from the native gate. Ground-truth offsets are used only to place interventions and score them. The intact-only and non-intact-only masks keep the original divisor but change the mixture's amplitude and content; the latter can still include fact fragments.

## Changed-answer outcomes

Each cell is exact free answers / 576 and complete families / 144, pooled over three seeds within its panel.

| Frozen intervention | Sender gate, known | Sender gate, unseen | Receiver-only, known | Receiver-only, unseen |
|---|---:|---:|---:|---:|
| Native | 409 / 63 | 399 / 64 | 396 / 62 | 381 / 56 |
| Only two intact spans, same divisor | 325 / 44 | 326 / 44 | 318 / 43 | 318 / 43 |
| Two earliest non-intact windows, same divisor | 163 / 2 | 165 / 3 | 167 / 2 | 168 / 1 |
| Swap intact gates | 404 / 62 | 394 / 59 | 396 / 62 | 382 / 56 |
| Favor correct intact span | 534 / 121 | 531 / 118 | 522 / 110 | 516 / 112 |
| Favor rival intact span (scored against original target) | 194 / 18 | 198 / 15 | 219 / 18 | 226 / 17 |
| Re-encode only the two intact payloads with exchanged value tokens | 391 / 59 | 385 / 59 | 389 / 57 | 370 / 51 |

Favoring the correct span increased the correct-minus-rival logit margin on all 1,152 changed rows for each architecture; favoring the rival decreased it on all. On the sender-gate model the rival-favored edit produced an exact rival answer on 664/1,152 cases; 375 were native-correct before the edit. The receiver-only model also responded (777 to 1,038 original correct on proper reallocation; 261 rescued, zero harmed). This is a useful comparison because its native gate lacked sender content, not query conditioning.

Re-encoding the intact payloads with their value tokens exchanged while retaining original keys and gates produced the expected changed answer on only **31/808** sender-gate rows that were originally correct (**22/777** receiver-only). Other overlapping windows retain the original value. This is a local path intervention, not a coherent prompt rewrite, and does not by itself identify a decoder failure. Equal-value cohorts are excluded above because they cannot distinguish which span won; the correct-favored edit preserved all 1,152 such sender-gate outcomes and 1,148/1,152 receiver-only outcomes, equal to their native totals.

## Decision

The frozen decoder can use the right existing value when given sufficiently selective routing, even with nearby distracting windows left in place. That makes query-to-span matching and gate selectivity the strongest architectural targets for the next learned test. The incomplete oracle result leaves other failures open, and dropping all overlapping windows was harmful as implemented. A prospective test should separate the learned inputs used to match a span from the value representation carried downstream; the end-to-end route must decide with raw tokens and no gold addresses on fresh families. No evidence supports adding persistent memory or more loop depth to fix this particular failure.

## Audit

Source: [Stage 1h exact parent report](https://github.com/FPC-effortless/cdl-attention-experiment/blob/aa2e97ac03edf1ae9500fab8caed61d42ab9b05a/research/nanocog-001/cog-compute-001-stage1h/RESULT.md). One input-pinned TrainLab engineering diagnostic, zero optimizer updates; 3.059465 elapsed role seconds, not a serving benchmark. Receipt SHA-256 `6fd7ff59a62ba645a5c49b23f6bf2ac093c896a2bb51bfb58efed41d59bc9b8e`; inputs manifest `682e781db2e25badd0923012a4cb21e8eead308766f11b373a36a983ce83a227`; detached tally `f6ee264ddd38bb980deaff9e043f6f6bf3d4544b7818397f11d50cdaddd2f874`. This is retrospective localization, not qualification.
