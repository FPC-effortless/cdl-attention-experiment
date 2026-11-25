# COG-COMPUTE-001 Stage 1k: Frozen competition at fixed channel weight

**Result: redistributing weight did not repair binding in the frozen COG.** Retaining native total weight in each channel while applying softmax across spans improved individual predictions and matched swap pairs slightly, but reduced complete changed-answer families from 47/288 to 39/288. No tested arm completed an eight-case value-swap family. Increasing channel gain alone gave a bigger individual-row improvement but also reduced family completion.

## Protocol and checks

One no-fit TrainLab diagnostic, three unchanged Stage 1j no-neighbor COG checkpoints, 768 existing primary prompts plus 384 coherent raw value swaps, 31-token vocabulary. These are exposed diagnostic cases, not newly held qualification evidence. At the decision after the naturally emitted answer marker, the same original logits, receiver, ten spans, values, and decoder were used. The residual message was changed only through the aggregation weights. After the changed token, normal autoregression continued from the actually emitted prefix. No answer labels or locations entered the actor. A detached scorer joined the original and alternate assignments.

For each channel with native logits s and N=10, let g=sigmoid(s)/N and m=sum(g). Native uses g; the identity uses (g/m)*m; unit sigmoid uses g/m; channelwise softmax uses softmax(s) with total weight either m or 1. The scalar arm uses softmax of the mean COG logit per span and multiplies by m; the uniform arm uses m/N. The intervention freezes all trained weights. Preserving coefficients' total does not preserve the realized message norm. Native full logits reproduced the parent tapes exactly. The identity sham reproduced every free output; its largest choice-logit floating-point difference was 5.73e-6. Maximum checked channel-mass error was below 3.0e-7.

## Results

Each arm sees the same repeated seed×layout rows (1,152 changed cases and 288 four-example families). The two layout panels use the same underlying maps. Both assignments exact is the number of original/swapped prompt pairs with the full correct free answer in each. Full reassignment means both assignments, both questions and both orders correct.

| Answer-site aggregation | Changed rows / 1,152 | Complete changed families / 288 | Both assignments exact / 1,152 | Complete eight-case reassignment / 288 |
|---|---:|---:|---:|---:|
| Native | 508 | 47 | 55 | 0 |
| Identity sham | 508 | 47 | 55 | 0 |
| Unit sigmoid (gain only) | 565 | 31 | 86 | 0 |
| Channel softmax, native mass | 517 | 39 | 73 | 0 |
| Channel softmax, unit mass | 567 | 26 | 107 | 0 |
| Scalar softmax, native mass | 509 | 43 | 71 | 0 |
| Uniform, native mass | 497 | 44 | 52 | 0 |

The main native-mass channel softmax rescued 58 and harmed 49 changed rows; it rescued 5 complete families and harmed 13. On matched swap pairs it rescued 48 and harmed 30. On changed rows mean normalized span entropy fell from 2.289 to 2.230 nats (uniform is ln(10)=2.303). Pre-/post-projection RMS perturbations were 0.0364/0.0469, compared with 0.0967/0.1431 for the gain-only arm. The distribution did change; the useful signal did not become consistently stronger at family level. The result is local to fixed temperature 1, these trained logits and this exposed panel.

## Decision

Do not plug frozen normalization into COG as a repair. The near-uniform native weights can become somewhat more selective without reliably selecting the right fact. Prior oracle rerouting showed that favoring the right span could help; this non-oracle change does not tell the model which span that is. A separate trained comparison on fresh cases would test whether introducing competition during learning changes score geometry. Such training was not run here, and neither a learned repair nor the broader limits of competition are established.

From [Stage 1j](https://github.com/FPC-effortless/cdl-attention-experiment/blob/a1de4495e9f7867e5586b928981a353eb8e16533/research/nanocog-001/cog-compute-001-stage1j/RESULT.md), the no-neighbor dot-product span-attention control got 225/288 eight-case families on the same evaluation prompts after training under its own routing, not this frozen substitution. That difference cannot be attributed exclusively to normalization.

## Audit

One native engineering run, 2.652980 elapsed role seconds, no optimizer updates. Original actor lineage and detached outputs were hash verified. Input receipt `858f0494d21aa95419a962b9f498f83153711682a579fc6842013c3761ba07e5`; TrainLab record `3d20460138d08446048b95a927a3c606c9d01a548f65c72fee10387c8d2db484`; detached tally `24f484157e43a39c08997044b11ddafd1398232b2b512b27a56f9e0540e44377`.
