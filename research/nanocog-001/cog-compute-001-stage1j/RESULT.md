# COG-COMPUTE-001 Stage 1j: Fixed local exchange and causal span attention

**Result: the fixed neighbor exchange did not repair COG binding. Same-input span attention was the stronger control.** Without the exchange, span attention completed **239/288** changed-answer families versus **47/288** for COG, and **225/288** coherent reassignment families versus zero. Intersection of all strict primary gates: 1/3 span-attention seeds, 0/3 in every other arm. There is no across-seed qualification.

## Prospective matched comparison

Four fresh arms crossed reader (exact earlier COG interaction or same-window scalar dot-product attention) with an identical parameter-free sender frontend (off or $e_t+0.5e_{t-1}$, zero predecessor at position 0). The frontend is applied before the shared span encoder that feeds both keys and payloads; the direct receiver stays untouched. It adds one preceding token to each applicable sender span's receptive field without changing which candidate starts are allowable or the total raw input. It is not a learned boundary detector. COG uses independently sigmoid-gated channels and divides by window count. Dot-product span attention uses a scalar softmax across the same allowable windows; its receiver projections combine differently in the query. This is a comparison of two exact routing packages, not isolation of softmax alone, a full Transformer comparison, or a claim of optimized serving performance.

Each arm sees the same raw 12-token, 31-symbol grammar, prior overlapping trigrams, and no oracle fact slots, support hints, vocabulary restrictions, memory bank, or loops. Both use the same 67,847 trainable parameters, common tensor initialization within seeds 173, 271 and 907, sampled rows, receiver input, learned decoder, and next-token loss. Equal parameters and exposure do not mean equal FLOPs or timing. Each fit used 4,000 steps × 48 rows from 4,096 FIT prompts. The endpoint was fixed before scoring. Both readers passed the fabricated causal, input-support and worker checks; these are not competence tests.

The two primary panels contain 768 fresh rows on 96 semantic maps, with both queries and both physical fact orders in each gap-layout regime. The held key pairs are absent current FIT and both Stage 1h and 1i held data. Literal component facts appear in FIT in both ranks in the same case. The panels reuse the same maps, questions, physical order and fillers; only gap arrangement changes. Another 384 views coherently swap the two values on each changed-answer row. All-eight success requires both assignments × two queries × two orders; the altered inputs update every overlapping window consistently. This is constructed one-hop binding, not language or long-context generalization.

## Results

Cells below pool identical cases across seeds/layouts descriptively, not as independent trials. Exact means free greedy ANSWER, correct node, EOS. The supplied-marker diagnostic matched natural choice logits exactly. Primary gate requires strictly over 90% exact rows and complete four-example families within each held case, separately by seed and panel, plus familiar mastery strictly over 95%.

| Outcome | COG | COG + exchange | Span attention | Span attention + exchange |
|---|---:|---:|---:|---:|
| Familiar passes across all four cases | 3/3 | 2/3 | 3/3 | 3/3 |
| Known-layout primary pass | 0/3 | 0/3 | 1/3 | 0/3 |
| Unseen-layout primary pass | 0/3 | 0/3 | 1/3 | 0/3 |
| Known-layout changed rows / 576 | 248 | 257 | 512 | 530 |
| Unseen-layout changed rows / 576 | 260 | 241 | 516 | 528 |
| Known-layout changed families / 144 | 22 | 21 | 120 | 116 |
| Unseen-layout changed families / 144 | 25 | 16 | 119 | 117 |
| Original and swapped answers both exact / 1,152 | 55 | 65 | 953 | 995 |
| Same output node despite swapped truth / 1,152 | 1,021 | 977 | 186 | 145 |
| All eight reassignment examples exact / 288 | 0 | 0 | 225 | 202 |

For changed-answer rows, exchange helped 124 and harmed 134 cases in COG; it helped 92 and harmed 62 in span attention. On complete changed-answer families, the corresponding rescue/harm counts were 9/19 and 27/33. That is a net row change of −10 COG and +30 attention, but a net strict-family change of −10 and −6 respectively. An improvement in average pair accuracy did not imply more complete families. Neither exchange arm improved qualification. Without exchange, attention got all 576 relation-change rows and all 144 corresponding eight-case roots right. Its remaining failures concentrate in subject changes; no-exchange attention subject-change primary roots on known/unseen layouts were 5/5, 23/23 and 20/19 of 24 across the three seeds.

## Decision

Do not advance the fixed neighbor exchange as a COG repair. Keep the no-exchange span-attention implementation as the positive control for relation changes and the stronger overall comparator, while retaining subject changes as the harder test. Next compare smaller, explicitly matched changes to channel selection and across-span competition without changing candidate windows, frontend, or decoder. The results do not show that a particular normalization, architectural family, or cost profile explains the gap.

## Audit

17 actor-only Forge checks with zero findings; twelve FIT and twelve raw-only OPEN TrainLab originals; one CPU invocation with 338.551859 summed role seconds, not a serving benchmark; detached original scorer and independently matching raw recount. DATA `6b652aafb42320fea32f5eff6037faceb83b6238f3bb078a72eae7fb830b7f80`; LOCK `7318628566b8f8c937838762cb486ad9d26906a4e762a3d2548d0dc8f48924e4`; WIRE `9bb49ec1074ab03ff089323ace87a696b4a698bb175b8608fb54719fc52af50e`; score `5220e8f28b213998d8adba94d98b608d7e3056485bc5bfd4c5744e70572b69f4`; independent receipt `3f1b2c376b41f07e545d648cc827d5881f42cb2f2265f3f9a0cc82f7086c7a42`. [Stage 1i match/payload split](https://github.com/FPC-effortless/cdl-attention-experiment/blob/c654d256041c68818f3a357411a5a176fca0122b/research/nanocog-001/cog-compute-001-stage1i/RESULT.md).
