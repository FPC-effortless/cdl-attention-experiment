# COG-COMPUTE-001 Stage 1l: One shared score versus channelwise weights

**Sharing a span's weight across channels made routing sharper but did not improve binding.** It reduced exact changed answers from **597/1,152 to 557/1,152** and complete changed-answer families from **59/288 to 51/288** relative to channelwise softmax. Neither COG softmax arm completed an eight-case coherent reassignment family. Identical-span dot-product attention completed **218/288** and again passed both primary held layouts in only one of three seeds.

## Frozen matched design

Four fresh width-64 arms consumed the same raw 12-token input, 31-symbol vocabulary, same strictly past contiguous trigrams and same no-exchange receiver, values, lag codes and decoder structure. The two primary arms use the same original COG score tensor (s_{i,c}) at the same model state, and softmax over the same spans. Channelwise independently softmaxes each channel; shared first takes the mean across channels and broadcasts one normalized weighting. Both have unit mass on nonempty reads and zero contribution with no valid span. This compares the fixed pooling choice and its learning dynamics, not pure sharing independently of concentration. The inherited 1k unit sigmoid and exact 1j span dot-product attention are reference arms, not extra toggles on the same mechanism.

Nominal width and 67,847 allocated scalars match across all four arms. The span-constant receiver bias has 4,160 mathematically cancelling scalars in both primary softmax arms. It remains functionally available in the other two read rules; no claim of identical effective capacity across the reference arms. Paired seeds 173, 271 and 907 use identical initial tensors, row draws, 4,000×48 exposures, AdamW, full next-token cross-entropy and fixed endpoint. No checkpoints, supplied candidates, support labels, candidate-output masks, persistent memory or loops.

4,096 FIT prompts; 768 fresh primary held prompts over 96 maps with both queries and both physical orders, rendered in both same-map gap layouts; another 384 prompts coherently reassign the two values on each answer-changing case. Exact held maps are new versus the tracked comparable splits; held key pairs are excluded from current FIT and Stage 1k-trained CAL, with earlier key-skeleton reuse disclosed in the data audit. All individual held facts appear in same-case FIT in both ranks. Layouts and seeds repeat the underlying maps; results are descriptive repeated counts, not IID trials or natural-language transfer.

## Exact free-generation outcomes

Each sequence must emit ANSWER, correct node and EOS unassisted. The public-marker check was a separate diagnostic, not a grant of a correct token. The predeclared gate checks strict rows and complete families per case, seed and panel (familiar >95%, held >90%), never aggregate counts alone. Both assignments exact counts completed original/alternate pairs. All eight means both assignments × both queries × both fact orders.

| Outcome | Channelwise softmax | Shared pooled score | Unit sigmoid | Span attention |
|---|---:|---:|---:|---:|
| Familiar passes, seeds | 3/3 | 3/3 | 3/3 | 3/3 |
| Both primary held passes, seeds | 0/3 | 0/3 | 0/3 | 1/3 |
| Known-layout changed rows / 576 | 304 | 281 | 300 | 521 |
| Unseen-layout changed rows / 576 | 293 | 276 | 293 | 512 |
| Known-layout complete changed families / 144 | 31 | 25 | 30 | 118 |
| Unseen-layout complete changed families / 144 | 28 | 26 | 29 | 115 |
| Both assignments exact / 1,152 | 126 | 108 | 125 | 939 |
| Same node despite changed truth / 1,152 | 982 | 975 | 963 | 193 |
| All eight examples exact / 288 | 0 | 0 | 0 | 218 |

Relative to channelwise, sharing rescued 84 changed rows but lost 124 the channelwise arm got right. It rescued 14 complete changed families and lost 22. The attention reference's remaining primary failures concentrated in subject changes (4/2, 23/23 and 21/21 complete subject-change families of 24 for known/unseen respectively); only seed 271 meets both held gates. Equal-value cases were near ceiling and do not by themselves distinguish relationship selection.

## Read diagnostic and next test

At the fixed public answer position on changed primary rows, mean span entropy normalized by log(valid span count) was **0.952** for channelwise, **0.863** for shared and **0.352** for span attention (1 means uniform). Applied score spread, measured as mean within-prompt standard deviation across eligible spans, was **0.385**, **0.719** and **2.036** respectively. These are separately trained endpoints; they do not measure a causal effect of changing entropy or a directly comparable latent scale. Projected message RMS was approximately **0.228**, **0.257** and **0.475**, so unit coefficient mass did not equalize message norms. Diagnostics are not part of serving time or qualification.

There is a concrete remaining structural contrast. At fixed receiver, the two primary COG scores are products of bounded tanh projections plus a span-constant bias; after cancellation, every applied logit is in [-1,1], including a mean across channels. With ten valid spans and softmax scale 1, even the largest possible span weight is at most exp(2)/(exp(2)+9), about 0.451. The dot-product reference has a different query construction and scaling, and is not subject to that bound. This motivates a fixed prospective logit-scale control at unchanged query, support and payload; it does not show that concentration alone will learn correct routing. Do not add memory or advance either tested COG scorer on these results.

## Audit

23 actor-only Forge checks with zero findings; 12 fresh FIT and 12 raw-only OPEN TrainLab originals, one CPU and 354.353767 summed role seconds. Detached scores and independent raw recount agree. No serving result. DATA `a84a7d3f01a95c8ef370c8e5c1625b0094090ba874c8065d43541b43e4e0c3cc`; LOCK `e72d07b49ab2b2b06b189abc788acbea350bc42728de31eb282d3ddd1c80b6a0`; WIRE `8463c81156a03553ac493452ab2430b92f907a45a0800aa66fd7c26dcc2d7117`; score `15f9408219bcc7a1e2cb20494d40a01a571e8565062557a6871d5e5a2eb048f3`; recount `dfd5165a4a7310de9ec1aaf177ac728bbf010bad0d0276d8703430779bbbabbf`. [Stage 1k trained](https://github.com/FPC-effortless/cdl-attention-experiment/blob/ec92a52f736bc7b9f7ea0e09888e1d8f138ac2e8/research/nanocog-001/cog-compute-001-stage1k-trained/RESULT.md).
