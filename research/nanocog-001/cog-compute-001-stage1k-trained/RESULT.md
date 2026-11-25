# COG-COMPUTE-001 Stage 1k trained: Competitive routing versus unit gain

**Training with competition helped COG slightly, but did not repair binding.** Mass-preserving channelwise competition completed **45/288** changed-answer families, compared with **39/288** for original COG and **61/288** for normalizing the existing sigmoid weights. The matched scalar span-attention control completed **238/288**. The strict eight-case coherent reassignment check was **0, 0, 1 and 223 out of 288**, respectively, for original COG, mass-preserving competition, unit sigmoid and attention.

## Matched design

Four fresh width-64 arms; 67,847 parameters each, exact common initialization within seed, 4,000 updates × 48 draws per fit, same stratified rows, AdamW and full-vocabulary causal loss. No prior endpoint, external state, support label, fact mask, answer candidate mask or neighbor exchange. They read the same causal trigram candidates on the same raw 12-token, 31-symbol prompt. The direct receiver, sender and value projections and learned decoder are shared in structure. The two exact controls are byte-identical Stage 1j COG and bounded scalar dot-product span attention, not a full Transformer or optimized serving system.

For valid spans and each channel, let g=sigmoid(s), with n valid spans. Native COG uses g/n. Unit sigmoid uses g/sum(g), preserving relative weights at an identical model state while fixing total gain at one. The competitive arm uses softmax(s) times mean(g); this redistributes coefficients but preserves that model state's native total channel mass, with gradients through mean(g). Empty receivers contribute zero. Training changes the logits, so these contrasts do not decompose mediation, norm, or gain between separate trained endpoints. The exact scalar span attention differs in score construction and normalization.

4,096 FIT prompts, 768 fresh primary held prompts (96 maps repeated in two matched gap layouts), and 384 coherent alternate value assignments. The same primary map, query, order and nuisance sequence appear across gap panels. Exact held labeled maps are absent current FIT and the tracked previous compatible panels; current held key pairs are absent current FIT and Stage 1j CAL. Some earlier held relation key skeletons repeat (8/24 in Stage 1h and 12/24 in Stage 1i; exact maps do not). Every literal component fact occurs in same-case FIT at both ranks. This is short synthetic relational recombination, not unfamiliar language. The eight-case diagnostic crosses assignments, both queries and both given physical fact orders. Seed/layout totals repeat underlying families and are descriptive.

## Verified free generation

Full-vocabulary greedy sequence exactly ANSWER, correct node, EOS. Public-marker choice logits equaled logits from the naturally generated marker. A seed must exceed 95% rows and complete families in every familiar case and exceed 90% in every case within each held layout. No promotion from pooling or from the supplementary swap panel.

| Outcome | Original COG | Unit sigmoid COG | Mass-preserving softmax COG | Scalar span attention |
|---|---:|---:|---:|---:|
| Familiar gate, seeds | 3/3 | 3/3 | 3/3 | 2/3 |
| Both primary held gates, seeds | 0/3 | 0/3 | 0/3 | 1/3 |
| Known-layout changed rows / 576 | 258 | 286 | 275 | 523 |
| Unseen-layout changed rows / 576 | 265 | 292 | 274 | 523 |
| Known-layout complete changed families / 144 | 18 | 28 | 23 | 119 |
| Unseen-layout complete changed families / 144 | 21 | 33 | 22 | 119 |
| Both original and swapped answers exact / 1,152 | 61 | 151 | 148 | 936 |
| Same output despite crossed truth / 1,152 | 1,013 | 939 | 941 | 209 |
| All eight examples exact / 288 | 0 | 1 | 0 | 223 |

Relative to native COG, mass-preserving competition rescued 134 changed rows and harmed 108, with net changes +5/+18/+3 by seed. It rescued 27 complete changed families and harmed 21. Unit sigmoid rescued 149 changed rows, harmed 94, and had seed changes −4/+54/+5; it rescued 38 complete families and harmed 16. The unit-sigmoid comparison improved more than native-mass competition in this implementation, but neither came close to the required generalization. The sole strict attention pass was seed 271. The baseline comparison cannot attribute the attention advantage to a specific choice of normalizer.

## Decision

Do not promote the new reductions as a COG binding fix. Learning with a shared weight budget produced measurable row-level improvement, but still left nearly all coherent reassignment families unsolved. The same-input span-attention comparator again provides a strong positive control, especially for relation changes, while subject changes remain less reliable. A more informative follow-up would isolate scalar versus channelwise matching and query construction at the same coverage and decoder, rather than tuning more normalizers or adding memory or loops.

## Audit

20 actor-only Forge checks with zero findings; twelve fresh FIT and twelve raw-only OPEN native TrainLab roles, 347.202964 summed role seconds on one CPU, detached original scorer and matching independent recount. Run time is not serving evidence. DATA `b59db49390d132217ba081fc5348b4d9592bf384e7d091ceb608cb8e24b43cd7`; LOCK `ea982cda14e66486fc1f5c36aaa7e5cb7a0a6488df3ef03d4f37a0d0b23be894`; WIRE `e46dc15dbcc18d8b701e657d66f6be03734c16b905bd47b226968d02c5d7cd72`; score `116e1159b2177fea286c636cda9af7fd75d8f8ea3e68391a8974bd1ce2761899`; independent receipt `8cb35ded18e5e8aca720856eecec31dbdc7995992367c4ac1c5ff038f389f60c`. [Stage 1k frozen](https://github.com/FPC-effortless/cdl-attention-experiment/blob/c2858b3179ce582342c3ff4f47ef3b0c648cadde/research/nanocog-001/cog-compute-001-stage1k-frozen/RESULT.md).
