# COG-COMPUTE-001 Stage 1h: Latent span mixing without supplied fact boundaries

**Result: sender-conditioned mixing helped familiar examples but did not qualify relational binding.** Both arms failed both held-out panels in all three seeds. The failure already appeared on held facts in familiar layouts, so merely moving the facts into new layouts is not a sufficient explanation.

## Frozen comparison

Both fresh 67,847-parameter models receive the same 12 raw tokens: two intact subject/relation/value triples, three nuisance tokens in variable gaps, then a public question delimiter and two query tokens. Some nuisance tokens are ordinary symbols used in facts. A shared encoder consumes every complete, prior contiguous trigram, including ones that cross fact boundaries. The model gets no fact-start IDs, support mask, external parser, persistent memory, answer mask, or symbolic lookup. This is a short synthetic grammar with supplied span length and visible query delimiter, not free-form language.

Both arms use the same sender values, relative lag code, direct receiver, and learned full-vocabulary decoder. The interaction gate uses the activation of each candidate span in addition to its lag code; the control gets only the lag code on that side of the gate. The receiver conditions both. Channelwise sigmoid-weighted values are summed and divided by the structural number of positions, not by the sum of learned weights. This is content-conditioned, attention-like interaction in the broad sense, but not softmax dot-product attention. The control is a matched receiver-only mixer, not a Transformer baseline. A fabricated intervention changed only the candidate's gate when its sender was changed and the receiver held fixed; this verifies wiring, not useful semantics after learning.

All fits used the same 4,096 training rows, paired sample draws and initialization within seed; each had 4,000 AdamW updates × 48 rows with ordinary next-token cross-entropy. OPEN was unrestricted greedy generation requiring exactly ANSWER, correct node, EOS. Fixed public marker logits exactly matched natural generation where emitted, across all rows. No new endpoint was selected after observing results.

The two held panels have the same 96 semantic families (24 per case), with both queries and both fact orders, 384 rows each. Each individual held fact appeared in FIT for both physical ranks; held pairs of keys did not. The panels keep the same facts, query, order and ordered nuisance tokens and swap familiar and novel combinations of three gap lengths with matched marginal fact positions. This tests new combinations and local layout changes, not new vocabularies.

## Results

| Qualification | Receiver-only | Sender-conditioned |
|---|---:|---:|
| Familiar rows and families, all four cases, passing seeds | 2/3 | 3/3 |
| Held key pairs in familiar layouts, passing seeds | 0/3 | 0/3 |
| Same held key pairs in novel layouts, passing seeds | 0/3 | 0/3 |
| Held familiar layouts, changed-answer families completely correct | 62/144 | 63/144 |
| Held novel layouts, changed-answer families completely correct | 56/144 | 64/144 |
| Held familiar layouts, exact rows including equal answers | 970/1,152 | 985/1,152 |
| Held novel layouts, exact rows including equal answers | 955/1,152 | 975/1,152 |

Family completion means resolving both queries correctly in both fact orders. The preregistered held gate requires strictly above 90% of rows and families in every case independently, for each seed. The identical test families recur across seeds; totals are descriptive, not independent sample counts. Sender-conditioned novel-layout subject-change complete families were 12/24, 13/24 and 11/24; relation-change 8/24, 11/24 and 9/24. Its equal-value cohorts were perfect, which does not show that it can select the right relationship when answers differ. On novel layouts, the new gate rescued 16 changed-answer family/seed outcomes but lost 8 the control got right. Within the new gate, 7 changed-answer families were solved only in the familiar layout and 8 only in the novel layout. Of 177 incorrect novel-layout changed-answer rows in the new gate, 127 chose the other fact's value.

## Decision

Do not treat Stage 1g's fixed-boundary positive control as a solution in this setting. Adding sender content to the gate did not make the relation reusable on the full test. Because the comparison changes layouts but never supplies boundaries or independently alters key versus payload paths, it does not locate the remaining failure inside span identification, matching, mixing or readout. A frozen diagnostic can first remove cross-boundary distractors, then supply separate key/value activations or swap value activations under an unchanged query. Keep those oracle results distinct from native competence. No persistent memory, looping, scale or serving performance claim is supported.

## Reproducibility

Single-CPU bounded invocation: six fits and six raw-only opens, 170.052034 summed role seconds (operational record, not a serving benchmark). Independent recount agreed with the immutable original on all counts. Exact native lock: `916100e4952ad5be7d21170db3d532896c94265e8d14e2d40985ecc1e71e7591`. DATA: `f8d7fdc711fef2a0217bb66e8c0caef2c3e4ef3eb56c27cee8597a9f646ddbc8`. WIRE: `639cdda479ff79ff3d5273a0cfa1842e842565b2bd0ec1d2eeabcd99f11b5ee7`. Score: `3f4d6ccd710cf33f6e9b8e4f66e993733e38155baeadea11659ce1f8005297b2`. Independent recount: `a1b4b8d3e1c1039748a26bcdb5e351afae6e043da52bd2a032f2d232c4bc7109`.
