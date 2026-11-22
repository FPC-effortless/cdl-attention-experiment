# COG-BIND-001A: support supervision helped, but did not qualify

With the same learned reader and dot-product inference, adding a training-only support-index loss improved both strict support selection and native answer accuracy. **No run passed the full study gate.** All three answer-only and all three support-supervised fits fell short in calibration and at least one held mode. This was the one-hop selector stage of COG-BIND-001, not a test of a general semantic binder.

The arms shared 53,958 parameters, paired initial tensors and batch schedules, fixed boundaries, input vocabulary and encoder/reader architecture, all training examples and 1,024 optimizer updates. The sole intervention was auxiliary support-index cross-entropy at weight 1 during training. Both native and own-top-one held actors read prompts only; no gold support or answer during inference. Total measured fit-process time was 99.1 seconds across six CPU fits, not production inference time or matched training FLOPs.

## Native soft answers: strict all four correct

Each cell is binding / exact date / latest mention, each out of 256. Both main panels use unseen key combinations; the novel panel also recombines familiar modifiers. The same-content rewrite is diagnostic only.

| Seed | Arm | Familiar syntax | Same-content rewrite | Novel combinations |
|---|---|---|---|---|
| 5371 | Answer only | 8 / 40 / 159 | 5 / 28 / 155 | 5 / 35 / 156 |
| 5371 | Support labels | 190 / 191 / 228 | 197 / 187 / 228 | 208 / 204 / 238 |
| 5471 | Answer only | 54 / 100 / 174 | 59 / 85 / 171 | 15 / 101 / 158 |
| 5471 | Support labels | 215 / 232 / 243 | 220 / 233 / 242 | 168 / 227 / 229 |
| 5571 | Answer only | 89 / 196 / 193 | 95 / 199 / 196 | 144 / 201 / 210 |
| 5571 | Support labels | 178 / 221 / 231 | 185 / 218 / 234 | 158 / 227 / 231 |

Support labels improved all 18 matched main-held cells by +14 to +203 families, from **2,038/4,608 (44.2%) to 3,819/4,608 (82.9%)** pooled model–family evaluations. These are descriptive counts across repeated seeds and related examples, not 4,608 independent families. Binding improved from 315/1,536 to 1,117/1,536 (20.5% to 72.7%), exact date from 673 to 1,302 (43.8% to 84.8%), latest from 1,050 to 1,400 (68.4% to 91.1%). No supervised binding cell met the 231/256 strict floor (best 215/256). Latest has an engineered recency cue; it does not establish general update reasoning.

No fit or arm is promoted. Fixed visible punctuation and chronology remain matched priors, not learned boundaries. This is synthetic, finite vocabulary, one hop, no full-cost inference comparison. The private replay contains checkpoints and the exact prompt/truth custody; retained actor logits, weights and diagnostic vectors are separate private wire bundles.

## Independent recount and mechanism check

The model-free reviewer reconstructed the tasks from literal prompts and matched all 18 held summaries and six qualification flags. For the main held panels, changing the training objective rescued 1,966 previously incomplete soft-answer model–family evaluations and broke 185 previously complete ones (net +1,781). Strict unique selection improved from 2,122/4,608 to 3,812/4,608, and correct unique selection from 12,154/18,432 to 16,699/18,432 rows. No run passed the full support-specific gate.

For the support-supervised fits, switching the frozen actor to its own top-one clause moved 3,819 to 3,855/4,608: 54 rescues, 18 losses. This was label-free but insufficient for qualification. Forcing the correct clause into the same frozen decoder gave 4,608/4,608 for support-supervised fits and 4,603/4,608 for answer-only fits; these gold runs are diagnostics and cannot qualify the model.

| Support-arm row cross-tab | Wrong support / wrong answer | Wrong support / right answer | Right support / wrong answer | Right support / right answer |
|---|---:|---:|---:|---:|
| Soft | 1,555 | 178 | 143 | 16,556 |
| Own top-one | 1,607 | 126 | 0 | 16,699 |

On these rows, the frozen hard readout never missed after a correct top-one choice. A wrong clause could still give the right value. Gold-hard used the same stored value vectors and query, replaced only the selected value, and exactly matched own-hard where the actor selected the correct clause. This supports selection as the residual bottleneck on this task, not proof of general compositional decoding.

**Decision:** Keep training-only support supervision as a useful control, not a solved binder. The largest remaining gap is choosing the right entity–relation pair amid confusable alternatives. Next, test a binder on fresh matched alternatives that share an entity or relation but differ in the other field, with a typed **query-independent** record-set intervention and matched exact versus learned readout. Keep oracle fields and gold support out of normal inference; leave longer chains, learned boundaries, TimeQA transfer and matched-quality cost pending.

Retained primary summary SHA-256: `6d70c5b07d665f128b6d2beaf638207c9f9265a6978bf553431402e59b741bb3`. Independent audit with gold replay SHA-256: `c4090e61c60fac84872fea771e09db8725be51e9e801c5a5f6a1b728bee6d004`.
