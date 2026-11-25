# COG-COMPUTE-001 Stage 1m: Raising the sender-weight ceiling improves binding but does not qualify

**At identical size, data and training budget, increasing the fixed logit scale improved counterfactual binding under both COG routing geometries.** The strongest arm, shared-weight COG at scale four, got both assignments right in **870/1,152** coherent swap comparisons, up from **127/1,152** at scale one. It completed **194/288** eight-example families versus **2/288**. All three seeds still failed the full held gate. Assigning different values to different subjects of the same relation remained the main failure.

## Frozen matched comparison

Four fresh width-64 COG variants use the same raw 12-token input and 31-symbol vocabulary, sender and receiver representations, strictly prior contiguous trigram candidates, value projection, causal mask and decoder. Channelwise routing applies a separate sender distribution per latent channel; shared routing pools the score across channels before applying one sender distribution. Within each geometry the only treatment is multiplying logits by a fixed positive scale, one or four, before the same unit-mass softmax. The scale-one paths are exact Stage 1l code. No learned temperature, support annotations, supplied fact boundaries, prior checkpoints, answer masks, external memory, looping, or extra context. All four have 67,847 nominal parameters including the same 4,160 sender-constant bias entries that cancel under softmax.

Seeds 173, 271 and 907 share identical initial tensors and row draws within seed, AdamW, full-vocabulary supervised answer/format cross entropy and the same fixed 4,000 × 48 training exposures. Raw-only evaluation includes 4,096 familiar, 768 novel-map primary and 384 value-swapped prompts. The two held layouts use the same maps, both query orientations and both physical orders. Swaps replace exactly the two value tokens and both assignments are individually supported in FIT in both ranks. Exact held maps are disjoint from current FIT and from the tracked prior map union, with no key-pair overlap against current FIT or the Stage 1l held set. Reused older relation key skeletons are disclosed: H 4/24, I 6/24, J 6/24, K 10/24 (zero subject), counted separately by prior panel; these counts are not independent. An initial wall-clock-limited data allocator did not replay FIT byte-for-byte and was discarded before any fit. The seeded single-thread fixed-node replacement replayed byte-identically under normal and optimized Python; 26 Forge checks completed with zero findings.

## Exact free-generation outcomes

Full exact output requires the model to emit the answer marker, correct node and EOS on its own. Familiar floors are strict >95% exact rows and complete families in each case; the known and unseen held layouts must each exceed 90% on the same per-case measures, per seed (at least 87/96 rows and 22/24 families). Aggregates below cover changed-answer cases, with repetition across seeds and both layouts. They are not independent semantic units, natural-language results, or a comparison against attention in this experiment.

| Outcome | Channel ×1 | Channel ×4 | Shared ×1 | Shared ×4 |
|---|---:|---:|---:|---:|
| Familiar pass (seeds) | 3/3 | 3/3 | 3/3 | 3/3 |
| Full held pass (seeds) | 0/3 | 0/3 | 0/3 | 0/3 |
| Changed exact rows, known / 576 | 321 | 398 | 303 | 495 |
| Changed exact rows, unseen / 576 | 308 | 408 | 297 | 493 |
| Complete changed four-example families, known / 144 | 39 | 63 | 39 | 112 |
| Complete changed four-example families, unseen / 144 | 34 | 65 | 39 | 109 |
| Both value assignments exact / 1,152 | 158 | 540 | 127 | 870 |
| Same predicted node despite changed target / 1,152 | 917 | 579 | 948 | 267 |
| All eight contexts exact / 288 | 1 | 61 | 2 | 194 |

Scale four rescued 236 changed rows and lost 59 under channelwise routing; it rescued 427 and lost 39 under shared routing. Corresponding complete-family rescues/losses were 64/9 and 147/4, respectively. The scale-effect gain was +55 versus +143 complete families, a descriptive +88-family difference in this matched comparison. At ×4 the shared arm passed all six relation-change seed/layout cells, with 141/144 complete relation-change families summed over them. For subject changes it completed **80/144** and no cell reached 22/24: 4/3, 19/19 and 18/17 for known/unseen by seed. The stronger relation result cannot substitute for the failed subject gate.

## What the route actually did

With ten valid spans, a single coefficient can be at most exp(2s)/(exp(2s)+9) for scale s because the centered COG scores are bounded by tanh products. This upper bound rises from 0.451 at ×1 to 0.997 at ×4. It is not a predicted realized coefficient. Positive scaling preserves score rank at a fixed parameter state; it need not do so across separately trained states, and averaging different channel allocations is not the same as ranking each channel.

At the public answer-marker site, mean normalized entropy (1 is uniform) went 0.948→0.804 for channelwise and 0.874→0.593 for shared; mean per-channel maximum weight went 0.207→0.323 and 0.258→0.525. The intact target span was uniquely first by mean actual weight on 557→715 (channelwise) and 515→749 (shared) of the 1,152 changed rows. In the strongest shared arm, free answers were correct on **708/749** rows where that span ranked first, and **280/403** where it did not. This is a useful diagnostic, not proof that a single span determined each answer. Correct answers still occur with other spans ranked first, and some high-ranking target spans still do not yield correct answers. No gold tags entered the actor; target identity and margins were joined only after custody release. The public marker diagnostic did not change native decoding.

**Decision:** preserve the ×4 shared configuration as a more capable research reference, not a qualified replacement. Before any broader fit, isolate the remaining subject-binding failure with a frozen comparison of correct, rival and irrelevant span interventions, separating the effect of the read rule at fixed weights from differences acquired during training. No runtime advantage, long-context competence, or comparison against a quality-matched attention model is established here.

## Audit

24 successful original native roles, 359.980406 summed role seconds on one CPU, no retries. Detached Veritas evidence and independent raw recount agree on exact rows, families, swaps, diagnostics and ranks. DATA `3f933977e46d44c2dd1efe4f70870c2f530380f1eaea71eb1d373481f55ceca3`; LOCK `f0b74dedf25826eecb16725e857077bdcd7757751aedbb4c16ea21fedbd1db20`; WIRE `45ccda04a8738638e956223637a08252996065f1d4621118e1d90fa8de10a333`; SCORE `831798f17a723dab302b2080456f8c5a772b440b1bcec4e3fa1799950acfe9ca`; recount `e5bbd45328f286d66ec4b72ed2acfa7566e194f9dfeca87b6e626413273b318e`. [Stage 1l](https://github.com/FPC-effortless/cdl-attention-experiment/blob/60c75a748bc06693bb9c09f08ff56a207d187e36/research/nanocog-001/cog-compute-001-stage1l/RESULT.md).
