# COG-BIND-001F: both models still fail binding under rewording
## Decision
No fit passed the full qualification. On fresh synthetic cases with reworded
facts, the small bidirectional Transformer was stronger in aggregate than the
capacity-matched pooled COG control. The same Transformer was weaker in
aggregate on familiar wording. Its unfamiliar-fact advantage is not consistent
across seeds; do not generalize this to Transformer LMs or an inference winner.

## Match and scope
Three fixed seed pairs. 29,352 learned scalars for the pooled model, 29,144
for SDPA (0.71% difference). Both use 1,024 AdamW updates, 81,920 identical
scheduled actor rows and exactly the same nonpadding tokens in each pair
(5,910,956; 5,907,916; 5,913,388), vocabulary, E-v2 varied wording, segmentation
and supervision. One-thread CPU fit, final endpoint, no CAL/HELD tuning.
1,280 CAL and 2,560 HELD rows per view; 128 fresh held semantic rectangles
(64 checkerboard, 64 distinct), with no old rectangle or exact prompt overlap.
All 143 individual held keys appeared in earlier studies, not in this TRAIN
split. Wording views cross fact and question edits on the same examples.

The control is one 32-wide, four-head pre-norm bidirectional SDPA+GELU FFN
encoder block over the full packed prompt; segment-relative positions and a
start marker preserve the supplied boundaries. It is not a causal LM.
The COG implementation is the exact B pooled clause encoder with dot-product
record ranking. Contextualization and internal encoder dimensions differ;
this tests the two fixed model packages, not a surgical test of attention
alone. Same six-record ranking and eight-way value output contract; same
TRAIN support + soft value losses. At evaluation the unique six-way native
rank chooses one slot, its hard value comes from that slot's saved logits,
and the exact-support diagnostic evaluates the same logits at the gold slot.
Gold-slot contextual Transformer logits can use other tokens in the prompt,
so this diagnostic does not prove that only the selected clause provides
the answer. Every target row in both counterfactual histories and both
presentations must pass for a strict family. Anchor counts are separate.

## Fresh held results
Each cell pools 384 model–family evaluations (128 repeated across 3 seeds).
Left/right are pooled COG / bidirectional Transformer.

| Wording | Strict native joint | Strict record selection | Strict correct-slot value |
| --- | ---: | ---: | ---: |
| Familiar facts, familiar queries | 315 / 296 | 315 / 296 | 384 / 384 |
| Familiar facts, shifted queries | 278 / 283 | 278 / 284 | 384 / 383 |
| Shifted facts, familiar queries | 41 / 128 | 197 / 223 | 77 / 219 |
| Shifted facts, shifted queries | 38 / 138 | 208 / 242 | 77 / 223 |

With both facts and queries shifted, this is 9.9% versus 35.9% strict joint.
The transformer gain is largely contingent on the seed: separate native
scores on that view were (15,13), (10,80), (13,45) out of 128.
Secondary all-four-view strict joint (same family succeeds in every view)
was 33/384 vs 109/384 (seed pairs 14/11, 9/63, 10/35). No seed cleared
the 95% target-row, 90% family and 95% anchor qualification across all
conditions and strata. No fit cleared both CAL strata. The fixed TRAIN
probe met the same thresholds on both observed TRAIN views for three
pooled fits and one SDPA fit. The SDPA baseline is not optimized beyond
this fixed recipe and nominal capacity.

| View / arm | Pass both | Miss only selection | Miss only value | Miss both |
| --- | ---: | ---: | ---: | ---: |
| Shifted facts, familiar query / pooled | 41 | 36 | 156 | 151 |
| Shifted facts, familiar query / SDPA | 128 | 91 | 95 | 70 |
| Both shifted / pooled | 38 | 39 | 170 | 137 |
| Both shifted / SDPA | 138 | 85 | 104 | 57 |

The shared output contract makes these exact strict family-level bins,
not independent failure estimates. Both models decode almost all families
when the facts are familiar and the correct slot is supplied. Fact rewrites
hurt the value readout even with the correct slot, particularly for COG.
Query-only changes primarily reduce source selection. This is evidence to
test stronger value invariance in COG, not to tune another retrieval score
alone.

## Verification and cost boundary
All six fits and 30 prediction wires (69,120 prompt rows) were sealed and
checked. The independent scorer recomputed exact ranks, answers, ties,
group joins and failure partitions. No hardware inference benchmark was
run. Fit receipts retain workload shape and a narrow forward-backbone
operation proxy; it excludes data transforms, shared heads, backprop and
other costs, and is not an inference performance claim. Compare complete
request costs only after competence is comparable. No claim on authentic
text, long contexts, plasticity, or general attention replacement.
