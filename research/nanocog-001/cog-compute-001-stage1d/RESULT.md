# COG-COMPUTE-001 Stage 1d: paired-query training loss

**Result: 0/12 qualified.** Adding a paired rival margin made both architectures change their answers more often when the question changed, but it did not reliably make them choose the correct fact. Neither model family earned promotion. The margin produced some complete-family gains alongside worse ordinary answer accuracy and degraded invariance.

## Locked comparison

Stage 1c receiver-conditioned COG and tied causal attention kept their exact mixer and forward code, width 48 and two passes through a single tied block. Each was fitted at seeds 173, 271 and 907 with ordinary next-token cross-entropy (CE) or with the same CE plus a bidirectional margin against the other question's true answer. With V17, COG had 25,844 parameters and tied attention 24,816, identical across objectives within each architecture. Both loss arms received the exact same initialized architecture, 1,000 AdamW steps, ordered minibatches (24 rows each), 24,000 answer draws and 72,000 ordinary token targets. Each batch has three pairs per case, or 12 pairs total, all sharing their own exact fact list and ordering. CE already saw both questions. The added hinge is computed before the gold answer can enter the model and applies only when answers differ: average of max(0, 1 − correct logit + other-question-answer logit) in both directions, at fixed weight 1. No training head, support coordinate, candidate mask, lookup, persistent memory or change to inference. Matched model passes and training draws, not exact training FLOPs.

The raw eight-token geometry is two local typed facts plus a question. The corpus expanded from Stage 1c to eight node and four relation symbols; ordinary and paired fits today saw identical data. TRAIN had 1,536 unique rows and OPEN 384 rows (96 families, 24 per case, crossed over both queries and both fact orders). OPEN families are disjoint from active TRAIN and the previous Stage 1c evaluation under registered relation-ID renaming, but 31/96 match that signature in older, nonreused Stage 1c TRAIN. New held relation symbols were present in current training. This is fresh labeled combinations on the same abstract templates, not token, syntax, or topology generalization, and not a controlled cross-stage delta. Free answers use native full-vocabulary greedy generation. Per seed and case the gate requires at least 87/96 exact rows and 22/24 complete four-world families. The public constant marker diagnostic never substitutes for free generation.

## Results

Complete families /24, ordinary CE → paired margin, split by seed and case:

| Architecture | Seed | Subject changes answer | Relation changes answer | Subject leaves answer unchanged | Relation leaves answer unchanged |
|---|---:|---:|---:|---:|---:|
| COG | 173 | 1→3 | 5→5 | 24→24 | 24→23 |
| COG | 271 | 0→1 | 4→5 | 24→20 | 24→19 |
| COG | 907 | 1→3 | 4→4 | 24→24 | 24→24 |
| Tied attention | 173 | 0→1 | 0→5 | 24→12 | 24→5 |
| Tied attention | 271 | 1→1 | 0→8 | 24→15 | 24→11 |
| Tied attention | 907 | 0→2 | 0→1 | 24→5 | 24→2 |

| Diagnostic, pooled descriptively across seeds | COG CE → paired | Attention CE → paired |
|---|---:|---:|
| Same emitted answer on answer-changing pairs /288 | 166→133 | 240→100 |
| Both answers right on answer-changing pairs /288 | 58→62 | 29→53 |
| Exact rows when answer should change /576 | 282→276 | 300→240 |
| Exact rows when answer should stay the same /576 | 576→557 | 576→388 |
| Node choice outside both facts on answer-changing rows /576 | 0→36 | 2→156 |

Pair and seed aggregates include reused test families across seeds and both orders, not independent sample sizes. All but five of 4,608 free outputs were correctly formatted; the fixed marker gives bit-exact answer-step logits and rescues zero answers. The tied-attention relation-switch gain is real on this locked panel, but appears with large losses elsewhere. Reject the tested objective as a general repair. The test supports an association between the margin and more unsupported choices; it does not prove a particular internal representation failure or rule out other objectives.

## Decision and verification

Keep the one-hop gate closed. Before more architecture or loss changes, compare existing endpoints on the exact training maps and fresh held maps using identical unassisted decoding, with a separate logit diagnostic for correct, other in-fact and outside-fact answers. If both facts can be selected in familiar worlds but not fresh ones, prioritize transferable representations; if training-world selection already fails, diagnose optimization and the computation itself. Do not count forced-choice diagnostics as native successes. No multi-hop, attention replacement, general language, or inference-economics claim.

One native invocation completed all 24 roles (12 fits and 12 open evaluations), 142.082 serial role seconds on one CPU core, not a serving benchmark. Independent raw score recount matched. The public report is aggregate; exact replay and newly created synthetic endpoints stay owner-only.
- DATA: `23b0cb688d9eb377a6345d933b30a8fc73033e87d8522d113c81b2f8aa397866`
- LOCK: `e16dd8cd60908b9bd2647ecde7008c24f705163bc290314f4b73c97535bb328c`
- WIRE: `cbd7d3d3c9fa53f167fd363247542c9894687816fac5d1762caa8f5041e0cba8`
- SCORE: `a570cdca8bb91edbdd70341a9224f8dfbd534672069851c6601911b5a189ffcf`
- Independent receipt: `8d8ce9b22ec8098b5db1c464be35d35c20945f30346245713bb6ce56c29f9962`
