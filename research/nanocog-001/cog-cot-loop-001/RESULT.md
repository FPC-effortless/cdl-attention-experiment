# COG-COT-LOOP-001: causal communication backbone, generated traces, and tied computation

**The architecture can reuse the same learned layer, but the first trained model did not learn reliable graph reasoning.** The two-pass COG emitted a completely correct generated route and answer on only **2 of 324** fresh two- and four-hop cases, and **0 of 432** six- and eight-hop cases. More inference passes did not recover a single exact long trace. None of the 12 endpoints, including the Transformer controls, met the prespecified one-hop competence check. This is a failed pilot, not evidence of an inherent limit on either architecture.

## Architecture

One COG block is reused at each internal pass, and its state is an ordinary dense token-by-channel representation for the current sequence. It builds learned messages from five causal token offsets and from the immediately preceding completed blocks at scales 8, 32 and 128. Gated feature mixing delivers those messages; gated nonlinear composition then updates the representation. There is no trainable pointer/address table, record store, explicit solver, persistent state across calls, or EQUIVALENCE compiler. Only the Transformer control uses token-pair attention. The 128-token block cannot activate at the tested total sequence length (at most 50 tokens), so the effective long paths here are 8 and 32.

Eight fabricated software tests checked tied parameter reuse, changed logits after reusing the block, causal prefixes, and branch isolation. The inference pass count was externally set, not learned halting. At the first prediction position, the intact model already has a path from every prompt position with one pass. Additional passes are therefore not needed merely for first-output geometric reach in this fixture.

## Experimental contrast

Each example presents two shuffled six-node directed maps, a start node, and a visible plan. For a trace-trained model, the target is the visited nodes in order, then an answer marker, final node and end token. The answer-only arm never received the intermediate labels. At evaluation all arms generated freely from the raw 36-token prompt, feeding back their own choices on the full 13-token vocabulary, stopping on the first end token or a fixed 14-token cap. No answer, trace, output length or marker was forced.

Training used 1,080 rows at depths 1, 2 and 4. Across three fixed seeds, each arm used identical within-seed row draws, 448 AdamW updates and batches of 24. COG was 11,584 parameters and the tied causal Transformer 11,168. This matches width, answer draws and update count, not FLOPs or number of supervised tokens between trace and direct-answer arms. The same-depth evaluation used 144 rows per seed from new graphs. The longer-depth evaluation used a separate 144 rows per seed at depths 6 and 8. Graph topologies are disjoint even under node renaming. Relevant paired edits swap two destinations without changing the query, layout, or token counts and require a different final answer. Irrelevant edits leave the entire correct path unchanged.

## Results

Counts aggregate the same raw cases over three seeds and are not independent samples. A correct formatted final answer can have a wrong or incomplete intermediate trace. Exact trace requires every step in order, the correct final node, and the end token.

| Trained arm, native pass count | Correct final, one hop /108 | Correct final, familiar lengths /432 | Exact CoT, familiar lengths /432 | Correct final, held lengths /432 | Exact CoT, held lengths /432 |
|---|---:|---:|---:|---:|---:|
| COG, two tied passes, trace supervision | 20 | 94 | 22 | 56 | 0 |
| COG, separately trained one pass, trace supervision | 18 | 78 | 18 | 70 | 0 |
| Causal Transformer, two tied passes, trace supervision | 19 | 64 | 21 | 5 | 0 |
| COG, two tied passes, answer-only supervision | 25 | 76 | Not requested | 37 | Not requested |

Of the 22 exact familiar-length outputs from two-pass COG, 20 were for one hop, two were for two hops, and none were for four hops. No run got both relevant one-hop counterfactual worlds right on any paired layout. At higher familiar depths only two native pairs had both final answers correct across all arms (one in each traced COG arm); neither completed a whole root or matched both exact traces. No native arm got both relevant held-length answers right. At six and eight hops, the two-pass trace arm usually ended prematurely (398/432), and hit the generation cap without an end token in another 20/432. Learning the format or occasionally naming the right final node does not demonstrate reliable sequential reasoning.

On the **same trained two-pass COG trace checkpoints**, one, two and four passes gave **68 / 94 / 77** correctly formatted final answers at familiar lengths and **19 / 22 / 21** exact trace outputs, respectively. At unseen lengths they gave **43 / 56 / 68** correctly formatted final answers, but **zero** complete traces and zero relevant paired answers at each pass count. For independently trained direct-answer COG, the same one/two/four inference interventions gave **49 / 37 / 39** held-length correct answers, with zero relevant pairs. Thus extra passes alter outputs but there is no reliable looping advantage. The separately trained one-pass trace arm did better than two passes on held-length final answers and worse on familiar ones; none was competent enough for a general comparison.

Removing longer-span communication dropped familiar-length final answers from 94 to 72 and exact traces to zero. In this two-pass condition, the first generated token cannot see any graph record without the longer path, so this is a structural check, not evidence that the intact path learned the graph. Removing local communication dropped familiar final answers to 12 and exact traces to zero, but also removes access to some recent plan positions. Neither ablation isolates an equal-information alternative.

## Decision

Keep the architecture-first direction. The next meaningful gate is reliable **one-hop composition without a bank**: keep matched relevant/irrelevant edge swaps, qualify exposed and fresh layouts, and only then raise task depth and test whether tied computation improves causal changes in the complete trace. Use the Transformer to check that the task is learnable at the allotted budget. Do not interpret extra loop iterations as reasoning or bring in external memory to mask the failure.

All 12 fits and their 12 raw-only evaluations completed once on one CPU, 106.015 seconds summed native role elapsed, not a serving benchmark. Exact arithmetic recount from the sealed output and logit tapes matched the original score. No paid or protected run, natural language result, or inference efficiency claim.
