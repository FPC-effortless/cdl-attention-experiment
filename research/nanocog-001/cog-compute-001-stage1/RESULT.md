# COG-COMPUTE-001: Stage 1 one-hop qualification

**Neither model qualified at roughly 100K parameters. The bottleneck was choosing the destination, not formatting or stopping.** COG scored 102/432 correct on fresh six-node shuffled maps, versus 84/432 for the small tied Transformer. The simple “last destination in the requested relation” rule scores 108/432 on these same rows. COG did respond correctly to more relevant paired edits (17 vs. 6 of 144), but completed just one of 72 repeated-seed graph roots against a prespecified requirement above 90% in every seed. No multi-hop or loop progression was started.

## What changed from the previous pilot

The architecture was unchanged except for increasing the hidden width from 32 to 96. COG uses the same learned causal local offsets, previous-completed-block messages, and nonlinear layer reused twice. Only the Transformer control uses token-pair attention; it also reuses one learned layer twice. There was no persistent memory, hand-coded inference solver, compiler, or earlier checkpoint. The 128-token message scale remains inactive because sequences are short.

Both models trained from scratch for 1,408 AdamW updates in batches of 24, across three seeds. They shared raw draw order, labels, hidden width, pass count, and positions. COG had 96,192 parameters and the Transformer 94,944. This is matched training exposure, not equal measured FLOPs. The fixed training pool contained 3,240 synthetic rows, mixed from the beginning rather than staged. Its four-node and six-node graph families had disjoint topology classes from the current evaluation. Many training graph classes appeared in the previous study, but none of the 72 six-node evaluation topologies did.

The 432 evaluation rows per seed cross sparse four-node and full six-node maps with near and shuffled placements. The counterpart uses the same graph, query, values, and correct answer. Near placement moves the relevant relation later and puts the target near the query among two plausible positions; it is an easier layout scaffold, not a pure change in distance. A relevant paired edit swaps two destination tokens to change the correct answer without changing the token bag. An unrelated swap leaves the answer unchanged. Rows share roots and layouts; repeated seeds are not new independent graphs.

The primary mode generated freely over the full vocabulary and had to emit the answer marker, the correct node and EOS with no forced tokens. A separate diagnostic supplied only the constant answer marker and scored the next token, never the correct node. To qualify, both exact row accuracy and complete relevant-edit roots across both layouts had to exceed 90%, per seed, on sparse-near and full-shuffled panes.

## Open results

Counts are in seeds 173 / 271 / 907. The semantic and formatting diagnostics agreed on all correctness counts.

| Pane | Exact free answers, COG | Exact free answers, Transformer | Complete relevant roots, COG | Complete relevant roots, Transformer |
|---|---:|---:|---:|---:|
| Near, 4 nodes | 40/39/44 of 72 | 30/30/35 of 72 | 3/1/0 of 12 | 1/1/0 of 12 |
| Shuffled, 4 nodes | 26/27/29 of 72 | 25/27/24 of 72 | 0/0/0 of 12 | 1/0/0 of 12 |
| Near, 6 nodes | 58/64/62 of 144 | 50/52/42 of 144 | 0/2/3 of 24 | 0/0/0 of 24 |
| Shuffled, 6 nodes | 33/38/31 of 144 | 24/31/29 of 144 | 0/0/1 of 24 | 0/0/0 of 24 |

Of 2,592 free evaluations across both models, **2,589 used the correct output format**. Supplying the marker recovered **zero** extra correct answers. Where the model emitted that marker itself (2,590 rows), its answer-choice logits were bit-identical to the diagnostic. Therefore the short answer format was mostly learned; reliable edge selection was not.

COG got both answers right on **17/144** relevant six-node shuffled pairs versus **6/144** for the Transformer. COG still emitted the same node on **93/144** of those pairs, although the correct node changed (Transformer: 103/144). Whole-root results were weaker than single rows. Across matched six-node layouts COG scored **184/432** with the edge nearby and **102/432** shuffled; Transformer scored **144/432** and **84/432**. Last-position copying scores **216/432** near and **108/432** shuffled, but cannot pass the near whole-root gate. These are diagnostics of locality and fragile binding, not a quality win over a competent control.

## Decision

Do not add loop depth or persistent memory. The evidence argues for a more targeted one-hop learning test: hold the graph and destination tokens fixed, change the queried subject or relation, and check whether the prediction follows the correct source–relation combination. Compare direct answer supervision with a separately labeled training-only signal for identifying that combination, without exposing the answer location at inference. The present run changed size, data and budget relative to the prior pilot; it does not isolate scaling alone. Neither a general architectural failure nor general superiority follows.

All six fits and six raw-only evaluations ran once on one CPU (241.082 seconds of summed native role time, not a serving benchmark). The score was independently recounted from the sealed outputs. No paid/protected execution, real-language result, or inference efficiency claim.
