# COG-BIND-001J: ordered comparison helps directed selection; learned alias override fails

## Result
At matched capacity and exposure, ordered channel comparison improved strict original held-out families in every seed: pooled 7/16/0 to ordered 27/26/7, each out of 256. Across seeds, strict accuracy rose from **23/768 (3.0%) to 60/768 (7.8%)**. The paired family-bootstrap difference was +4.82 percentage points (95% descriptive interval +2.86 to +6.77; trained seeds held fixed). The 768 observations are repeats over 256 families, not independent cases. Neither arm passed a registered cell. **Changing the meaning of a visible alias produced zero correctly resolved pairs in either arm.**

## Matched design
The two 33,826-parameter models share order-aware token encoders, two learned summaries per segment, learned preliminary soft read over six visible clauses, and native learned value output from the selected record. Only the final candidate-affinity reduction changes: mean of all four query–record channel combinations versus mean of the matched diagonal. Both got the same initialization per seed, 1,024 updates, 65,536 training rows and 3,122,880 nonpadding tokens per fit. Training support and answer labels enter the loss; no gold role, alias, record or answer is passed into inference.

There were 512 training, 64 calibration, and 256 held families. The four transfer and four query forms and token atoms are shared, but exact unordered transfer pairs and families are disjoint between splits. This is a controlled fixture, not authentic-language or OOD-syntax evidence. Each normal family crosses two opposite worlds, two orders, two direction queries and direct versus alias reference. In 2,048 held reversal pairs, corresponding candidate token bags and queries are unchanged while the correct record and answer change. A deterministic bag comparator cannot solve both worlds; an alias-resolved bag control achieved 2,048/4,096 individual joint rows and the scripted directional parser 4,096/4,096. Neither is a learned model, and the learned pooled arm is position-aware, not a bag model.

## Held-out measurements
| Metric | Pooled | Ordered |
|---|---:|---:|
| Original joint rows, correct record and value | 4,492/12,288 | 5,514/12,288 |
| Original strict full families, all 16 rows right | 23/768 | 60/768 |
| Original answer from correct supplied record | 12,220/12,288 | 12,196/12,288 |
| Transposed-alias strict full families | 0/768 | 0/768 |
| Alias pairs right before AND after transposing declarations | 0/6,144 | 0/6,144 |
| Alias pairs with the same hard prediction despite changed correct result | 6,142/6,144 | 6,144/6,144 |
| Direct (non-alias) controls right before AND after | 2,246/6,144 | 2,778/6,144 |

In the alias intervention the query, transfer facts, clause order and per-clause transfer evidence are held fixed; only the two explicit alias declarations exchange their referents. Gold record and answer change for all alias questions and for none of the direct questions. Correct-slot value decoding still works across both views on 6,070/6,144 pooled and 6,062/6,144 ordered alias pairs. The bottleneck is choosing the record, not rendering its value. The ordinary alias results do not establish that the models resolve aliases: original training aliases always name the primary pivot, whereas the intervention points to the alternate one. The swapped-alias subset resolves to alternate edges that are not world-reversed, so the original bag impossibility does not apply to that subset.

## Same-weights intervention
For the three trained ordered models, replacing the final diagonal score with either the all-entry mean or opposite diagonal cut original strict families from 60/768 to 0/768. Per-slot value logits were bit-identical. This isolates a causal contribution of the rank rule at fixed weights on this fixture. It does not label either channel as a semantic role or identify the internal cause of alias insensitivity.

## Verification and decision
Six CPU fits and 24 locked prediction files covering 79,872 prompt rows were independently recounted. No attention, language-model, memory or full-cost efficiency gate was run. Keep the ordered selector as a directional control only. Next: train on both declaration assignments with identical question and transfer text and matched exposure, then evaluate on a fresh panel with disjoint alias–referent combinations to see whether a learned resolver uses the declarations and composes their meaning with direction. Do not scale the present alias path.

Data pin: `caec8b0839bbad34c491c754b28141269441f89df75ccf973de3e9aea24a77ea`. Scored result SHA-256: `abf480dff2d3fd7bcac8649f00fa3cbde20e8dda177fa18d3769710e0c42f474`.
