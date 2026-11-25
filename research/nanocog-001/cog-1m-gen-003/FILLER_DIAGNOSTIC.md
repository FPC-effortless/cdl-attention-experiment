# GEN-003 post-score audit: false refusals and wrong-subject answers

This is a separate replay of outputs from the four frozen runs in [the primary report](./RESULT.md), not a new fit or model run. Within each aligned same-width contrast the correct fact, its position, and the query did not change; only the other clause was replaced. The `saw` renderer includes a cue already seen in prior absence training and also changes other words, so this is not a single-token attribution.

| Output under the supplied-evidence and absence conditions | GEN-002 seed 20371 | GEN-002 seed 20471 |
|---|---:|---:|
| False `none.`, correct fact present, natural clause | 682/2,048 | 708/2,048 |
| False `none.`, identical correct fact present, `saw` clause | 1,740/2,048 | 1,724/2,048 |
| Item word emitted, only other subject's fact present, natural clause | 1,232/2,048 | 1,278/2,048 |
| Item word emitted, requested subject absent | 632/1,024 | 657/1,024 |

Every GEN-002 miss after natural same-width evidence selection was a literal `none.`, not another item word. In contrast, the model often generated an item when the text included only an unrelated fact. This supports reporting false refusals and false answers separately; one absence score would obscure the trade-off. The 128 source families are the independent grouping unit. Identical serialized prompts within the intervention panels never disagreed on emitted text and stop reason. No learned grounding, architectural ranking, or production abstention claim follows.

Private saved-output audit SHA-256: `9374234798325bdd68430b401267300a4e0af08bba95b2812fec980720171035`.
