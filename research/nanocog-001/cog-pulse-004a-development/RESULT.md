# COG-PULSE-004A: Why a useful write can look unprofitable

The model-free benchmark separated two problems: **how the objective charges a write** and **what the writer can know before acting**. It ran on 288 constructed traces in 72 paired families across five horizons. It did not fit or load a model.

| Test | Observed result |
|---|---|
| Cumulative vs average utility at the **same effective write price** | Identical action rankings and ties across all 1,440 trace–horizon cases. |
| Average utility with the **same numerical penalty**, left unscaled | Changed the best action set on **144/288** cases at both H=16 and H=32; **0/288** at H=1, H=4 and H=8. |
| At H=32, can the prefix-only rule pick the right action in both continuations? | **0/72** with observationally identical blind prefixes; **72/72** if the same rule receives a supplied query-identity cue. |
| Same expected budget: half a write per case over two equally weighted worlds | The hindsight allocation advantage was positive in **60/60** full-bank families at H=32, averaging **0.40 constructed utility units**. Forcing one write in each world left no victim-selection advantage in those families. |

A useful write that avoids one wrong read gains 0.80. At 32 reads, charging 0.055 once yields **+0.745** cumulative utility, or **+0.02328** per read. Averaging the read gain and still charging 0.055 yields **−0.030**. That changes the decision because it changes the price. A scale-preserving rewrite of exactly the same objective is not a new learning mechanism.

The twin result shows a separate limit: if pending facts and occupied states are identical but the next useful query differs, the decision rule cannot condition on which continuation happens. The supplied clue reveals the future query identity by construction, not through learned routing. Hindsight is an upper diagnostic, not a deployable comparison; the half-write test matches expected spend, not spend in every world.

Forge found no cross-split prefix or family overlap in these open fixtures. TrainLab verified the engineering run, and all **2,880** independent fixture-reference numbers agreed. This is not evidence of learned identity, better accuracy than another model, or faster inference. The next scientific test must use text where the current stored fact changes a future observable target and compare learned prefix-only selectors at matched budgets with no cue supplied at inference. The immutable graph-cache provenance issue from PULSE-004 remains separate.

Trace result SHA-256: `427f229dedfcf7a394c85772d35a18b1287c028d574a846a0680202959fe4235`. Engineering receipt SHA-256: `1aecafac5a6df2b2eea69a8daddb67d3ff35f649da73018300e18783032ac63c`.
