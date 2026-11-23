# COG-BIND-001Y: learned subclause roles with frozen temporal selection

**Isolated item binding:** outcome-only did not pass across all three seeds; role-supervised did not pass across all three seeds. **Native end-to-end:** outcome-only did not pass across all three seeds; role-supervised did not pass across all three seeds. The extra role-label contribution did not clear the prospective gain-and-qualification gate.

## Why this comparison
Earlier [001O](https://github.com/FPC-effortless/cdl-attention-experiment/blob/c98b53818d5c5374595ed060f49e7b219acd7c8d/research/nanocog-001/cog-bind-001o/RESULT.md) established an explicit learned alias-to-record composition path on controlled held-out word order. [001X](https://github.com/FPC-effortless/cdl-attention-experiment/blob/37c1d8b3096a3ae364d92f302d5173382ae4c9f3/research/nanocog-001/cog-bind-001x/RESULT.md) showed that reading a requested noun remained difficult even when a scorer supplied the correct temporal event. 001Y moves the explicit intermediate to the two subclauses inside an event, and changes the requested role on the exact same context. Both arms share the new architecture and the same observed conjunction boundary. The arm-level difference is the auxiliary role loss. X and Y change several things at once, so their scores cannot isolate the boundary effect.

The native predictor receives only the original W-format temporal passage and query, raw continuation text, and a raw alpha/beta request. Temporal selection uses the frozen W routing inputs, not the continuation. Reader outputs are computed for both events, not given the true one. The reader has two visible phrase regions but receives no reference role, span, noun shortlist, event index, or output value. It scores all live tokens. The arms share parameter count, initialization, answer targets, and schedules. The treatment alone adds explicitly counted four TRAIN segment role labels per update row. Correct-event and restricted correct-subclause scores are scorer-only diagnostics.

## Held design and thresholds
Each view has 128 independent synthetic root families, 32 cases per root, and 2,048 role-switch pairs; the four views cross novel reader and novel temporal renderers on the same roots. A complete role-switch pair requires both different role requests on identical text correct. Role × noun-swap quartets also require swapping the nouns while preserving the token bag. Novel refers to combinations of known fragments, not arbitrary natural language or unknown role vocabulary. Positional and nearest-role rules do not pass this benchmark; the finite grammar witness does. Inference on new prose and discovery of subclause boundaries are not tested.

Both-novel gates: at least 95% rows, 90% role pairs, 85% role × noun quartets, and no more than two points familiar exact-row loss. The auxiliary objective also needs +5 points in strict isolated role-pair accuracy in every seed to establish the specific claimed benefit. Results are reported separately from the frozen-route ceiling.

## Strict role-switch pairs

| View | Seed | Isolated outcome | Isolated role-aux | Native outcome | Native role-aux | Native route ceiling |
|---|---:|---:|---:|---:|---:|---:|
| held-familiar | 20371 | 0/2,048 | 1,968/2,048 | 0/2,048 | 1,968/2,048 | 2,048/2,048 |
| held-familiar | 20471 | 248/2,048 | 504/2,048 | 248/2,048 | 504/2,048 | 2,048/2,048 |
| held-familiar | 20571 | 0/2,048 | 1,864/2,048 | 0/2,048 | 1,576/2,048 | 1,672/2,048 |
| held-route-only | 20371 | 0/2,048 | 1,968/2,048 | 0/2,048 | 1,968/2,048 | 2,048/2,048 |
| held-route-only | 20471 | 248/2,048 | 504/2,048 | 248/2,048 | 504/2,048 | 2,048/2,048 |
| held-route-only | 20571 | 0/2,048 | 1,864/2,048 | 0/2,048 | 1,700/2,048 | 1,864/2,048 |
| held-reader-only | 20371 | 0/2,048 | 1,468/2,048 | 0/2,048 | 1,468/2,048 | 2,048/2,048 |
| held-reader-only | 20471 | 0/2,048 | 0/2,048 | 0/2,048 | 0/2,048 | 2,048/2,048 |
| held-reader-only | 20571 | 0/2,048 | 1,052/2,048 | 0/2,048 | 862/2,048 | 1,672/2,048 |
| held-novel | 20371 | 0/2,048 | 1,468/2,048 | 0/2,048 | 1,468/2,048 | 2,048/2,048 |
| held-novel | 20471 | 0/2,048 | 0/2,048 | 0/2,048 | 0/2,048 | 2,048/2,048 |
| held-novel | 20571 | 0/2,048 | 1,052/2,048 | 0/2,048 | 960/2,048 | 1,864/2,048 |

## Both-novel strict scores

| Seed | Outcome isolated rows, role pairs, quartets | Role-aux isolated rows, role pairs, quartets | Role-aux native rows, role pairs, quartets | Restricted subclause copy (outcome → aux), role pairs | Auxiliary role-pair change (root bootstrap 95% interval) |
|---|---|---|---|---|---|
| 20371 | 2,048/4,096 · 0/2,048 · 0/1,024 | 3,460/4,096 · 1,468/2,048 · 592/1,024 | 3,460/4,096 · 1,468/2,048 · 592/1,024 | 1,640/2,048 → 1,504/2,048 | +71.68 pp [+68.36, +75.20] (1468 wins, 0 losses) |
| 20471 | 80/4,096 · 0/2,048 · 0/1,024 | 308/4,096 · 0/2,048 · 0/1,024 | 308/4,096 · 0/2,048 · 0/1,024 | 0/2,048 → 0/2,048 | +0.00 pp [+0.00, +0.00] (0 wins, 0 losses) |
| 20571 | 1,996/4,096 · 0/2,048 · 0/1,024 | 3,100/4,096 · 1,052/2,048 · 512/1,024 | 2,824/4,096 · 960/2,048 · 468/1,024 | 1,156/2,048 → 1,052/2,048 | +51.37 pp [+50.39, +52.34] (1052 wins, 0 losses) |

## Exact fit exposure

| Seed | Arm | Parameters | Optimizer steps | Item targets | Segment role labels | TRAIN strict role pairs |
|---|---|---:|---:|---:|---:|---:|
| 20371 | outcome | 15379 | 384 | 49152 | 0 | 0/4,096 |
| 20371 | role-aux | 15379 | 384 | 49152 | 196608 | 4,096/4,096 |
| 20471 | outcome | 15379 | 384 | 49152 | 0 | 4,096/4,096 |
| 20471 | role-aux | 15379 | 384 | 49152 | 196608 | 4,096/4,096 |
| 20571 | outcome | 15379 | 384 | 49152 | 0 | 0/4,096 |
| 20571 | role-aux | 15379 | 384 | 49152 | 196608 | 4,096/4,096 |

Native execution was checked against the original TrainLab run records, with six FIT and six blind PREDICT runs. The sealed results were recounted independently. The frozen temporal arrays are bitwise unchanged where the raw W input is unchanged. Forge audited the projected data boundaries; the private Research Lab graph preserves custody, not a general capability promotion. Data digest: `abdbc6147596dba393e3da2e5babe440f985612a5ed3837c9fcd3b71454e88ea`. No inference-speed, general-language, or full attention-replacement claim.

## Decision
Role supervision stabilized TRAIN competence: three of three treatment seeds were perfect on complete role switches, while only one of three outcome-only seeds was. On the both-novel held view, role supervision improved isolated role pairs from 0 to 1,468, 0 to 0, and 0 to 1,052 out of 2,048. No arm passed the full gate. This is a reproducible partial improvement on held combinations in two seeds, not a claim of uniform improvement or a qualification.

The strongest localization is that treatment predicted the requested subclause correctly on 4,048, 4,096, and 4,096 of 4,096 held rows while exact item selection with the correct event was only 3,460, 308, and 3,100. Restricting the scorer to the true subclause gave that reader 1,504, 0, and 1,052 complete pairs, nowhere near the threshold. In the worst seed, both temporal and role choices were perfect, yet even correct-subclause scoring could not recover a single pair. The first two runs had perfect temporal selection; the third froze it at a 3,728/4,096 native row ceiling. These facts separate further routing work from the remaining copy problem on this panel. They do not identify a unique cause of item failures.

Next, hold the role head and visible boundaries fixed. Compare the same learned noun pointer with and without feature mixing from the other subclause, with balanced local noun positions and matched supervision. A deterministic one-noun-per-subclause reader is a ceiling only, not learned performance. If the learned head clears that isolation test, retest on unfamiliar wording and clauses without a reliable visible separator before claiming wider transfer.

The model-free descriptive breakdown reinforces that conclusion without a new fit: in the 20471 auxiliary fit, restricting to the correct subclause recovered zero exact beta items in each of four form/position cells (512 rows per cell). In 20571, the same diagnostic got all first-subclause cells and second-subclause alpha cells perfect, but only 0/512 and 28/512 for second-subclause beta under the two tested forms. These are conditional diagnostics over the same held families, not independent samples or proof that any one feature caused failure.
