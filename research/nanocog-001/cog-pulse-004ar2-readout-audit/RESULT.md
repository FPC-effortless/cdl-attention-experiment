# COG-PULSE-004A-R2: the correct record needed a stronger read, and that still wasn't enough

**Putting only the correct record into the unchanged answer head at full strength helped all three seeds. Removing the competing records without strengthening the correct one did not consistently help.** Even with this best-case intervention, the reader missed many answers. This separates an observed signal-strength problem from a remaining payload or decoder problem.

## Three occupied records, fixed placement

Each seed reuses the same **768 exposed constructed cases from 24 independent roots**. The table counts exact full-vocabulary answers. The original current, obsolete and unrelated record states were not re-encoded or retrained.

| Frozen answer-head input | Seed 173 | Seed 271 | Seed 907 |
|---|---:|---:|---:|
| Original native mixture | 398 | 422 | 290 |
| Keep only the correct contribution, at its original weight | 396 | 452 | 308 |
| Keep only the correct contribution, at full unit weight | **608** | **616** | **464** |
| Remove only the obsolete contribution | 396 | 448 | 308 |
| Remove only the unrelated contribution | 398 | 430 | 292 |
| Remove the correct contribution; leave both others at their old weights | 0 | 2 | 4 |
| Remove the entire read vector | 0 | 0 | 0 |

At the same current weight, removing both competing contributions rescued **40 / 36 / 36** cases but harmed **42 / 6 / 18**. Removing either competitor alone also had seed-dependent effects. The aggregate does not support a rule that all competing evidence is harmful. Once only the correct contribution remained, restoring it to full strength rescued **216 / 164 / 156** additional cases and harmed **4 / 0 / 0**. This is an amplitude intervention, not a new learned policy.

Full strength raised complete three-record families from **2 / 2 / 2** to **9 / 11 / 3 of 24**, leaving **160 / 152 / 304** wrong answers. On singletons with no competing factual vector, full strength got **76 / 78 / 50 of 96** right, versus **76 / 72 / 50** natively. Removing routing ambiguity does not by itself solve the remaining encoding, query or answer-head errors.

## Why the trace mattered

Before intervention, the sealed trace showed **274 / 278 / 346 of 768** cases with the correct record uniquely top-ranked but the wrong answer. Even among cases where its native weight was at least 0.90, the model missed **72/272, 92/284 and 84/222**. The competing vector was still present in those native rows, so the threshold alone could not show which part caused the miss. The frozen cut and full-strength controls above answer that separately.

For occupied banks with no matching owner, native routing gave null top weight on **0 / 38 / 0** rows while natively returning NONE correctly on **84 / 92 / 82 of 192**. Supplying an all-zero read externally produced **192 / 192 / 184** correct NONE responses. That shows what the head can do when evidence is reliably withheld, not that the model learned when to do it. The same zero read produced no complete factual families.

## What to test next

First qualify a strength-stable answer path using fresh singleton and same-record/irrelevant-record pairs before training a more elaborate router. Compare the original head with a scale-stabilized reader under equal joint answer-loss supervision and the same data and budget. Keep correct-record-at-unit as an external ceiling, not a serving rule. Gate occupied no-match rejection separately using same-bank present/absent pairs; don't equate a larger null weight with correctness. This study does not establish a learned repair, general chronology, inference speed or an attention replacement.

The three frozen R2 content+null checkpoints were used once. Only the original four readout tensors were used for computation on already saved query, payload, route and weight arrays. Original native and zero-read logits reconstructed with **zero numeric drift** at all eight original placements; causal variants were then evaluated at one fixed placement. Independent model-free readback matched the full score. No fitting, protected cases, encoder replay or publication of raw tensors or checkpoints.

Trace joint SHA-256: `a5959ca0c75c22ab5cf3eb16c5cb50feb3379b5565f1808d9571af2cf43dd53d`. Head WIRE: `147f0ccf75cd2e9fdaf87dde7d1dd09846f83c533fed6fb9c260d4ad0fc12552`. Head score: `e267fe4bc4d3e144e20aa30d53abeb3557dde9d1ff9dfe7edd97f112a35a907b`. Independent head recount: `810c395495d477ab6cb985fba9fad1bbbf27058e4d2a2fa82ebe61b219772061`.
