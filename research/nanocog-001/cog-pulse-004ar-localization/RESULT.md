# COG-PULSE-004A-R · Frozen reader localization

**The reader often answers correctly with the current record alone, but its native read breaks when that record shares the bank.** This localizes much of the open pilot's failure to how the reader weights and mixes packets. It does not demonstrate a learned repair or rule out remaining limitations in the encoder or answer head.

| Read condition | Seed 173 | Seed 271 | Seed 907 |
|---|---:|---:|---:|
| Current record alone | 1,184 | 1,312 | 1,504 |
| Current + unrelated, native | 726 | 734 | 802 |
| Current + obsolete, native | 352 | 596 | 524 |
| Correct contribution at its native weight, competitor removed from the read | 644 | 804 | 732 |
| Correct contribution at full weight, either two-record bank | 1,184 | 1,312 | 1,504 |
| Native memory disabled | 0 | 32 | 0 |

Each cell is exact outputs out of 1,536 ordered placements (192 previously exposed open queries × eight placements) per frozen seed. Placements are repeated interventions, not independent samples. With only one record the eight placements reproduce 148/192, 164/192 and 188/192 unique query results. The singleton and forced correct contribution match exactly at the logit boundary. Complete roots correct in all placements were 14/24, 17/24 and 23/24 for singleton or forced correct contribution, versus zero on both native two-record banks across all seeds.

## What failed

1. **Physical slot preference is not content selection.** Scores compare the encoded query to learned physical-slot anchors, not to the stored record. With the query and occupied slots held fixed, replacing the unrelated record with the obsolete record did not change a single slot weight. Across the exact two-way swaps, the correct record had strictly greater weight in 768/1,536 placements in each seed (no ties). The stored contents still influence the mixed vector, so the 50% bound on correct higher-weight slot selection is not a bound on decoded accuracy.
2. **The obsolete record interferes, and removing it is not enough.** Holding both physical slots and the correct record's weight constant, cutting the obsolete contribution raised correct outputs from 352→644, 596→804 and 524→732. Then restoring just the correct contribution to unit weight raised those counts to the singleton results. Both interventions use the same native answer head. Cutting the unrelated contribution helped one seed and hurt two; do not treat every competing contribution as uniformly harmful.
3. **The reader is not fully qualified even with ideal selection.** On the native obsolete bank, 792/496/808 outputs were no-evidence rather than direct stale answers (96/256/140 stale tokens). Ideal selection still missed 10/7/1 complete roots. The foreign-vs-obsolete replacement is not a pure time intervention: the raw records also differ in identity and value.

## Scope and next test

The three exact successful PULSE-004A-R pilot endpoints were used once each, frozen. The external harness placed already-observed raw records in four disjoint pairs of physical slots, testing both directions for each pair. Records and query were encoded by the original encoder. No answer-specific vocabulary restriction, optimizer updates, previously sealed tests, new data, learned writer test, runtime comparison, or architecture promotion. The full paths include the original learned channel payload and the unchanged end-to-end answer head; the supplied-correct path is a diagnostic intervention only.

Test bounded query-to-record compatibility before mixing, preserving the same learner and bank size. Use a no-match option and test revision order only after selecting a matching record. Train the path jointly and compare with the original weighting and forced-selection ceiling on fresh held-out cases when separately admitted. The frozen diagnostic supports this design choice, not a performance prediction.

Source lock SHA-256: `668bbf20308300160b79328e0177f84b8a6108e1e51b045badb2fbfb3783625d`. Immutable open outputs manifest: `ea250ba36deaab5ed55ccc91ce660407d47bac2230bf16aec2278ec30b3f5887`. Original score: `179f3fe65ffa9a37f3d6aab865dc434391c5ecf5bed3ea2ed875d13c17ff5ddb`. The raw activations, endpoint bytes and case-level outputs are not published here.

Independent first recount: all 69,120 full-vocabulary outputs reproduced the official score and placement comparisons, without rerunning the model. Recount receipt: `9021ce91b959c48927af42e6ff3d51955fcbf13e002a9ff38103b779092abd12`.
