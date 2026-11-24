# COG-PULSE-004A-R2: top-ranked record at full strength

**Giving the top-ranked record full strength improved factual answers but made no-match handling worse.** The tied-value head rose from 200/210/126 to 268/284/172 exact answers out of 480 when the bank contained current, obsolete and unrelated facts. On 120 fully occupied banks with no matching identity, correct NONE responses fell from 56/58/48 to 24/12/20. The native-shaped head showed the same tradeoff.

This is a retrospective replay of the six sealed R2 answer-head tapes. For each case, the existing router alone picked the uniquely highest-weight occupied record or NULL. A record winner used the already logged full-strength logits for its slot. A NULL winner used the learned head's logged zero-read logits, not a forced NONE. Both used all 48 outputs, with ties counted wrong. The correct-record comparison below is diagnostic only; it uses the true current slot on factual cases. No model, weights, fitting or threshold search was needed.

| Exact outcome (seeds 173 / 271 / 907) | Native-shaped: mixture → top at unit | Tied-value: mixture → top at unit |
|---|---:|---:|
| Mixed-bank factual answers / 480 | 174/186/114 → 248/284/164 | 200/210/126 → 268/284/172 |
| Complete mixed-bank roots / 15 | 0/2/0 → 1/4/1 | 0/2/0 → 1/4/1 |
| Both answers after a current-value change / 240 | 38/62/18 → 76/112/38 | 48/68/24 → 96/112/46 |
| Correct NONE on an occupied no-match bank / 120 | 58/52/52 → 22/12/20 | 56/58/48 → 24/12/20 |
| Both correct for factual and no-match queries over the same bank / 120 | 14/24/6 → 12/8/12 | 16/26/10 → 10/4/12 |

For the tied head, the change rescued 74/82/64 factual cases but harmed 6/8/18. On no-match cases it rescued none and harmed 32/46/28. The router selected NULL only 0/4/0 times among the 120 actual no-match cases, and had no ties. Correct packet selection was itself imperfect: the current record ranked first on 320/364/292 of the 480 mixed-bank factual cases. Giving the head the correct packet at full strength on every factual case increased the tied head to 416/376/304 exact answers and 7/6/3 complete roots, so even perfect selection would not solve the benchmark. All panels reuse 15 constructed roots, not independent tests on every row.

**Decision:** Do not adopt hard top-at-unit as the default. It amplifies whichever record wins, including an obsolete or unrelated one. The next clean test is to learn whether any record is applicable independently of which record ranks first. Fit that decision on TRAIN only with the same frozen features, and evaluate paired same-bank hit/no-match transfer on fresh roots without choosing thresholds on CAL. Keep this open constructed result separate from any deployment or natural-language claim.
