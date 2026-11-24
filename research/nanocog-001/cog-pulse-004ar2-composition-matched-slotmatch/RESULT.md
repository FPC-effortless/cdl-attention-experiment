# COG-PULSE-004A-R2: matching one record at a time

**The slotwise applicability gate did not qualify.** With repeated-owner and three-distinct-owner banks represented in both splits, the learned query–record interaction beat fitted maximum cosine on all three distinct-owner runs, but it completed **zero** distinct-owner roots. On repeated-owner banks it lost to cosine in two seeds. More importantly, changing only timestamps could flip its hit/no-match decision even when the presence of a matching owner did not change.

### What was compared

We used 768 new constructed TRAIN cases and 384 separate open cases, with matching proportions of fully occupied repeated-owner and distinct-owner banks. Each pair used the exact same bank for a factual query and a true no-match query, with matched query-token frequencies. The three original R2 encoders, routers and tied full-48 answer heads stayed frozen.

Both estimators received the same binary bank labels, optimizer budget and fixed 0.5 decision cutoff. The control fit the maximum native cosine. The candidate scored each record with its own shared, centered query–record interaction before taking the max, without any labeled target slot. For primary full answers, a positive gate used the unchanged router's top occupied record at unit strength; a negative used the frozen head's zero-read logits, never a forced NONE. The candidate's own top-record choices were audited separately, not used in the answer policy.

### Open results

Counts are in seed order **173 / 271 / 907**. Each composition has **96 exact hit/no-match pairs across 12 roots**.

| Exact result | Repeated owner | Three distinct owners |
|---|---:|---:|
| Both gate decisions, fitted max cosine | 60/68/36 | 24/42/24 |
| Both gate decisions, slotwise candidate | 58/46/46 | 34/46/28 |
| All gate decisions in a root, slotwise candidate / 12 | 3/3/4 | 0/0/0 |
| Both full answers, fitted max cosine | 44/42/12 | 14/32/8 |
| Both full answers, slotwise candidate | 42/26/22 | 16/32/10 |

Compared with always reading the top occupied slot, the candidate increased correct NONE outputs from 14/10/8 to 80/70/62 in the repeated-owner panel, but correct factual outputs fell from 54/54/38 to 46/38/34. In distinct-owner banks, correct NONE rose from 24/14/22 to 52/74/64 while correct factual outputs fell from 44/58/30 to 32/38/18. A no-source decision is not itself an exact NONE output; these counts include the unchanged answer head.

**Applicability, recency and answering remain separate.** On repeated-owner hits, the candidate's own top record belonged to the matching owner in 78/66/64 of 96 cases, but was the latest matching record in only 70/56/50; the native router's latest counts were 62/66/50. The candidate flipped its existential gate on 8/14/12 repeated-owner and 22/22/34 distinct-owner timestamp-only pairs out of 96 per composition, although the hit/no-match truth never changed. Physical order was not responsible: no gate decision changed under the matched placements, and realigned score drift was below 10⁻¹⁵. Even giving the head the oracle latest record at full strength produced only 70/78/70 correct repeated-owner hits and 60/72/50 distinct-owner hits out of 96.

**Decision:** Do not promote this gate or its own record ranking. Matching the composition removed one confound but did not establish reliable transfer. A useful next test would isolate owner identity from timestamps: add an applicability consistency penalty on paired TRAIN banks that differ only in clock tokens, keep the same fitted cosine comparator, and evaluate on new matched time, value and distractor interventions. Retain latest-record selection and unrestricted answer accuracy as separate gates. Do not tune thresholds on this open set.

All six native roles completed once (15.44 seconds of recorded serial CPU-role elapsed time, not a serving-cost benchmark). No reader, router or answer-head weights changed, no protected or paid run occurred, and these rows share roots rather than providing independent repetitions.
