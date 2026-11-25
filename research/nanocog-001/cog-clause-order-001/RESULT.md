# COG-CLAUSE-ORDER-001: Learned compositional binding in an eight-token grammar

Both learned clause arms passed every frozen calibration and held criterion in all three seeds. Exact symmetry preserved logits across reversals, but the position-sensitive, otherwise matched clause arm reached essentially the same accuracy. The useful capability demonstrated here is learned clause-level binding with an explicit segmentation prior, not a unique accuracy gain from symmetry.

Each fit used 1,024 updates, 8 complete 24-row bundles per update, a 40-token full-vocabulary learned decoder, and counterbalanced assignments. The two clause arms had 37,600 nominal parameters and identical initial tensors and data streams per seed. Both use one forward pass, receive the eight raw token IDs, and assume the first two groups of three tokens are clauses. No gold IDs, output masks, lookup tables or exact executors at inference. The 35,944-parameter raw two-block causal attention control had exactly the same learning data and updates but no explicit segmentation. FLOPs were not matched.

## Held exact accuracy

Entries count histories for which **all 24** answers were correct, not just the original question. Each stratum contains 256 histories from 64 held-out key pairs: 6 worlds (base, two single-value edits, value swap, and two single-value edits of that swap) × 2 clause orders × 2 queried keys. These are descriptive counts; four histories sharing a key pair are not four independent groups.

| Arm / seed | Same entity, distinct relation | Same relation, distinct entity | Distinct both | All gates |
| --- | ---: | ---: | ---: | --- |
| SYM 3371 | 243/256 | 249/256 | 256/256 | Pass |
| SYM 3471 | 254/256 | 252/256 | 256/256 | Pass |
| SYM 3571 | 255/256 | 249/256 | 256/256 | Pass |
| ORDER 3371 | 247/256 | 247/256 | 256/256 | Pass |
| ORDER 3471 | 252/256 | 254/256 | 256/256 | Pass |
| ORDER 3571 | 245/256 | 256/256 | 256/256 | Pass |
| RAW 3371 | 256/256 | 0/256 | 256/256 | Fail calibration and held |
| RAW 3471 | 256/256 | 0/256 | 256/256 | Fail calibration and held |
| RAW 3571 | 256/256 | 0/256 | 256/256 | Fail calibration and held |

The frozen individual-accuracy, ordinary-family, strict edit-triplet and complementary-swap gates all passed for both clause arms in every stratum. SYM produced identical full logits for each reversal (27,648 in total across the three seeds); ORDER had at most 0.934 absolute logit change and one to five top-1 disagreements per affected stratum, while maintaining high accuracy. Strict disagreement and strict correctness are different measurements. The score fields first/second intersect the target and distractor triplets across the two original/swapped assignments and both orders, with the queried identity chosen by physical position; both_order_triplets intersects both roles and equals joint under this exact design.

RAW scored 3,077–3,084/6,144 individual rows (50.1–50.2%) and 0 complete ordinary or all-24 families when two entities shared the relation. It also scored only 1,527–1,551/3,072 individual rows in that calibration stratum, so this is a failure to solve the task under the tested budget, not evidence that well-trained attention cannot bind entities. Broad accuracy would hide a systematic mistake. The result is consistent with reliance on the easier relation cue; it does not prove the exact circuit or that training longer would or would not fix it.

## Decision

Keep SYM as a mechanically safe bounded reference for independent assertions and ORDER as a matched learned control, not as the next architecture promotion. Train a new comparison on variable clause lengths, unseen syntax, and distractors, giving attention the **same visible clause boundaries** in the matched structural control. Compare against a raw-token version that learns its own segmentation. Include order-sensitive revisions of the same key with an explicit directional or temporal path; making these invariant would be semantically wrong. A separate fixed-slot retention test and matched-quality full-cost inference benchmark are still required. No MAIN or protected TEST used.
