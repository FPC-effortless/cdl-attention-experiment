# COG-BINDING-SCALE-001: retrospective error breakdown

This is a read-only analysis of the existing, unqualified prediction tapes, not another trained experiment. The original TrainLab failure and evidence status are unchanged.

At 6.4M parameters, the two weaker **top-2 pooled** seeds (271 and 907) made 5,324 individual wrong predictions in the 256-token subject cohort. **5,170 of 5,324 (97.1%)** were values actually present elsewhere in the same example, not absent values. Of those 5,170, 1,747 belonged to the counterfactual swap partner and 3,423 to the other two facts. They emitted the same answer despite the subject-value swap on 1,416 and 1,096 of 2,048 pairs. In the matched relation cohort, these very checkpoints got 2,048 and 2,047 complete pairs correct. The problem for these models is getting the present value attached to the correct entity, not an inability to emit values in general.

For top-2 separated at the same size, seeds 271 and 907 still made most subject errors on present non-target values, but 1,647 of 6,585 errors (25.0%) were absent values. The mechanisms need not share a single failure; this distinction motivates a matched routing intervention. “Present” only means recoverable from the actor text; the analysis does not observe which windows the model chose or show that the reader consumed a value it emitted.

The audit independently parsed all four values per actor row, verified that twin histories expose the same value set, and matched complete pairs, world accuracy and unchanged predictions against the previous recount for 288 cells at both recorded endpoints. Token bins: correct target, counterfactual partner's value, value of either other fact, or absent. Seeds reuse the same held families and the token counts above are descriptive, not extra independent families. Nothing here qualifies the native run, general language capability, or sparse routing.
b21d29ec35d3f3eb9f36813c3b9c6b968fdf7025acda023ec4ad54a5d264fa05  ERROR_COUNTS.json
