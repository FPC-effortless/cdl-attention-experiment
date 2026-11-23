# COG-BIND-001Q: explicit checks on learned evidence improve rewrite decisions, but do not fully qualify

## Result
Against a similar-size confidence head on the same frozen routes, the proposed-set check increased exactly answered valid rewritten inputs from **4,216 to 4,537 of 4,992 (84.5% to 90.9%)**, while reducing invalid answers from **189 to 111 of 10,752**. The structured system passed both predeclared rewrite screens in **one of three seeds**, versus **none** for confidence alone. This is an aggregate improvement, not a seed-stable qualification. Structured decisions were slightly worse on familiar wording, where both passed all three seeds.

## What changed
The frozen composer still proposes the original answer and supporting transfer. The structured head learns which declarations match the query, which clauses are transfers, and their canonical participants from frozen text-derived features. Its rule counts distinct people first, then distinct transfer witnesses. Equivalent declarations for one person collapse. Different transfers do not collapse just because the item agrees. A unique proposed witness must match the frozen route's selected slot, and the item selection must be unique. It can only allow or reject, not substitute a witness or answer. Tied unresolved evidence and route disagreement have different explicit reasons. “Missing” describes the predicted set, not proof that no evidence exists.

The compared heads have 723 versus 725 trainable parameters, the same 10,496 actor rows, 32 passes and 1,312 updates per fit on each of three frozen diverse 001O checkpoints. The structured head receives richer local TRAIN supervision (83,968 clause annotations per corpus) rather than just reason labels, so this is a comparison of complete methods, not an isolated architecture effect. The literal grammar parser, perfect proposal sets and reference IDs are separate from native decisions.

## Held measurements
Counts reflect three checkpoints over 128 underlying held families per wording, with the same semantics across the paired rewrite. There are 4,992 valid and 10,752 invalid model–row observations per wording.

| Metric | Familiar confidence | Familiar structured | Rewritten confidence | Rewritten structured |
| --- | ---: | ---: | ---: | ---: |
| Valid accepted | 4,988 | 4,963 | 4,262 | 4,595 |
| Valid accepted and exact record+item | 4,988 | 4,963 | 4,216 | 4,537 |
| Invalid accepted | 34 | 36 | 189 | 111 |
| Whole 41-row families exact or properly rejected / 384 | 366 | 351 | 232 | 282 |
| Both predeclared screens / 3 seeds | 3 | 3 | 0 | 1 |

On the rewritten crossed test, false accepts when distinct people shared an alias **and the same item** fell from **56 to 8 / 1,536**; when their items differed they fell from **68 to 8 / 1,536**. The structured check got **2,784/3,072** equivalent-duplicate valid rows exactly right after rewriting. Its improvement is not uniform across failure types: among core conflicting incoming transfers, equal-item false accepts fell 16→15/960 while different-item false accepts rose 13→30/960. Across all rewritten invalid rows, missing-alias false accepts rose 5→42, conflicting-alias fell 155→19, missing-incoming rose 0→5, and conflicting-incoming rose 29→45. On familiar wording the structured check falsely accepted no matched-cross alias conflicts, but it still admitted 31 missing-alias rows.

## Why the candidate check did not qualify
The candidate sets were fully exact on **13,379/15,744 (85.0%)** rewritten rows; the canonical referent set was exact on **13,799/15,744** and incoming witness slots on **14,357/15,744**. Substituting perfect reference proposal sets as a diagnostic eliminates all 111 false accepts and recovers the full **4,923/4,992** exact native coverage, but still accepts 65 incorrect items from the frozen route. An independent raw check found 65 wrong item readouts despite the right record and four wrong records across the three seeds. The rule certifies neither extraction quality nor the item's correctness. The next change should improve the two proposal stages and explicitly check evidence for the item itself, not just for the selected record.

## Data and verification
TRAIN/DEV/HELD use 256/64/128 grouped families, 41 rows each, and held-only renderer variants over known O vocabulary. The eight novel renderer triples are balanced against the incoming item and duplicated-referent factors in the active v2 panel. The draft was held before any model call because those factors were coupled. Q role splits share no exact alias link, transfer pair or actor row; transfers and actor rows are fresh versus O/P and alias links fresh versus P. **254 of 516** Q HELD alias links appeared in the frozen O model's TRAIN, so this is not a zero-shot alias-link result. In the harder crossed panel the same passage simultaneously supports one alias query and conflicts for another.

The frozen routes were not refit; there were six new head fits, 12 frozen feature packets and 18 blind output packets. All sealed results independently agree across nine seed/panels and 99 policy/oracle metric groups. No test of unrestricted language, inference cost or deployment reliability.

Data pin `8f1b85a76340a0963c1b0a3bbe88fb64eeee26844dc5e66e289a1727272212b2`; score `b59f9447cf4ad4189c29834fa64ab4c09d96bc45448f8990408ad956409049e3`.
