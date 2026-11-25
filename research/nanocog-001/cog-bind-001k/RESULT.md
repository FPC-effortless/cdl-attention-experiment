# COG-BIND-001K: counterbalanced training did not repair alias binding

## Decision
All six fits scored **50% exact source and item** on held rows. Neither training on one fixed declaration assignment per family nor training on both assignments solved **any of 768** complete held families, **any of 3,072** complete four-cell grids, or **any of 6,144** alias-switch pairs per arm. Both arms decoded the correct item in **12,288/12,288** rows when supplied the correct record. This is a failure of selecting the right directed record under changing alias meanings, not evidence that increasing data or training can never help.

## Matched intervention
Both arms used the same 35,594-parameter learned ordered ranker, initialization and schedule per paired seed. The learner sees four text clauses and one question, with no typed or correct record/recipient input. Fixed repeats its one within-family alias assignment; balanced sees both. Same 512 families, 1,024 steps, 65,536 row exposures and 2,030,848 nonpadding tokens per fit. Fixed covers 4,096 unique rendered training prompts and 825 distinct alias–referent links; balanced covers 8,192/1,372. This is a regimen comparison, not an objective-only or architecture comparison.

Each held family crosses two declarations, two directions for the same pair of reciprocal records, two query aliases and two clause orders. Each single change flips the correct record and item; the double edit preserves the item and stable record identity while the correct recipient changes. Exact transfer pairs and alias–referent links are split-disjoint; names, item classes and the small grammar are familiar. A scripted directional parser solves 4,096/4,096 individual held rows. Restricted no-map and no-direction comparators get 2,048/4,096 and cannot solve a full crossed grid. These are model-free controls, not overall learned-model ceilings.

## Held evidence across three matched seeds
| Measurement | Fixed | Balanced |
|---|---:|---:|
| Record and item correct | 6,144/12,288 | 6,144/12,288 |
| All 16 presentations correct | 0/768 | 0/768 |
| All four alias × direction cells at fixed query and order | 0/3,072 | 0/3,072 |
| Exact both sides of alias-only switch | 0/6,144 | 0/6,144 |
| Exact both sides of direction-only switch | 716/6,144 | 482/6,144 |
| Item from supplied correct record | 12,288/12,288 | 12,288/12,288 |
| Correct canonical recipient from detached probe | 54/12,288 | 50/12,288 |
| Record, item and detached recipient correct | 28/12,288 | 20/12,288 |

Every fitted model always retained the same unique selected slot and answer under a declaration-only switch. This is hard-decision invariance, not numerical invariance: a separate audit found that raw rank vectors changed in every held alias pair, including after removing a common shift across slots. The shift did not cross the decision boundary. Direction-only changes did move the chosen slot on 1,432 fixed and 964 balanced pairs; only half of those flips put both answers right. The detached linear recipient probe is a diagnostic, with no feedback to native inference. Its weak result neither proves absence of alias information elsewhere nor identifies the internal cause.

## Verification and next decision
Six bounded CPU fits, 12 blind prediction packets and 30,720 scored prompt rows; independent recount matched all 12 tracked metrics per packet. Neither arm passed any held gate. No natural-language, latency, or general attention-replacement claim. Keep these exact controls, but do not scale the same soft preliminary read based on label balance alone. The next comparison should separate learned alias-declaration selection from learned recipient-to-directed-record selection. Test the native chain plus a separate correct-bridge intervention to localize failure, and hold out fresh alias/link and transfer combinations. Keep any typed parser or oracle outside normal outputs.

Data pin SHA-256: `14d18709dda22980b4cb96789b0c3d0341cd2fe7bd7cac6b2f4851d35bbbbce2`. Scored result SHA-256: `9683f163e893e933e3432fea0d8c904a9ef95faa6eb130129264bc571e5f78fc`.
