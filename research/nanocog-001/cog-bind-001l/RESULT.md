# COG-BIND-001L: learned bridge routing passes the crossed alias-and-direction test

## Decision
The learned two-step route got every held record and item right: **12,288/12,288 rows, 768/768 complete families, and 6,144/6,144 alias-switch pairs** over three seeds. The same-budget bypass got **6,132/12,288 rows, 0/768 complete families, and 6/6,144 alias-switch pairs**. All three routed fits passed the preregistered bounded gate; no bypass fit passed. The 768 are 256 distinct synthetic families repeated across three model seeds, not 768 independent test families.

Both arms identified the correct alias declaration in every held row. The crucial difference was routing the final decision through its learned referent. Given a correct transfer record, both decoded the item perfectly.

## Matched mechanism
Both arms used the same learned embeddings, five learned token-pooling heads over the query and clauses, learned clause eligibility, learned identity projection and item readout. The routed arm forms its record address from the learned distribution over declarations and their learned referent vectors. The bypass uses the query plus the uniform average of the four clause referent vectors; it does not use the predicted declaration weights to form that address. The bypass is not an architectural upper bound or a theorem of alias blindness.

The two arms use 17,338 parameters each, same paired initialization, exact same balanced actor and label corpus, optimizer schedule, and supervision. Each saw 1,024 updates, 65,536 row exposures and 2,034,304 nonpadding tokens. Bridge slot, target referent, support and answer supervision enter the training losses only. No gold mapping, parser, typed slot or reference label enters native inference. Both arms rank all four raw-text clauses.

The held grid crosses both alias meanings, both directions for the same complementary transfers, both alias queries and two clause orders. Entire transfer pairs and alias–referent links are held out, and entity/alias/item vocabularies are different from the previous 001K study. Within 001L, atomic names, item classes and templates are shared across splits. The exact directional parser solves all 4,096 unique held rows, while the no-map and no-direction restricted controls get 2,048 each. This is not a natural-language or novel-syntax result.

## Held measurements across three seeds
| Measurement | Bypass | Routed |
|---|---:|---:|
| Unique correct declaration | 12,288/12,288 | 12,288/12,288 |
| Correct composed canonical recipient | 4,266/12,288 | 12,288/12,288 |
| Correct final record and item | 6,132/12,288 | 12,288/12,288 |
| Correct declaration, recipient, record and item together | 2,312/12,288 | 12,288/12,288 |
| Complete family, all 16 final decisions correct | 0/768 | 768/768 |
| Complete alias × direction quartet at fixed query and order | 0/3,072 | 3,072/3,072 |
| Both sides right after alias-only swap | 6/6,144 | 6,144/6,144 |
| Both sides right after direction-only swap | 984/6,144 | 6,144/6,144 |
| Item read from supplied correct transfer | 12,288/12,288 | 12,288/12,288 |

## Same-checkpoint causal test
Independently of native prediction, the actor enumerated all possible forced first-hop choices before reference scoring. Choosing the routed model's own highest-ranked declaration as a hard first hop preserved all 12,288 final successes and all three-stage successes. With the same actor and weights, forcing the other valid declaration made the routed model pick the correct alternate record and item in **12,288/12,288 rows** and retain the original in **0/12,288**. The bypass output never responds to the forced bridge by construction. These interventions verify causal use of this interface on this task, not validity of internal semantic roles beyond it.

## Scope and next
Six CPU fits, 12 blind actor packets, 30,720 scored prompt rows, then independent recount. No hardware inference comparison, free-form decoding, novel syntax, long contexts, or general attention-replacement qualification. This is a usable reference result for a narrow compositional routing primitive. Next, keep the trained architecture and supervision fixed and test independently frozen surface forms, extra plausible declarations and records, irrelevant edits, and missing/ambiguous evidence with an abstain path. Only then test measurable end-to-end cost against a competent contextual baseline.

Data pin `48b6fcd4a999bf20bfb2b76314b1f30ef8c1373f2258b39f16869fcacf54cb69`; score SHA-256 `a13648326e452bf1abc74d87695919e1e895f5fedaecfc7b6a7b54bfd710b23d`.
