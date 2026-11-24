# COG-PULSE-004A-R2: does any record apply?

**The learned applicability gate did not qualify.** On new three-owner banks it beat the native router's nearly absent NULL decision, but lost to a single maximum-cosine feature in all three seeds. It improved correct NONE responses when paired with a full-strength read, at the cost of too many correct factual answers. No gate or threshold should be adopted from this run.

Both controls and the candidate used only the same 576 original TRAIN rows: 288 exact same-bank pairs, evenly split between a factual query and a true no-match query. The original encoders, router and tied full-48 answer heads stayed frozen. The relation gate saw 48 pooled coordinates of frozen query×record key interactions. A query-only control had the same width. Scalar controls used maximum cosine or route margins. All were fit once on TRAIN with the same regularized logistic procedure, and evaluated at the fixed 0.5 cutoff. A positive gate sent the router's top occupied record to the head at unit strength. A negative gate used the frozen head's real zero-read logits, not a forced NONE.

### Primary: 288 exact hit/no-match pairs across 24 fresh three-owner roots

Counts are in seed order **173 / 271 / 907**. Within each pair the occupied bank is unchanged and only the query differs.

| Policy | Both applicability decisions correct / 288 pairs | Exact factual answers / 288 hit rows | Correct NONE / 288 no-match rows | Both full answers exact / 288 pairs |
|---|---:|---:|---:|---:|
| Always read top occupied | 0/0/0 | 154/150/100 | 42/36/44 | 20/14/14 |
| Original router's NULL choice | 0/12/0 | 154/150/100 | 42/48/44 | 20/20/14 |
| Learned relation gate | 94/94/80 | 106/88/56 | 184/180/190 | 62/62/32 |
| TRAIN-calibrated maximum cosine | **102/124/90** | 122/120/72 | 190/206/196 | **66/82/46** |

The relation gate correctly accepted 172/170/142 hits and correctly rejected 174/164/178 true no-matches. At the same time it accepted 114/124/110 no-matches, and completed zero of 24 applicability roots in every seed. The underlying router chose the current record on just 184/184/160 hit rows; even giving the frozen answer head the oracle current record at unit strength yielded only 200/228/170 correct hit answers. These are separate applicability, selection and answer errors.

The candidate scored 226/252/234 complete TRAIN pairs out of 288, versus 94/94/80 on the fresh panel. The observed drop does not isolate its cause: fresh banks have three distinct owners, whereas TRAIN banks repeat an owner. In the **separate previously exposed revision stress panel**, the relation gate also trailed maximum cosine, 38/36/38 versus 62/60/48 exact applicability pairs out of 120. The replay reproduced the archived logits and weights with zero measured difference. The fresh relation decisions were unchanged under all 288 matched placement reversals. Query-only was exactly 288/576 correct on applicability; logit magnitudes were ≤2.34×10⁻¹⁴, so its nonzero signs were numerical residuals, not evidence of a learned query prior. The open panels and seeds reuse cases within shared roots and balancing blocks; rows are not independent tests.

**Next:** Use a new, precommitted TRAIN/validation split with both duplicate-owner and distinct-owner banks and matched hit/no-match queries. First find out whether the simple cosine gate transfers when bank composition is matched. If it still fails, test a shared query-to-record match score before pooling, with separate held gates for applicable evidence, latest-record selection and full-vocabulary answers. Do not tune on these exposed panels.

Six capped native roles completed once. Neither the original reader nor the answer-head weights changed. No protected benchmark or paid compute was used; this is not a production rejection guarantee.
