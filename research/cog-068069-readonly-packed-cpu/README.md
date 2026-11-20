# COG-068/069: governed advice ran; neither promotion gate passed

The compact predictive representation did not beat the simpler references on a constructed read-only advice task. Packing several coavailable requests preserved the exact emitted advice, but it did not establish the required whole-request CPU benefit. **Keep the simpler policy and folded per-request implementation.** This is not a fully learned runtime, authentic interaction result, sparse-accelerator benchmark or qualified deployment.

## 068: look-ahead advice, not an action

Each simulated request used only three already observed, timestamped readings per eligible lane. The system advised which lane to prioritize for verification of a *later* measurement, which arrived regardless of the advice. An exact authority layer checked availability, scope, same-tick position and revocation **before** semantic use. Scores did not grant authority or create an observation. Every emitted record said prediction-only, not observed; the selector never wrote external or simulated plant state. Later independent target records scored the advice.

We compared 96 independent source groups with two paired authority variants each, using 36,288 effective TRAIN and 12,096 CAL rows for the fitted comparisons. Each active policy selected one lane whenever at least one was eligible. Twelve all-revoked requests emitted no lane, as required. There were 46 intact and 31 revocation-variant requests with an eligible future exceedance. Those counts were checked before fitting; the threshold came from disjoint design seeds.

| Same-input policy | Intact hits / 96 cards | Revocation hits / 84 cards | Total hits / 180 cards | Cards without an exceedance |
|---|---:|---:|---:|---:|
| Raw affine history | 44 | 31 | **75** | 105 |
| Tuned last-value plus slope | 43 | 31 | **74** | 106 |
| Predictive rank 4 | 42 | 30 | **72** | 108 |
| Matched-rank PCA | 41 | 28 | **69** | 111 |
| No advice | 0 / 0 | 0 / 0 | 0 / 0 | 0 |

Predictive minus raw was −1.56 percentage points over **all 192** paired requests (95% whole-group interval −4.17 to 0.00); minus slope −1.04 (−3.65 to +1.04); versus PCA +1.56 (−0.52 to +4.69). None met the preregistered requirement of a positive lower bound against all three strong controls. Exact output and authority legality passed; correct status alone does not prove model utility. Sensor forecasts could still be wrong, as the false-card counts show.

The rank-4 fit used all 16 next-observation channels for its pre-projection objective; this is disclosed, not additional sensor streams unavailable to the alternatives. Learned storage was 293 coefficients including scaling for the compact form. Both it and the raw linear form can be folded to a 49-coefficient affine inference map, so the comparison tests a training bias rather than a new inference architecture. Selected three-fit wall/CPU were .259/.255 seconds.

## 069: parity yes, speed promotion no

Both CPU paths already used the **same folded affine function**. The comparator made one optimized BLAS call per request; packing made one matrix call for admitted lanes across a coavailable wave, then split and serialized exact output. A nonfolded reference checked every result. Grouping availability was synthetic, not oracle time-based lookahead. All 2,520 timed waves, 40 fresh-model-read first waves, and two separate traced allocation samples were retained and checked. Emitted advice was identical across variants. No numerical near-tie fallback occurred in this held timing cohort; the hand-authored tie fallback test separately passed.

| Coavailable wave size | Folded-loop warm complete p95 | Packed p95 | Paired 95% interval for p95 reduction |
|---|---:|---:|---:|
| 1 | 0.0450 ms | 0.0601 ms | −86.47% to −9.19% |
| 4 | 0.1729 ms | 0.1657 ms | −13.48% to +20.82% |
| **16 (primary)** | **0.4945 ms** | **0.4601 ms** | **−15.62% to +25.55%** |

At size 16 the observed p95 reduction was **6.95%**, below the required **10% interval lower bound**. Fresh model read, preparation and first size-16 wave p95 was 0.8557 → 0.8182 ms, meeting the no-more-than-5%-harm guard. The single first-wave traced incremental allocation grew from 5,032 to 22,752 bytes, still below the 8 MiB cap. Whole-process Linux high-water RSS was 128,824 KiB; it is not incremental or attributable to either arm separately. Warm/cold clock boundaries included occupancy, scoring, ranking and output serialization and exclude actual arrival waiting, network or process start. Fresh model reads happened in the same process with unspecified OS file caching, not a cold-process benchmark. The one-wave traced measure can omit native allocations.

The data do not establish CPU speedup under the frozen rule; the best point estimate at size 16 does not outweigh uncertainty or the size-1 regression. No GPU or sparse kernel ran. Do not promote packing as a default from this study.

## Evidence boundary

- Closed constructed pipeline: Forge held separate from learner; exact independent admission for data preparation, TRAIN/CAL fit, fresh DEV prediction/timing, and later scoring. Native TrainLab verified effective sizes/work and stored results, Veritas-style scorer checked outputs and timings, Research Lab linked immutable evidence.
- Pinned ecosystem eec439492407b0aa26922c23b5ae40d95289f4eb, Python 3.12.14 and one OpenBLAS thread. Native stages succeeded once each and both science/engineering gates were deliberately allowed to fail.
- No protected TEST, spend, external communication/action, live sensors, authentic prospective interactions, package qualification, or broad learned-runtime claim. This public package excludes raw observation rows, labels, weights, forecasts, timing rows and private approvals.
- Independent retained-artifact audit rechecked all three native receipts and lineage, used a separate authority reference to verify all 960 advice records (zero unauthorized lanes), reproduced paired intervals and utility outcomes, checked all timing keys and independently confirmed both failed gates. No candidate or native stage ran again.
