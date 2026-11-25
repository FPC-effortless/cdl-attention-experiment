# VENTILATION-ALERT-007: intermittent calibration and request budgets

**The simple periodic check beat the adaptive policy on late sensor shifts.** With accurate checks, a late shift and dense opportunities, the adaptive policy got 222/288 complete terminal pairs and missed 22/192 severe cases. The periodic policy got 276/288 and missed 3/192 at nearly the same request cost. Always fresh got 280/288 and missed none, but used eight checks per history versus three for periodic.

## Frozen design

Five deterministic policies use the same cutoff rule on primary minus witness plus the most recently acquired estimate of witness offset. They differ only in whether to request an offered synthetic independent calibration observation. Never takes none; first-only takes one; periodic takes up to three spaced by at least three ticks; adaptive takes up to three with a two-tick cooldown and triggers on estimate age, near-cutoff estimated concentration, or witness change; always takes every offered check. An unrequested check is not passed to the policy. Requests have immediate results at eligible ticks; delay metrics measure onset to acquisition, not communication latency.

Replay 288 independently generated paired eight-tick families through eight cells: early/late offset onset (ticks 2/5), dense/gapped offers (every tick or {0,1,2,6}), and accurate/degraded independent check error (uniform integer ±1/±5). Common nuisance draws and check opportunities are fixed across the policies. The same families are reused across cells, so do not add denominators and call them independent. No training, real measurements, hardware, control or architecture ranking. Unit cost is check count, not currency or deployment latency.

## Terminal complete pairs / severe misses / mean requests per history

Pairs /288; misses /192 severe final states. Count a pair only if both counterfactual alerts are correct.

| Cell | Never | First only | Periodic | Adaptive | Always fresh |
|---|---:|---:|---:|---:|---:|
| early-dense-accurate | 84 / 64 / 0.00 | 82 / 69 / 1.00 | 276 / 3 / 3.00 | 276 / 2 / 2.99 | 280 / 0 / 8.00 |
| early-dense-degraded | 84 / 64 / 0.00 | 93 / 63 / 1.00 | 226 / 19 / 3.00 | 231 / 14 / 2.99 | 234 / 16 / 8.00 |
| early-gap-accurate | 84 / 64 / 0.00 | 82 / 69 / 1.00 | 276 / 3 / 2.00 | 276 / 3 / 2.85 | 276 / 3 / 4.00 |
| early-gap-degraded | 84 / 64 / 0.00 | 93 / 63 / 1.00 | 226 / 19 / 2.00 | 226 / 19 / 2.81 | 226 / 19 / 4.00 |
| late-dense-accurate | 84 / 64 / 0.00 | 82 / 69 / 1.00 | 276 / 3 / 3.00 | 222 / 22 / 2.98 | 280 / 0 / 8.00 |
| late-dense-degraded | 84 / 64 / 0.00 | 93 / 63 / 1.00 | 226 / 19 / 3.00 | 170 / 36 / 2.99 | 234 / 16 / 8.00 |
| late-gap-accurate | 84 / 64 / 0.00 | 82 / 69 / 1.00 | 276 / 3 / 2.00 | 276 / 3 / 2.44 | 276 / 3 / 4.00 |
| late-gap-degraded | 84 / 64 / 0.00 | 93 / 63 / 1.00 | 226 / 19 / 2.00 | 226 / 19 / 2.61 | 226 / 19 / 4.00 |

Adaptive met the prespecified budget/risk gate in 6/8 cells. It failed both late/dense cells despite saving approximately 63% of requests. In late/dense accurate it trailed always-fresh paired accuracy by 20.14 points (descriptive same-family bootstrap interval −25.00 to −15.28) and missed 11.46 points more severe cases (interval +7.03 to +15.93). Of 576 late/dense accurate histories, 162 exhausted the adaptive cap before the shift and 171 never took a post-shift check despite having an opportunity. A median delay of zero among successful checks therefore must not be mistaken for full coverage. Periodic obtained a post-shift check in all 576 histories at delay one tick. On these fixed cells, periodic meets the same point thresholds in all eight, but its paired interval against always-fresh in late/dense accurate (−3.82 to +1.04 points) includes losses outside the 3-point tolerance. That is not proof of equivalence.

Terminal risk can hide early misses. Late/dense accurate post-onset severe misses were 2/576 always, 64/576 periodic and 87/576 adaptive across all three post-onset ticks. Terminal false severe alerts were 2/384, 5/384 and 18/384 respectively. Read both terminal and post-onset counts before selecting a schedule. The point result favors the periodic rule as the next baseline, not as a deployable policy; its alignment with the fixed final tick may explain the strength. Test randomized shift and query timing, or an adaptive policy reserving a check for late risk.

## Information limit

In a separate 72-family exact-alias panel, both severity alternatives had identical complete available observations and no check after the change. Every deterministic policy produced identical trajectories and solved **0/72** complete pairs. This is a fixture-level impossibility, not a claim that all aged observations are ambiguous.

## Audit and execution

A first attempt failed before actor execution because the workspace disallows the extra bubblewrap network namespace. The preserved recovery changed launcher isolation only; policy bytes, tapes, truth and scoring remained exactly identical. The local actor does not use networking and has no detached truth in its filesystem view. Successful frozen TrainLab engineering diagnostic (no model qualification); a separate scalar recount matched seven published endpoints in all 45 policy/condition cells, including the alias cohort. Replay retains both attempts and verifies all artifacts.

- LOCK.json SHA-256: 22eb6f0361e3e169fcdf0db2977d15645795f809b44c01744c63b8ae2b72bda6
- RUN.json SHA-256: d563e3fc04ffce6e33172734f333c119221d0a7a65f6f0fe641406aa2a8f3918
- SCORE.json SHA-256: 73b2304b03458846343510a4f04c37914fc102f15fd5314aeb32a6b59abfe6a6
- RECOUNT.json SHA-256: 714c80bb0011e6d13761323c49f190bbd62deee2038946b00f5a0e5dc9d9a0e6
