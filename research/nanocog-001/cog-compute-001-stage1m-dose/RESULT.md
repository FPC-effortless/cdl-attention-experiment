# COG-COMPUTE-001 Stage 1m: frozen subject-routing dose

**Stronger routing recovered more subject answers, but did not fix subject binding.** On the weakest original ×4 checkpoint (seed 173), moving all of the competing fact's weight to the correct fact recovered 35 previously incorrect exact answers with no native-correct answers lost, taking 79/192 to 114/192. It solved 6/48 complete eight-case subject-swap families. The 0%, 25%, 50% and 75% doses solved none.

## Same observations, frozen computation

One CPU TrainLab actor, same original 1,152 already-exposed synthetic raw prompts and unchanged checkpoint. No training, new held data, or new memory. At native ×4, enumerate all 100 ordered pairs of ten earlier spans at each of five prospective fractions of the donor's native weight. Keep payloads and other routing weights fixed. The diagonal is no movement. No label-identified span is supplied to the actor. Detached scoring selects correct and competing intact spans, and compares with eight other recipients from the same donor at the same dose; their windows can overlap evidence. Score the exact answer only when the model naturally emitted the answer marker and native continuation emits EOS after the intervened full-vocabulary argmax. The opposite direction is a separate diagnostic with its own donor weight, not a dose-matched comparison.

## Subject changes (both exposed layouts pooled)

| Competing weight moved | To correct /192 | Reverse /192 | To other spans /1,536 | Correct vs other wins/losses | Complete 8-case families /48 | Mean total weight moved |
|---|---:|---:|---:|---:|---:|---:|
| 0% | 79 | 79 | 632 | 0/0 | 0 | 0.00% |
| 25% | 87 | 76 | 667 | 43/14 | 0 | 2.40% |
| 50% | 97 | 70 | 702 | 97/23 | 0 | 4.80% |
| 75% | 106 | 65 | 755 | 135/42 | 0 | 7.20% |
| 100% | 114 | 59 | 776 | 174/38 | 6 | 9.60% |


Matched comparisons share rows and recipients and are not independent observations. At full transfer the target scored 59.38% versus 50.52% across other recipients. Both counterfactual answers were exact on 9/192 pairs natively, 26/192 at half transfer and 52/192 at full transfer. Most families still fail even at full dose. Removing all of one rival's weight moved only 9.60% of the entire mixture on average. This intervention does not show whether the residual failure is in other routes, the value representation, or the readout. A clean target-payload probe should hold the route fixed and separately manipulate payload content; it should not introduce token supervision at inference.

The rows and weakest seed were selected for mechanism diagnosis, not new qualification. The replay score separately retains the relation-change, subject-same and relation-same cohorts.

## Execution and evidence

One frozen actor succeeded in 13.586 s with zero optimizer updates. Engineering diagnostic, no promotion. The 0% transfer exactly matches the native subject endpoint. At 50%, all 115,200 ordered marker choices, the stored suffix mask/choices and donor doses exactly match the prior tape. Across all doses native decode, weights and marker logits match the prior tape. The original detached scorer failed on NumPy's default index dtype for empty swap cohorts; a separately recorded integer-dtype-only scoring recovery succeeded without actor rerun. A separate scalar implementation verified five published endpoint counts at every dose.

- LOCK.json SHA-256: 641feee74bcc2ef8b73979b7b46585f72ca9a8ae5ff4c88265d47a0b10acef69
- RUN.json SHA-256: 51bd4f806cbdb5c1dfb9af5a8cc8ebf33bf661a933e4eac5203f639fa096b212
- SCORE_RECOVERY.sha256 SHA-256: 94a099385a11a0bed4ee18cc115df8caacbca4d10596524442abca3053b7e67b
- SCORE.json SHA-256: 89b954ce7dbc4c4ffe2c73eca6e7168fc6e85ef43b15205e166a3650ae5c2858
- RECOUNT.json SHA-256: bec3f28b111c322a4c9b5502fac41088f92f4786987371fb6c80afcaed8cec7b
