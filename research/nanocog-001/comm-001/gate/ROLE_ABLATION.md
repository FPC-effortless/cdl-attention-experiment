# COG-COMM-ROLE-001: local architecture ablation

Status: prospective, not fitted. Prerequisite geometry and deterministic
shortcuts are reported in ROLE_PREFLIGHT.md. Historical EASY is not replayed.

## Question and decision order

Can a model learn to select the correct binding on the familiar seven-token
task when all cues are in view? First compare two fixed endpoints from a
single trajectory (128 and 1,280 updates); do not restart at step 128 or
change the schedule. Then isolate which architecture changes help at the
same budget. No packet store, recurrence, or global routing in any arm.

| Arm | Change relative to gated32 | Parameters at vocabulary 287 |
|---|---|---:|
| gated32 | Original Arm A, two width-32 blocks, four causal taps per layer | 37,414 |
| static32 | Same taps and trunk; trained position weights replace input-conditioned gate | 37,158 |
| attention32 | Same width and trunk; two causal attention mixers, rank 10 with learned relative lag; matched active parameter budget | 37,420 |
| gated48 | Same gated family and depth with width 48; capacity intentionally increases | 69,926 |

Static weights start at the same logits as setting the gated model's input
weights to zero. They are still trained. The comparison includes removal of
256 active weights, not dummy capacity; at fixed vocabulary the difference
in total parameters is 0.68%. Causal attention is six parameters (0.016%)
above the width-32 reference. Equal parameters are not equal computation.
If narrow attention also fails, that is not evidence that standard
Transformers cannot solve the task. Attention changes graph, projection,
position encoding and parameterization together; the contrast is for
competence at this budget, not a single operator causal isolation.
The three width-32 arms start from exactly the same shared trunk tensors.

## Frozen comparison proposal

Use the 320 synthetic FIT families and the 120 disjoint joint combinations
on the public familiar control panel, eight co-varied worlds per family.
Original full-vocabulary prefix CE + supervised answer CE + oriented
sibling foil loss, fixed warmup 8 updates, AdamW peak .001, weight decay .01
and gradient clipping 1.0; report each component, preclip norm and clipping
frequency. One family per step, same minibatch order per seed across arms
(seeds 1171, 1271, 1371); no data-dependent stopping or tuning on CONTROL.
Each endpoint has 1,024 or 10,240 presented worlds. The 128 endpoint is
not exhaustive FIT exposure and is not directly comparable to historical
EASY. Track exact independent-family exposure and measure fixed endpoint
in-sample competence without confusing it with moving training accuracy.
A loss ablation is separate: if needed, predeclare one otherwise identical
alpha=0 arm, retaining both CE terms. Reduced loss is not evidence of
correct role selection.

Prediction actor sees only token IDs, with exact model/checkpoint binding;
scorer owns the answers, full-vocabulary and two-color metrics. Use strict
rank, failing ties. Primary per-seed numbers: all-eight full-vocabulary
families and full-vocabulary worlds; bounded competence screen proposed at
108/120 and 912/960 respectively, both every seed, threshold frozen in the
admitted recipe. Show paired seed and family contrasts, not pooled seed
promotion. Also report both query orders and clause orders, canonical
margin effects, exact training exposure, panel subgroup with unseen name
pair (9 families), and reference exact oracle separately. The 120 families
are distinct combinations with shared lexical constituents, not independent
domains or untouched language TEST. Report complete training time and
matched CPU inference latency as separate measurements, not a deployment
advantage. If both the wider and budget-matched attention controls fail,
check comparator competence and objective/optimization before attributing
a limitation to the local decoder.

## Engineering checks

Four unit tests on CPU Torch 2.10.0 checked exact trainable parameter counts,
shared trunk initialization, exact intended static intervention and
full-prefix versus truncated-prefix agreement plus mutation of the future
suffix at every cut for each arm. Float32 tolerances atol=1e-6/rtol=1e-5 for
prefix consistency. This verifies batch causal behavior on a fixture, not
production streaming (the attention control intentionally rejects carried state).
Research fits require exact native admission, independent review and
separate score/replay; these unit tests did not train any weights.
