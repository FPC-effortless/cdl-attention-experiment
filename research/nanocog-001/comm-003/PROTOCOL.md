# ROLE-003: change only global clipping threshold

In ROLE-002, gradient clipping was common and quarter-weight prefix loss did
not qualify across all seeds. Clipping as a cause of failure is unproven.
This follow-up tests one lever: raise the max L2 gradient norm in
`clip_grad_norm_` from 1.0 to 2.0. Keep the ROLE-002 gated32 model (37,414
parameters), FP32 CPU, initialization, three seeds (1171, 1271, 1371), exact
TRAIN bytes, family shuffle, 1,280 updates, 128/1,280 endpoints, raw gradient
accounting, AdamW hyperparameters and learning rate, and .25 prefix CE +
answer CE + paired foil softplus. No warmer restarts or new data exposure.
Compare frozen ROLE-002 clip1 checkpoints; do not refit them. Each fit has
one serial thread, one attempt, 180 s native cap; stop on invalid evidence.
No paid compute, historical/protected TEST, retuning, or remote writes.

## Prospective scoring

Primary is strict full-vocabulary 1,280-step accuracy. Ties fail. Require
at least 912/960 strict worlds and 108/120 complete eight-world families
in **each** seed. Prefer clip2 as a repair only if clip2 passes every seed,
clip1 does not, and clip2's worst-seed complete-family count exceeds
clip1's worst seed. If both pass report both without claiming repair
necessity; if neither passes report the failure. Score 128-step outputs
descriptively, never choose them as alternate qualification.

Report every paired seed and family contrast, effect by query/order,
two-color versus full-vocabulary errors, familiar/unfamiliar name pairs,
and prefix/answer/foil terms and raw-gradient/clipping counts over time
for each arm. Distinguish raw norms above thresholds from actual clip
coefficients near epsilon boundary. The same initialization means the
first pre-update raw gradient norm/objective should agree exactly, before
clip treatment; later gradients differ with the trajectory. A result can
identify a tested threshold effect with these seeds, not mechanism,
gradient conflict, language quality, persistent memory, or inference
speed. Compare costs only if separately measured on an equivalent contract.

## Fixed synthetic evaluation, prediction and truth separated

Canonical sorted 16-name×8-color unordered pair product, removing all
320 FIT, 120 ROLE-001 CONTROL and 120 exact ROLE-002 confirmation family
keys. Seed Python `random.Random(20261124)` to shuffle the remaining keys;
first 120, no replacement. Full 2^3 assignment/statement/query factorial
per family, four of each answer, 960 unique seven-word actor prompts.
Class marginal frequencies are not balanced overall. Familiar vocabulary/
syntax only; 117 familiar and 3 unfamiliar name-pair families, all color
pairs familiar. No out-of-domain inference. No rows, labels or past
outcomes are used in fitting. Freeze actor, judge and analysis before any
prediction; only join truth after the accepted predictions are sealed.

Actor SHA256 `536ba918ab5122549cd1142df0c02789fb406731a2fdf4928db75ac04a1c1555`.
Judge SHA256 `edade4df767fff4ce0321d09c4e5511514f3944e8ea5f52077539402b7f4de84`.
Panel receipt SHA256 `26e4fc6f7e79adf8d4fc55091c02a10c45456de8b8b594e07d96461017ee1a8f`.
