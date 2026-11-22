# ROLE-001 duration result: learnable, not reliable

The same gated32 local model, with no packet store, gained substantial local
binding ability from more updates. However, it failed the predeclared
all-seed gate. This is a seven-token, constructed, familiar-vocabulary test
on unseen joint name-pair/color-pair combinations; not natural language,
long-context retention, reasoning in general, or a serving cost benchmark.

| Seed | 128 worlds correct (/960) | 128 all-eight families (/120) | 1280 worlds (/960) | 1280 all-eight families (/120) |
|---|---:|---:|---:|---:|
| 1171 | 12 | 0 | 946 | 112 |
| 1271 | 18 | 0 | 952 | 116 |
| 1371 | 0 | 0 | 547 | 0 |

The frozen screen is 912/960 worlds and 108/120 complete families, both for
every seed. Longer training passed both counts in 2/3 seeds. In seed 1371,
the same checkpoint also got zero of 320 complete FIT families, despite
training across them. It is not merely an unfamiliar-combination failure.
At 1,280, exact and two-color-restricted results agree in every seed; the
failed seed did not meet the role criterion even with supplied alternatives.

Early checkpoints came from the same uninterrupted trajectories, not
restarts. By 128 each had 1,024 world presentations and 128 distinct families
used in updates (tokenizer derived from all TRAIN); by 1,280 each had
10,240 presentations, 320 distinct families, exactly four exposures per
family. The new prompt geometry deliberately differs from the old EASY task.

FIT receipts: seed 1171 20261122T160245-1e36aaf3461b; seed 1271
20261122T160254-beac69309987; seed 1371 20261122T160302-cf5576c5cd00.
Native executed conditions passed for each; receipts are NOT_QUALIFIED.
Independent trace audit reconciled all 3,840 updates, exact schedules and
all six checkpoints. Native separate actor receipts verified for all 12
endpoint/split cases (21,120 full-vocabulary rows), then a separate
weight-free scorer rederived answers from raw text and independently
recounted strict successes. Actor-ledger SHA256
9f53d947e7c81950e52368c2a6f71aee2362551c9d3676cfc011cf707413f970;
prediction-seal SHA256
a3649904c94cfac51627a1b49a97bf5492a1c732c55fc21e699cf7d2948a22b9.

This is an affirmative learnability result for two trajectories, a negative
seed-stability result, and not evidence for or against a parameter-matched
Transformer advantage. The separate static-gate, matched attention and width
arms are the prospective comparison. Do not relax thresholds or proceed
to packet retention on these results.
