# ROLE-003: doubling the clipping threshold did not fix the weak seed

The 2.0 gradient-norm threshold kept the quarter-weight prefix objective
and changed no model or data. Both thresholds pass the local screen in
only two of three matched seeds. The weakest seed got worse under clip2.

| Seed | Clip 1, 128 worlds / families | Clip 2, 128 worlds / families | Clip 1, 1280 worlds / families | Clip 2, 1280 worlds / families |
|---|---:|---:|---:|---:|
| 1171 | 148 / 0 | 148 / 0 | 953 / 115 | 956 / 118 |
| 1271 | 109 / 0 | 109 / 0 | 954 / 114 | 956 / 116 |
| 1371 | 199 / 0 | 199 / 0 | 842 / 62 | 828 / 58 |

Unchanged gate: ≥912/960 strict full-vocabulary worlds and ≥108/120
strict all-eight families in **every** seed. New synthetic families,
familiar seven-token grammar and vocabulary, no packet store, no
natural-language or inference-cost claim. There is no basis to attribute
a training plateau to clipping from this threshold comparison.

At 128, both arms clipped every update under their respective schedules;
paired checkpoint predictions agreed in top-1 for all 960 worlds in
all three seeds, despite small nonzero differences in logits (maximum
3.46e-5). That observation is consistent with AdamW's approximate
rescaling insensitivity and is not a causal demonstration of it.

For the late weak seed, each contrast has 480 complete pairs. Flipping
the queried name gave 364/480 (clip1) and 348/480 (clip2); reversing
the clause order gave 369/480 and 349/480. Flipping the assignment with
query and order fixed gave 409/480 under either threshold. Full-vocabulary
and two-color correctness agree at the late endpoint. The remaining
failure is which supported value to select, not just whether the output
lands in the right answer vocabulary.

The last 32 pre-update batches for the weak seed show that most answer
CE is within-pair selection under both thresholds: clip1 0.19691 nats
choice + 0.05197 pair-mass shortfall; clip2 0.24487 + 0.06760. This is
the algebraic decomposition of CE, not endpoint accuracy or proof of
why the model fails. Verified counts preserve descriptive subgrouping
(117 familiar/3 novel name-pair families), but the latter subgroup is
too small to support a separate transfer claim.

Native fits: 20261122T165110-24d85a9102a3,
20261122T165117-389e9253b379, 20261122T165125-d1a9a0e9055b.
Independent FIT audit checked 3,840 updates, the first matched objective
and raw norm, actual clipping-input bands, and all six exact endpoints.
Twelve receipt-bound, label-free actors ran on the new 960-world panel.
All 11,520 vectors were sealed before the separate, weight-free
scoring and independently recounted. No protected TEST or MAIN input.

- Actor ledger SHA256:
  d9b6bd3cd2a9a089933c51a14383bc3378d88e837d9d5427e81175d22a826d48
- Prediction seal SHA256:
  a03f10d2ec6b2ef1fee936c031dad2f17ab6d7499f61cc6fb761bb1c72321d12
- Scoring aggregate SHA256:
  8f80309fb58d1f39a7533e8aeb3b28305957c4894e0ccee154470f25fc8b43d2
- Native FIT independent audit SHA256:
  ca0583042e8f92cca0f1a7dec7368a86d1f3a7ea0988f78cc85e3e50d70ea923

Decision: keep the quarter-prefix/clip1 baseline. Do not scale or repeat
threshold tuning. Focus the next falsifier on how a query changes
the selected binding and how that depends on clause order. The measured
gaps make this a target for diagnosis, not an established mechanism.
Keep retention and main training gated.
