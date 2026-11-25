# ROLE-GATE-003: all weak errors were reachable, but no fixed gate repaired them

The preceding matched-dose sham lacked bidirectional support, so no
directional comparison was run. This test asked a different question:
whether an intervention restricted to the final gate can recover existing
errors at all. It exhaustively applied a **fixed finite grid of 11**
legal mixtures to **every** prompt on the same exposed 32-family panel,
on two exact frozen checkpoints. No answers or predictions influenced the
actor. Only the four weights of the final second-layer local gate were
changed; all upstream paths and learned tensors were unchanged.

| Strict full-vocabulary metric | Strong 1171 | Weak 1371 |
|---|---:|---:|
| Native correct worlds | 255/256 | 223/256 |
| Best fixed choice on this same panel | 167/256 | 130/256 |
| Native error worlds recovered by **some** grid choice | 1/1 | 33/33 |
| Native correctly answers both queries, same history | 127/128 | 95/128 |
| Retrospective oracle, native or **one shared** grid choice per history | 127/128 | 97/128 |
| Native complete families (all eight) | 31/32 | 16/32 |
| Oracle with one grid choice per family, or native | 31/32 | 16/32 |

All 33 weak errors were on questions about the second clause. The
best single fixed choice for the weak checkpoint, a near-equal
mixture of lag2 and lag1, rescued 26 worlds and damaged 119. Its 130/256
overall is much worse than the native 223/256. A different fixed
choice was highest under conditional two-color scoring (145/256, still
below native). Under strict scoring there was no single tested choice
that repaired the whole panel, even when selected retrospectively.

**What this establishes.** The existing upstream representation and
downstream readout are sufficient to yield the correct top-1 answer on
these exposed errors **for some** final gate choice. That is a causal
reachability result, not proof of why training failed, that there is a
shared optimum, or that a learned controller can choose the right gate.
The per-world oracle chooses after scoring and is never a model result.
A single fixed intervention cannot substitute for a query-dependent gate.

**Next proposed falsifier.** Keep the backbone frozen and test whether a
small adapter that compares the current query representation with each
available lag representation can improve strict binding on fresh
combinations, against a matched query-only adapter. Supervise the
answer on TRAIN only, with no oracle inference features. Require exact
initial native parity and report collateral damage on passing seeds.
This experiment is not yet fit or admitted. The main/retention gate
and all-seed qualification thresholds are unchanged.

## Custody and limits

Two native TrainLab fit-free calls (one per frozen seed):
`20261122T173618-57887e9441a3`, `20261122T173621-632799071225`.
Total 896 small forward batches, **zero** optimizer updates. Native
outputs were bit-exact to the corresponding previous probe at all
146,944 logits. Both runs checked no-op and replay parity, all earlier
positions, prescribed probability support, and frozen weight immutability.
Both complete 12×256×287 float32 tensors were sealed before judging
(6,144 full-vocabulary rows across the two runs). No protected TEST,
MAIN, paid compute, or new training. Bounded synthetic diagnostic only.

- Exact ledger SHA256: `d00ab48c52a3ba4557b21e40a12e436dabd1aa31d459d453e4430cb737cbb8c2`
- Prediction seal SHA256: `770686ff7de4fc2174d16312f3771ba0263fe7373d898ed0195f18fbe37d3a41`
- Results SHA256: `e986446bd59f0bb41a733a5bcc6a4d7dbf7b76db0bf768ceec8a1ec510980187`
