# ROLE-002: quarter-weight prefix loss improves the worst seed, but does not qualify

A one-variable change (prefix CE weight 1.0 to 0.25) strengthened the weak
seed without adding parameters. It still did not meet the unchanged
gate. This result is for known vocabulary and one synthetic grammar,
without a packet store, not for language quality or serving cost.

| Seed | Original, 128 worlds / families | Quarter, 128 worlds / families | Original, 1280 worlds / families | Quarter, 1280 worlds / families |
|---|---:|---:|---:|---:|
| 1171 | 20 / 0 | 167 / 0 | 946 / 112 | 948 / 111 |
| 1271 | 16 / 0 | 137 / 0 | 937 / 108 | 956 / 117 |
| 1371 | 0 / 0 | 170 / 0 | 536 / 0 | 851 / 70 |

Denominators: 960 strict all-vocabulary worlds and 120 complete
eight-world families. Both arms pass in only 2/3 seeds; required each seed
≥912 worlds and ≥108 families. Worst seed 1371 gained 315 worlds and 70
families, but still misses by 61 worlds and 38 families. The gain appeared in
both name-pair strata (familiar: 0→65/110; unseen pair: 0→5/10). Early
color-only accuracy stayed near half and no seed completed a family;
the early output improvement alone does not show role learning. Late
all-vocabulary and two-color counts agree, so a shortlist does not fix
the remaining errors.

Training matches the old control on initial tensors, schedule, data,
parameters and runtime. Native fit IDs: 20261122T164450-31e421e7a5cb,
20261122T164459-6869c6f8cb98, 20261122T164507-ffa625aebb76. An independent
audit reconciled 3,840 trace rows and six new exact checkpoints; the
actual quarter-weight formula matched within 4.77e-7 nats. Frozen original
checkpoints were reused; no control refit. The 120 fresh joint combinations
are disjoint from every ROLE-001 fit/control family. Twelve native actor
runs were verified and sealed; 11,520 vectors were scored with all 287 IDs
and independent recount matched all 252 scalar counts. No protected or
MAIN split was accessed. No paid compute or new retention run.

- Evaluation lock: ead0efa5743ee3e01976a217e499ba5c32bfe27270d189c5dc4e4a6dfb13ad01
- Actor ledger: 2f28213c2eedcbdfce2151a42250ba57b0ca48bd92a2119726dc10818679ad25
- Prediction seal: e6a31a5ceaecb1ee296289f4f38a63d541f44de6e0508c02ad55ac9542e056f8
- Official aggregates: a6c354d69ee3ac3583cab31125404987fe510c22e3e282d93325ab0a04b390cb
- Independent FIT audit: 92d9bbf3b1b4643236be235fa7c9292dcceb29de7aeb65eca5a00ce3e2cf880c

Decision: preserve the objective improvement as the next reference,
not a successful repair. Test one new training variable on a new panel,
measuring actual gradient norms and clipping frequency. Keep the gate,
the simpler controls, and main/retention hold.
