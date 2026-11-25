# VENTILATION-ALERT-004: Does supplying the measured correction close the gap?

**No.** The same observable difference helped some learned models but hurt others. On the hardest paired cohort, the fixed drift-corrected rule remained best, at **228/288 complete pairs (79.17%)**. The strongest learned result was tied COG with the matched raw input, **648/864 repeated decisions (75.00%)**. Supplying the difference reduced COG's mean to **635/864 (73.50%)**; the paired interval included no change.

## Matched ablation

Offline synthetic building telemetry. A fresh deterministic draw of the previous late-change generator: 512 FIT pairs, 288 held families each rendered across four stress regimes, matched early history within pairs. Both arms retained both observed sensor readings and masks. A matched fifth and sixth field supplied the primary reading again or its difference from the witness reading, only where both were measured, plus the same joint visibility mask. Neither arm received any new physical sensor signal or hidden label. Both used the same FIT-only normalization rule, producing different feature scales. All 21 raw/difference matches had identical starting weights, parameter counts, draw hashes, task shapes and schedules (192 fixed updates × batch 32). No held tuning.

## Late compound: both worlds must be correct

Each learned cell contains the same 288 families evaluated under three optimizer seeds; six native measured fields are the same shape in both views. The severe-under column counts decisions out of 576 (192 families with one severe world per seed). Bootstrap intervals are descriptive, paired over held families and seeds; they are not multiplicity-adjusted.

| Architecture | Raw pairs / 864 | With difference / 864 | Change (points; descriptive 95% interval) | Severe under-alerts raw → difference / 576 |
|---|---:|---:|---:|---:|
| Last-row MLP | 0 | 0 | 0.00 [0.00, 0.00] | 192 → 0 |
| GRU | 561 | 600 | +4.51 [+1.50, +7.93] | 97 → 70 |
| Tied SDPA | 547 | 558 | +1.27 [−1.74, +4.28] | 84 → 68 |
| Tied COG | 648 | 635 | −1.50 [−5.27, +1.85] | 60 → 72 |
| Shared span ×1 | 554 | 502 | −6.02 [−10.76, −1.74] | 29 → 47 |
| Shared span ×4 | 560 | 513 | −5.44 [−11.00, 0.00] | 42 → 37 |
| Native TAC observe | 465 | 500 | +4.05 [−1.04, +10.30] | 154 → 98 |

The last-row MLP sees the same all-missing final row in every compound world, so it cannot solve a pair. Its zero severe misses with the added feature do not signal good classification; a fixed severe prediction can avoid all such misses while still failing every pair. The fixed measured-difference rule under-alerted on 12/192 severe cases in this cohort, also above the 2% advance floor. No learned arm met the prospective gate.

The explicit correction is useful but insufficient. This is a feature and optimization intervention, not a proof about whether a network learned subtraction internally. Nor do the rest of the errors identify one specific temporal deficiency. The next discriminating experiment would independently manipulate the age of the late evidence and the calibration of the witness sensor, while testing measured-difference aggregation and the same raw input controls. This prevents missing recent readings, a regime shift and sensor calibration from being treated as one unidentified difficulty. No field conclusion or deployment guidance.

## Execution

Native TrainLab completed 42 fits in 82.389234 s: EXECUTED_CRITERIA_MET and NOT_QUALIFIED. A separate endpoint counter matched 168 arm/view/seed/cohort cells. The dataset rebuilt byte-for-byte. DATA a54daec8ab281df7fea2e3892c2c8860bd88b0d2d41b644c99693c12c2e46b40; LOCK 4f66dd560e1bae8dfdd4300c9a3f86b0621c2d6cf7f2147b5e9dc7ef5b8a0d70; native f695edd4aa7a69077f37d0b4121e7cb0023cde0470a856c1f0589f0c6021c627; SCORE e61fdb0e7145fd145fec6ea3b3b4e1efd15dfa4e14f54e6f158a29ba4dbfcd04; recount f72827d67ea0a9fd27aa4e61defc79247bfd5731d6cec3df7dc18e4dffc3a904.
