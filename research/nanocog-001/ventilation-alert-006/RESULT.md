# VENTILATION-ALERT-006: fresh calibration check vs shadow sensor

**Scope: independent calibration evidence in synthetic offline telemetry, not a new architecture or a real sensor validation.** In stale/offset cases, the unassisted subtraction rule scored 86/288 complete pairs and missed 55/192 severe worlds. Adding the noisy reference check gave 262/288 complete pairs and 3/192 misses.

## Design

42 one-thread fits: seven adapters, three seeds (173, 271, 907), two matched views, 192 optimizer steps and batch 32. 512 disjoint FIT families; 288 novel HELD families replayed across four age and calibration cells. Both histories in a family share identical early observations and calibration checks, but differ in the correct alert. The check measures the witness's zero offset via a separate synthetic known-zero observation, with independent integer error ±2 per late tick. It does not measure target concentration or common drift. The matched shadow sensor has the same generative distribution and noise realization but its own independently drawn signed bias in stress cells. It still signals whether the calibrated or offset regime applies, so it is not equivalent to withholding all calibration side information. Zero-offset evaluation rows are exactly identical between views; differences there reflect training on different evidence elsewhere.

No mechanism used held labels, changes after the last sample, real hardware, or control. The seven models have matched input shapes, row draws, training budgets and same initial weights within seed/arm. Parameter envelope 9,000–13,000; not FLOP matched. Same-family-and-seed bootstrap, 3,000 draws, descriptive unadjusted intervals. Full-history finite-panel ceilings are 288/288 in every cell and each view; identical prefix inputs cap early-only success at 0. This does not establish population learnability.

## Terminal counts

Each cell gives complete pairs / severe under-alerts / false severe alerts. Models: /864 pairs, /576 severe worlds and /1,152 nonsevere worlds, pooled over three repeated seeds. Model arrows show shadow → real check. Rules: /288, /192 and /384 respectively.

| Arm | Recent calibrated | Recent offset | Stale calibrated | Stale offset |
|---|---:|---:|---:|---:|
| Last-tick MLP | 295/105/476 → 333/214/240 | 187/150/528 → 307/229/253 | 0/192/768 → 0/384/384 | 0/192/768 → 0/384/384 |
| GRU | 753/4/82 → 789/8/40 | 434/73/210 → 754/18/51 | 714/12/88 → 776/30/22 | 442/95/171 → 745/32/27 |
| Tied SDPA | 717/8/88 → 746/25/44 | 459/80/179 → 699/36/43 | 751/12/52 → 752/36/15 | 452/113/132 → 741/34/18 |
| Tied COG | 725/23/63 → 743/14/65 | 408/108/169 → 712/26/57 | 656/50/66 → 735/8/87 | 363/195/112 → 712/19/66 |
| COG span ×1 | 750/27/66 → 766/19/33 | 402/93/207 → 740/20/47 | 605/59/103 → 680/59/36 | 410/123/175 → 656/51/56 |
| COG span ×4 | 730/21/84 → 761/12/40 | 408/84/218 → 743/18/55 | 581/33/124 → 564/108/9 | 425/104/182 → 552/91/22 |
| TAC-OSM observe | 733/4/96 → 783/9/37 | 471/83/167 → 772/12/37 | 697/39/57 → 783/24/17 | 434/143/127 → 764/29/15 |
| Late primary only | 61/79/78 | 61/79/78 | 61/79/78 | 61/79/78 |
| Primary − witness | 267/3/5 | 86/55/74 | 267/3/5 | 86/55/74 |
| Primary − witness + shadow | 262/3/10 | 113/56/66 | 262/3/10 | 113/56/66 |
| Primary − witness + check | 262/3/10 | 262/3/10 | 262/3/10 | 262/3/10 |

## Stale with biased witness: paired change from real versus shadow check

Points, with descriptive 95% intervals. Negative changes in either error rate are improvements.

| Arm | Complete pairs | Missed severe | False severe |
|---|---:|---:|---:|
| Last-tick MLP | +0.00 [+0.00, +0.00] | +33.33 [+0.00, +96.88] | -33.33 [-97.92, +0.00] |
| GRU | +35.07 [+28.70, +41.44] | -10.94 [-20.14, -2.95] | -12.50 [-17.53, -7.03] |
| Tied SDPA | +33.45 [+25.46, +41.32] | -13.72 [-23.44, -5.20] | -9.90 [-14.15, -5.64] |
| Tied COG | +40.39 [+34.49, +46.18] | -30.56 [-54.18, -13.19] | -3.99 [-12.41, +7.03] |
| COG span ×1 | +28.47 [+17.71, +38.66] | -12.50 [-33.16, +2.26] | -10.33 [-19.01, -2.78] |
| COG span ×4 | +14.70 [-7.65, +33.56] | -2.26 [-16.15, +8.68] | -13.89 [-27.86, -4.08] |
| TAC-OSM observe | +38.19 [+32.06, +43.98] | -19.79 [-29.34, -10.76] | -9.72 [-14.58, -5.47] |

**Gate:** No learned arm qualified. Qualification required ≥85% complete pairs and ≤2% severe misses in every cell, within two points of the same-evidence check rule everywhere, and at least two points better in one stress cell. The simple matched-evidence rule is evaluated separately, not promoted as a learner. This is evidence about whether sensing additional calibration information helps the tested decision, not about the universal ranking of architectures.

## Reproduction

TrainLab reported EXECUTED_CRITERIA_MET; 42 fits took 92.378s (not serving latency). Two fixture builds matched byte-for-byte. A separate detached scorer recounted all 168 terminal model/seed/view/condition cells; not an independent operator replay. Private replay includes source, generated observations, detached targets, raw terminal evidence and integrity hashes.

- LOCK.json SHA-256: 4588b8b38b3f79b92a295a7d367a58ac59cc2426488334ea24c9d2dd48a8c564
- DATA.json SHA-256: e88d7f29698c19cf263745b8eb80366fb136ed984693d37f869749c688315487
- run.json SHA-256: b77d24536a254b5602ded01193a8642b44ef858f75426f400c1a3bb8affea1a4
- SCORE.json SHA-256: 6ef015af21da431cb2858d97afce9042ef4f1c42537dbc8ec02301a1f745ba45
- RECOUNT.json SHA-256: 7403bca2e7a4d0749fae4c40c75f5dd830d41297d88e1bca5c486600a3a53907
