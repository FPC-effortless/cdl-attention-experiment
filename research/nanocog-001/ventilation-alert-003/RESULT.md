# VENTILATION-ALERT-003: Late severity changes under sensor drift

**A fixed drift-corrected rule beat every learned arm under drift.** On the hardest late-change panel, the rule got both worlds right for **232/288 pairs (80.56%)**; tied COG led the learned arms at **582/864 paired decisions (67.36%)** across repeated seeds. No arm advanced.

## Design

Constructed, offline building telemetry. Each pair shares the exact first 12 readings, censoring masks, change time, reference readings and nuisance draws; the two worlds diverge in their later environmental reading and final severity. The separate reference instrument measures shared drift imperfectly and is equally observed by all arms. Actor input contains only 24 × 4 values/masks, not pair labels, change time, drift truth or IDs. The held cohorts contain the same 288 scenario families, rendered separately. Compound stress includes later change, missingness and unequal observed reference gain. Both final ticks are unavailable in that cohort; a last-row-only model has no usable input there.

Seven arms, three paired optimizer seeds, 512 fresh FIT pairs, 192 fixed AdamW updates at batch 32, and the same per-seed draw order. The FIT mix contains clean, common-drift and drift-plus-dropout pairs; late compound is absent from FIT. Fixed thresholds for all controls. No parameter promotion. Exact pair correctness, not merely changed prediction or marginal accuracy, is the main measure.

## Complete pairs
Learned rows use 864 repeated decisions (288 families × three seeds) per cohort. Rule rows use 288 families once. They are not independent family counts across seeds or cohorts.

| Arm | Clean | Shared drift | Drift + dropout | Late compound |
|---|---:|---:|---:|---:|
| Last-row MLP | 85.19% | 84.38% | 47.92% | 0.00% |
| GRU | 96.88% | 96.76% | 89.47% | 66.67% |
| Tied SDPA | 96.41% | 95.37% | 89.81% | 62.96% |
| Tied COG | 95.60% | 94.56% | 90.05% | 67.36% |
| Shared span ×1 | 94.10% | 89.47% | 85.19% | 64.93% |
| Shared span ×4 | 90.86% | 87.85% | 84.38% | 63.54% |
| Native TAC observe | 92.94% | 91.20% | 84.14% | 54.63% |
| Early-only threshold | 0.00% | 0.00% | 0.00% | 0.00% |
| Late single-instrument mean | 100.00% | 9.38% | 8.68% | 9.03% |
| Late median of measured difference | 95.49% | 95.49% | 95.14% | 80.56% |

COG's compound advantage over tied SDPA was +4.40 points descriptively (paired family/seed bootstrap –5.67 to +14.70); this does not distinguish them reliably. COG was 13.19 points behind the drift-corrected rule (interval 5.38 to 20.08 points behind). For severe late-compound outcomes, COG under-alerted on 121/576 repeated decisions, tied SDPA on 74/576, and the drift-corrected rule on 13/192 families. The rule also misses the 2% screening target. TAC's earlier lead on corruption of an otherwise fixed severity did not carry to this changed target and sensory contract.

The early shortcut is closed by construction: any deterministic prefix-only model must output the same label for both worlds, so it scores zero complete pairs. Sharing the measured channel difference explicitly with every model, then testing noise, witness loss and calibration shifts, can distinguish failure to learn the subtraction from genuine temporal generalization failure. Keep the fixed rule as the baseline, not as a verified field policy.

## Evidence
Native TrainLab: 21 fits, 44.790768 s, EXECUTED_CRITERIA_MET and NOT_QUALIFIED. A separate endpoint implementation matched 84 seed/architecture/cohort cells. Rebuilt data are byte-identical. DATA c664632be57867866ae594887729a7f620d02482275fa938b57b45ecebb40457; LOCK bef1d0fef115b6792c36f13fabef7e62028871f292b0b32b8ac649d5c3a0e0e0; native 53ead62bc9feae495c065b87cd911a89aa9be5ad87af796befc2f6b3bb43cf20; SCORE 24e8253448af2b028380b724584421693198f030cd24571032b881e369bfae26; recount cc4d5315b0b0acc80bc407da4c4cf3cbdd252414fde766652ae8878e164bde9d.
