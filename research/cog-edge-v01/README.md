# COG-EDGE v0.1 — development, not flight qualification

Stacked on COG-DRONE audit PR #100 at cf4e8f10f907fc7a56c40e52e14e10cc7903f0b1. Scope: offline civilian battery/equipment monitoring (missing voltage reconstruction and 5/15-second voltage forecasts). No flight-control, navigation, physical fault detection, deployment, embedded power or safety claims.

## Preserved evidence

The ten binary Git blobs in archive-parts are contiguous chunks of the **original locally executed** 3,721,355-byte ZIP. Their Git hashes were matched against the original local bytes before committing. ZIP SHA-256: `3d1bc2eeb7efec14a5b2e5c9c08cf8dbb4b65d0171028b7541ca810d0c67f768`. Reconstruct with `python research/cog-edge-v01/restore.py --out /tmp/cog-edge-v01`. This validates the archive and every file against SHA256SUMS.txt. There are 149 archived files: exact model source, data used (pilot.npz), manifest/splits, protocol, logs, all 135 initial/selected/final checkpoints from 45 fits, optimizer/RNG states, metrics and local replay receipts. No regeneration is substituted for custody.

Original gated-ablation streaming repair: all 18 previously failing panels repaired locally with original unchanged weights and batch outputs; maximum streaming discrepancy 4.76837158203125e-7; 655,038 exact HOLD checks; restored outputs exact. Historical negative evidence is unchanged.

New 45-fit developmental pilot (150 optimizer updates, 3 seeds x 3 aircraft x 5 arms, 12 already studied short flight extracts): severe missingness plus artificial outage, equal-fold mean RMSE at 15s: COG 0.2343 V, identical dense ablation 0.2306, GRU 0.2313, static 0.2356, gated 0.2358, lag ridge 0.2722, persistence 0.2685. 90% nominal COG interval coverage 73.27%. Not a win against dense or GRU; 373,248 protected-state checks without changes, 45 exact one-step optimizer-resume tests, 540 replayed metric panels and tested prefix invariance. No official campaign credit or promotion. Reference models, approximate parameter matching, limited training, artificial missingness, three physical airframes, no independent new vehicles.

## Independent verification and next expanded cohort

Workflow restores archive, replays the repair and selected checkpoints, reconstructs the used pilot data from historical archived extracts, and retrains from initialization on the same pilot (separate evidence, not original checkpoint custody). The expanded workflow excludes all 12 historical log paths, retrieves pinned AMOVFLY revision 67069ed00ddbebd62b71aa9bb1272415e9b15ff8, verifies source Git blob IDs and derives causal battery-only 1Hz records. Unknown pack labels are excluded and train/dev/test labels disjoint; label identity is not proof of physical pack identity. Frozen protocol inside archive sets minimum cohort sizes; failure must be recorded. 600 updates, same five arms and three seeds per fold. No test-dependent modifications. Source data redistribution permission not verified: new raw logs and arrays are not committed to public repo; source hashes, extraction/split details and all new checkpoints/results will be retained where jobs complete. Actual full official Mamba/Delta, tuned VAR Kalman forecast, compute equivalence, physical device tests and independent fleet still pending.

Independent remote results are pending until observed; no promotion or merge. 
