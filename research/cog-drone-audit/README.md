# COG-DRONE-001 and 002–006: reproducibility hold

The earlier work is developmental. No new fit, promotion, flight control, deployment, or official research-campaign credit authorized by this audit. Passive voltage/current equipment-monitoring only.

## Original evidence identities

- COG-DRONE-001-evidence.zip: SHA-256 46c0c7b41e76b4f561df18a76da7dd74b3ed49ec76743725884932de778d675b, 597978 bytes, 42 files, 30 checkpoints.
- COG-DRONE-002-006-evidence.zip: SHA-256 80ac5e498a3927ba1c07deb7629e11ce703e79b4eebb8ef3e3be485063754ad2, 536400 bytes, 97 files, 81 checkpoints.

The original bytes exist in the local evidence archives. The repository connector accepts text/base64 content but the local runtime lacks authenticated network access or a direct binary-upload bridge. Until the archives appear in archives/ and pass the custody gate, do NOT describe the original weights or evidence as preserved in GitHub. The data source is public but no redistribution license was established; obtain permission for any full dataset redistribution. The source manifest and exact extraction procedure enable verification of the 12 excerpts without republishing the entire dataset.

## Local read-only audit receipt

Python 3.13.5, Torch 2.10.0+cpu, NumPy 2.3.5. Both inner manifests matched: 41 and 96 entries. All 111 checkpoints replayed 252 neural test panels with zero maximum metric discrepancy, including per-flight/trajectory error. All 18 real-data classical panels replayed. 144 real-data recurrent batch-vs-step panels were checked; 18 failed, all Dual gated evolution ablation (max absolute output error range 0.07560718 to 0.43576002). Its forward gates shared state but inherited step does not. Do not silently repair historical weights or conflate batch and step behavior. Protected primary arms: 1,033,662 scalar checks, zero violations. Local receipt is not GitHub-hosted replay.

## Scientific admission

12 selected short excerpts / three aircraft / six train logs per fold; repeated seeds and masks are not independent flights. Artificial independent missingness; warmup excluded, shared engineered causal last-value/age features, validation-selected 160 updates. Baselines not compute matched; Mamba and gated delta are reference recurrences. Static MLP and VAR Kalman competitive/better. Uncertainty undercoverage; no physical board or power measurements. Selected model weights and normalization/calibration metadata survive; optimizer, scheduler, RNG and full raw logs were not captured and cannot be recovered by re-labeling a new fit. No end-to-end inference-cost comparison at matched quality.

Gate before any later research: byte-exact original archives and independent runner replay; source hash checks and disjoint splits; preserve failing diagnostics; resolve stream discrepancy in a separately versioned repair; define representative flight/airframe/pack splits, mask regimes, competent tuned baselines and matched compute, held-out evaluation with independent-unit statistics, calibrated uncertainty, actual device metrics. No subsequent cycle until reviewed; no automatic merge.

Workflow: independent source provenance and archive/replay jobs. A red custody job is intentional if originals are absent. Never substitute a regenerated weight for an original.
