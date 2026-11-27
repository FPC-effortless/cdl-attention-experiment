# COG-EDGE component qualification — developmental checkpoint audit
Scope: passive civilian battery/equipment telemetry only. Not a flight controller or operational safety qualification.

This is a separate branch from expanded-cohort training. Frozen training protocol unchanged. Use only original v0.1 historical pilot archive (149 members, verified by restore.py). No newly trained model or additional independent vehicle is claimed here. No third-party raw data is redistributed.

## Components and admission gates
- Battery: use expanded run 46052547333, retain all seeds/folds, compare per-flight forecast RMSE in volts and 90% coverage plus interval width against tuned classical and neural baselines. Do not promote unless independently reproduced on fresh data, and report heterogeneity.
- Sensor: same expanded run, horizon zero scores only missing voltage targets; other horizons are forecast, not reconstruction. Explicit simulated dropout and outage labels; do not call them observed real failures.
- Memory: use all 18 COG/gated selected checkpoints from three seeds and three folds. Check protected HOLD coordinates, streaming vs batch, preprocessed raw history parity, serialized restart, digest and model-identity rejection. Pass thresholds 2e-5 normalized output tolerance, exact protected-state and identical-input restart; keep every failure.
- Runtime: single-stream CPU eager PyTorch conditional execution of exact weights versus compute-then-mask reference. Warmup, alternate order, seven raw latency repeats, same inputs. Report even slower results; do not extrapolate host timings to embedded board, energy, flight or optimized kernel superiority.
- Health: input-integrity guard for voltage/current observation validity only. Check natural valid values and deterministic injected NaN, Inf, negative voltage, masked invalid inputs. These are input-validation fixtures, not an equipment fault dataset or trained health classifier. Genuine time-stamped component fault labels and independent vehicle validation are still needed.

Actual embedded latency/power, full official SSM/Delta comparisons, fresh-fault labels, external replication and broader independent airframes remain blockers. Preserve source, inputs, original checkpoint identities, every measurement, environment, and CI evidence. No auto-merge or automatic architectural promotion.
