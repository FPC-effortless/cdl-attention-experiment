# Reproduction boundaries
This package contains a bounded constructed read-only advice protocol and CPU service comparison. It does not include generated data, fitted models, held outcomes, row-level clocks, a launch review, or proof of production use.

- `python test_gateway.py` and `OPENBLAS_NUM_THREADS=1 python test_batch.py` use hand-authored fixtures only. They do not train a model or score held data.
- The experiment used the exact eec439492407b0aa26922c23b5ae40d95289f4eb Veritas integrations, Python 3.12.14, NumPy 2.3.5 and OpenBLAS 0.3.30 with one observed thread. Match the pinned ecosystem before attempting a separate reproduction.
- `freeze.py` is the intended data constructor, not `plant061.py`, which preserves the original upstream simulator source. Source histories, auxiliary Forge views and evaluator truth are separate.
- The original fit, prediction and score stages each had a different exact reviewed lineage and no native retry. `native.py` is stage-selective and requires the corresponding independent approval; this document is not one. Truth is excluded from fit and prediction. `independent_check.py` recomputes comparison/authority outcomes from retained artifacts only, never running a learner. If optional host-clock observations differ, report the new evidence as separate, not replaying deterministic timing.
- Do not infer authority, authentic interaction, learned runtime, sparse acceleration, external actions, package qualification or production performance.
