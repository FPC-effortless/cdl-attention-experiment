# FROZEN-VALUE-001 — basic value failure localization

14 November 2026. Exploratory follow-up motivated by LIFECYCLE-LEARNING-001's concrete-payload confusions. Freeze its six final checkpoints; no optimizer, retraining, model selection, new architecture or infrastructure changes. This is a diagnostic on known keys/values in new history distributions, not a held-out architecture benchmark.

Run all 32 combinations of four keys and eight values in four panels:

- single: SET target value, then QUERY target.
- filled_last: SET the other three records, then SET target value, then QUERY.
- filled_first: SET target value, then the same three background writes, then QUERY.
- overwrite: the same background writes, SET target old, SET target new, then QUERY.

Assign background values new+2,+3,+4 modulo8 to the sorted non-target keys; old=new+1 modulo8. Thus new, old and all background values differ. Background assignments match across the three filled panels. Keep the original 27-token encoding with NOP padding to eight events plus QUERY. The filled panels have identical authoritative final maps. Single versus filled_last changes occupancy, history and context jointly, not occupancy alone. First versus last is an order/retention-context contrast. These limitations must remain explicit.

128 fixed cases, one declared diagnostic subject, released through existing Forge with a fixed salt. Forge's TRAIN partition is bookkeeping for this complete, correlated probe grid; no fit occurs and no holdout/independent-task claim is made. The executed assay does not open historical third-bucket examples or protected benchmarks. An initial preparation, superseded before inference, hashed historical dataset bytes through the parent verification helper; it did not parse or score examples. The revised verifier checks approved parent source, metadata, receipts and checkpoints only.

For each checkpoint (CSM/pointer-slot ×seeds101/202/303), run normal token-only inference and the existing privileged supplied-control diagnostic. Exactly1,536 predictions. Report per-panel/per-seed counts, per-key and per-value errors, paired prediction changes, stale-old answers on overwrite, raw categorical-head errors, answer-logit margins and query-state preservation. Supplied controls also force full gain and are not learned success. Labels come from the independent Veritas state semantics and are cross-checked by the historical backward oracle. Freeze source, cases, parent closure, checkpoint identities and runtime before inference. Preserve failed attempts; no adaptive panel revision after outputs.

Question/falsifier: if even simple SET→QUERY fails with correct supplied controls, lifecycle composition is not necessary for that checkpoint's observed errors. If single succeeds but matched filled/overwrite panels fail, history/context or update handling matters. Neither result alone distinguishes representation loss from a learned decoder failure. Class-specific failures on these probes do not prove a globally unlearned class.

Use existing Forge → TrainLab → Veritas → Research Lab interfaces. One bounded inference command, CPU, timeout120seconds, zero fits, no cost comparison. Independent pre-execution review and post-execution rescoring/native replay. This does not amend the earlier fixed training runs or relax their unsuccessful reliability/qualification screens.

Pre-inference code-derived expectation: with supplied controls and full gains, pointer-slot target logits should match across single, filled_first and filled_last (same zero target row and SET path). Record full logits and maximum absolute paired differences; allow floating-point tolerance 1e-6. Overwrite may differ because its proposal sees the old target representation. CSM operator/controller context need not be invariant. This is a mechanistic diagnostic, not an architectural advantage.
