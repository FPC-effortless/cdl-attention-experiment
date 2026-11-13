# Independent pre-fit review — SLOT-CONTROL-001

Fresh review session `/root/review_changing_state`, 2026-11-13.

**PASS for the three specified bounded CPU calibration fits**, subject to the
user's existing research scope. This is an engineering review, not independent
scientific admission, new paid-resource authority or an architectural finding.
No training or stress scoring was performed by the reviewer.

Reviewed freeze SHA256:
`a3143df3d74cac89230be2fa19e80f619554ce74c379cab7f6c82516c1da53a7`.
Exact per-file hashes and runtime are in `initial_model_checks.json` (the name
refers to model checks; it binds the repaired final pre-fit freeze).

## Checks actually run

- All five author tests pass: parameter count, causal prefix/step parity,
  no-admission frame, rejected label argument, fresh-state non-aliasing.
- Separate checker verifies 18,254 parameters and unchanged unselected records
  during an admitted selected-slot update.
- Forward signatures accept token IDs/state/options, no semantic labels.
- Reconstructed answer/destination/admission/0.25-gain loss independently and
  matched the unchanged historical objective exactly on synthetic tensors.
- Reconstructed all three 256x32 index schedules and matched the original
  supervised CSM receipt hashes. Each has 8,192 answer exposures.
- Uninstrumented RecordRewrite and the instrumented historical wrapper produced
  exactly equal logits on a small synthetic random-token test with equal weights.
  No checkpoint or stress stream was used for that check.
- Verified local freeze source bindings.
- Closure validator passed 11 generated temporary-file cases in both normal and
  optimized Python: one valid case accepted; duplicate/missing seeds, early
  scoring, receipt mismatch, incorrect budget/freeze, corrupt checkpoint/log,
  reordered steps and wrong schedule rejected. Source preflight was mocked only
  inside these isolated generated closure tests, not in an actual research run.

## Required repair completed before fitting

The first evaluator draft checked only closure length and a Boolean. It did not
bind seed identity, exposure, source freeze or per-run artifacts. The author
added `validate_closure()` and calls it from evaluation and timing. It now checks
exact seeds, budgets, complete-record equality, freeze/checkpoint/log hashes,
complete step order and historical schedule hashes. The prior freeze and source
commit were retained. Timing's comment now accurately says prediction parity;
the timing routine itself does not compare every logit.

## Scientific and timing assessment

This is a deliberately simple **new learned slot-store control**, not a published
state-of-the-art memory-network implementation. It receives the same physical
identity labels, valid-update labels and overwrite supervision as supervised CSM.
Its token frontend has the same shape, and parameters are close, but initialization,
computation and inductive biases differ. The experiment may show that extra CSM
machinery is unnecessary for this task; it cannot establish a best achievable
architecture, matched training FLOPs or general language superiority.

The timing path excludes hooks in CSM, uses a shared step loop, and includes
encoding, tensor creation, state initialization, all token steps and argmax.
Expected-answer calculation occurs outside timing. Loading/import time is excluded
explicitly. Alternating order and per-cell warmup are useful controls, but three
repetitions on one CPU remain exploratory. The fixed one-query sessions do not
measure reuse across queries, maintenance, restoration, retries, energy or serving
infrastructure. Report only the measured setting, retain raw durations and do not
convert logical tensor bytes into peak memory claims.

All three fits must close before scoring either new diagnostic. The reviewed
workflow must preserve that external execution order: the older stress runner
can technically be invoked separately and does not itself enforce the new slot
training closure. Source/hash checks are reproducibility controls, not hostile
runtime attestation. A final postfit audit should independently recompute metrics
and validate actual closure receipts before relying on results.
