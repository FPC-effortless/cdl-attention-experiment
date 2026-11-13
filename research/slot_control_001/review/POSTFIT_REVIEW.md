# Independent postfit review — Changing State and Slot Control 001

Fresh review session `/root/review_changing_state`, 2026-11-13.

**PASS for the saved development results and sampled native replay.**
This is an independent reviewer-agent audit in the same execution environment,
not external replication or independent retraining. No fit or timing was run by
the reviewer. CPU replay ended before the author's timing began.

## Verified evidence

- All three slot-store fits used the reviewed source freeze, 256 updates and
  8,192 answer exposures each: 768 updates and 24,576 exposures total.
- Closure records match individual completion receipts. All training log and
  checkpoint hashes match; step sequences are complete; losses and recorded
  gradient norms are finite. Schedule hashes match original supervised CSM runs.
- Checkpoint seed/update fields and every saved optimizer step equal the intended
  final run. Source/data freezes, original dependency hashes and original
  checkpoint identities were verified.
- Independently reconstructed semantic truth and immediate stale predecessors
  from every world's write history; verified every prediction's answer, changed
  flag, prior-value label and correctness flag.
- Independently recomputed **all 35,328 saved prediction records** into **138
  scored metric cells**, plus **30 deterministic control cells**. No missing or
  duplicate query record or metric discrepancy was found.
- Original historical Transformer has 12 explicitly unscored long-panel cells.
  Their absence is an evaluator scope choice, not a failed prediction.
- Replayed **1,728 predictions** natively from frozen checkpoints: all four queries
  for eight shift-zero block representatives, both regimes, depths 1 and 256,
  all three seeds for raw CSM, supervised CSM, supplied-controller CSM and slot
  store. Historical Transformer replay used depth 1 only. Every prediction
  matched the saved record; weights remained unchanged.

The full metrics audit and sampled replay took about 21 seconds. That is an audit
receipt, not a benchmark. Exact records are in `postfit_checks.json`, with a
read-only supplemental original-dependency check in `postfit_dependencies.json`.
The audit script was subsequently given that same dependency check for future
invocations; no model, result, prediction or metric was changed.

## Verified result and interpretation

Supervised CSM and the equally supervised learned slot store each obtained 100%
answer accuracy and 100% complete-world accuracy in every scored development
panel in all three seeds. The supplied-controller diagnostic was also perfect.
Raw CSM remained weak, and the historical Transformer remained weak on its scored
short panels. Those historical arms have less training information and are not
fair architecture-superiority comparators.

This establishes a narrow negative discriminator: these four-identity SET,
replacement and preservation tasks do not require the extra CSM operator/link
machinery to reach perfect performance under the tested supervision and budget.
It also confirms a narrower positive behavior: CSM's learned selective-write
reference survives repeated SET updates through the tested depth of 256.
Neither result establishes learned deletion, COPY, live-link use, larger active
state, natural-language parsing, open-ended streaming or superiority over tuned
strong alternatives. Timing is audited separately if performed.

## Limits retained

There are eight independent base stream blocks, each with eight correlated value
shifts. Regimes, depths and four independent query replays are paired/correlated;
depth-one regimes duplicate the same histories. Raw query count is not an
independent sample count. These are public development data, not fresh sealed
confirmation data. The neural replay is sampled, although the saved prediction
metric reconstruction is complete. Aggregate controller-route counters were not
independently replayed in full and should not be described as such.

The fresh-directory replay helper was source-inspected. It requires a new output
path, validates the exact original archive hash and archive member paths, copies
trained evidence, excludes existing results, and invokes only frozen evaluation
(and optional timing). The full helper was not rerun during this review.
