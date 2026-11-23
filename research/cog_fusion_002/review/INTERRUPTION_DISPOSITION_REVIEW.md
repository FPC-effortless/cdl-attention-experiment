# Independent disposition of the unresolved recovery

**Review time: 2026-11-23, 23:13 UTC. Disposition: COG-FUSION-002 is incomplete. No new retry or held scoring is admitted by this note.**

## Established evidence

The [original campaign exit receipt](../evidence/CAMPAIGN_EXIT.json) records five zero exits and a return code of `-9` for `continuous_seed17`. The five other endpoint receipts report `completed` at 1,200 updates, and their final and optimizer files exist. This review checked those receipts and file presence; it did not load their tensor payloads or certify their full integrity.

The [declared recovery](../evidence/RECOVERY_PROTOCOL.json) allows one retry per interrupted job and expressly excludes a second launch or implicit resume even if recovery itself is interrupted. The [recovery start receipt](../evidence/RECOVERY_STARTED.json) identifies only `continuous_seed17` for replacement. The original exclusive marker and policy remain authoritative records of what was attempted.

At 23:13:18 UTC, the [replacement trace](../evidence/runs/continuous_seed17/train.jsonl) contained 107 complete rows, last step 107, and a recorded training-loop CPU lower bound of 197.215294733 seconds. Its SHA256 was `22b81773bdfc6a5c692c93e9b0eef50450c39c4383be7026e1ef49bd0cf7ffe9`; its modification time was 23:07:36.308514 UTC. No `RECOVERY_RESULT.json` or evaluation lock existed.

The runtime reviewer observed `Unknown process id 80197` from the supported session-control call at approximately 23:11 UTC. That observation exists in the tool transcript; there is no corresponding OS exit or liveness receipt. An unavailable control handle, a stopped log, and absent completion artifacts do **not** establish process termination, its cause, or an out-of-memory event. Per-execution PID namespaces prevent the current supported session from inspecting the old process namespace. No namespace escape or process signaling is justified by this review.

## Present disposition

Preserve the original policy, start marker, logs, failed-attempt archive, and completed endpoint artifacts. Do not synthesize a recovery result, delete the marker, reinterpret the second attempt as an unstarted first retry, or weaken the original all-six admission gate. A packaged copy is a dated evidence snapshot; it must not imply that a potentially unresolved writer has been proved dead.

A separately labeled, read-only integrity audit may inspect the five completed endpoints. It should verify their raw checkpoint/optimizer identities, fixed update counts, finite contiguous logs, Python draw/RNG correspondence, inactive-board invariance, and known input/source hashes. Its result must say **five completed endpoints audited; campaign incomplete**. It must not invoke the original all-six audit as though its admission passed, score held panels, calculate the three-seed qualification, or select among endpoints using learned performance.

Publish the complete source and retained evidence with this limitation. There is no qualified 002 learning-effect result to report. The interruption neither establishes nor refutes the architecture's adequacy. If trustworthy completion evidence subsequently appears, record that as a new observation and reassess through the original admission checks; retain this earlier snapshot.

## Conditional future amendment proposal — unexecuted

A transparent operational amendment could support a later scientific comparison while remaining an explicit departure from the original recovery policy. It would be prospective only to the new attempt, not retrospectively compliant with the prior once-only rule. The following conditions are necessary:

1. **Resolve execution status first.** Obtain supported supervisor/backend evidence identifying the old controller and child as completed or terminated, or recover the original handle. If they are alive, preserve and observe the existing attempt. An unknown cause may be reported as unknown; it must not be replaced by an invented exit code. A future environment must provide reliable persistent job identity and completion receipts before launching another long fit.
2. **Declare one bounded decision before additional work.** Hash a new amendment naming the missing endpoint, the observed facts, the exact departure, and one additional allowed attempt with a final stop rule. Existing completed endpoints are retained. No held results, training-loss preferences, or later outcomes may determine whether to retry, change settings, or extend the allowance.
3. **Preserve the scientific configuration.** Retain the exact parent, seed, source/data hashes, architecture, optimizer settings, 1,200 updates, 100-step typed warmup, batch/draw schedule, rehearsal, one-thread setting, and 3,600-second per-attempt CPU cap. Restart from the original parent without partial weights. Any different output path or execution harness must be the declared operational change, with both original and replacement commands retained.
4. **Preserve every attempt and charge its cost.** Use new attempt paths and receipts; overwrite no original or recovery evidence. Report all discarded training-loop CPU and token counts as measured values or explicit lower bounds. Include any unmeasured tail as unknown. Do not count a replacement's whole-process and training-loop CPU twice or present repeated work as the cost of one fit.
5. **Keep qualification unchanged and provenance explicit.** Require all six fixed endpoints, the unchanged data and scoring rules, and separate admission of the amended provenance. Leave original scripts and locks intact. Use a new, audited admission wrapper and explicit amended-study reporting rather than fabricating a successful original `RECOVERY_RESULT.json`. The original campaign remains incomplete under its original policy; any subsequent result is identified as obtained after the operational amendment.

Until the first condition can be met, the scientifically defensible outcome is the incomplete-study package and this unexecuted proposal. No original protocol, training, evaluation, recovery, or summary script was modified by this review.
