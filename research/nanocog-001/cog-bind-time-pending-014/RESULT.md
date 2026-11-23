# COG-BIND-TIME-PENDING-014: withhold stale answers after an unresolved edit

**The pending-edit guard passed every preregistered gate in all three runs.** On the fresh 24-session controlled outage fixture, the unchanged control emitted the exact older answer on all 72 targeted asks per seed. The guard instead withheld all 72, without exposing an event, item or source-position answer. It retained each unresolved intent across a cold process restart and service restoration, and resumed with the exact corrected answer in all **24/24 sessions per seed** only after a successful retry.

**The guard did not change any unrelated output or any committed evidence.** Other-anchor exact groups were 24/24 across all seeds. Other-board exact groups were 23/24, 23/24 and 24/24 because the first two frozen model runs already abstained on one unrelated board. Overall exact nonpending query counts were 260/264, 260/264 and 264/264. No unrelated answer was newly withheld, no ambiguous request guessed a scope, and no stale repeat caused a second toggle. Control and guard intentionally kept identical pending state and changed response policy only.

Both compact modes re-encoded **zero old prompt tokens**; this guard adds no new prompt-replay savings relative to its control. Its role is to keep an unconfirmed edit from silently becoming a stale committed answer. All registered failures were **injected** on grounded version-current commands, so this does not estimate natural grounding failure rates or establish recovery from those failures. If a retry remains ungrounded the code keeps waiting; the live held panel had no such retry. No statements here qualify free-form language, multi-agent concurrency, production reliability or KV-cache performance.

## Exact per-seed results

| Seed | Injected failures | Eligible stale-control sessions | Guard blocks at targeted asks | Unchanged anchor | Other board | Exact scheduled nonpending asks | Exact recovery | Exact cold pending load | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 20371 | 24/24 (100.00%) | 24/24 (100.00%) | 72/72 (100.00%) | 24/24 (100.00%) | 23/24 (95.83%) | 260/264 (98.48%) | 24/24 (100.00%) | 24/24 (100.00%) | PASS |
| 20471 | 24/24 (100.00%) | 24/24 (100.00%) | 72/72 (100.00%) | 24/24 (100.00%) | 23/24 (95.83%) | 260/264 (98.48%) | 24/24 (100.00%) | 24/24 (100.00%) | PASS |
| 20571 | 24/24 (100.00%) | 24/24 (100.00%) | 72/72 (100.00%) | 24/24 (100.00%) | 24/24 (100.00%) | 264/264 (100.00%) | 24/24 (100.00%) | 24/24 (100.00%) | PASS |

## Version, parity, and input custody

| Seed | Paired matches outside allowed answer policy | Continued post-retry withholds | Ambiguous and duplicate state nonmutation | Cold state bytes read, guard | Final compact snapshot bytes per session | Old model prompt tokens, both modes |
|---|---:|---:|---:|---:|---:|---:|
| 20371 | 552/552 (100.00%) | 0 | 24/24 and 24/24 | 29,663 | 1077–1221 | 0 / 0 |
| 20471 | 552/552 (100.00%) | 0 | 24/24 and 24/24 | 29,661 | 1081–1221 | 0 / 0 |
| 20571 | 552/552 (100.00%) | 0 | 24/24 and 24/24 | 29,797 | 1209–1221 | 0 / 0 |

The same 24 fresh held sessions were used at each frozen seed; the seeds do not multiply the number of independent session roots. All targets use raw words/coordinates/version. Control and guard have the same model and same pending metadata; only response policy differs. This is a test of whether to withhold an answer, not a test of learning new information.

Use the same three frozen PACKET checkpoints for both modes, with the
established learned base/query paths and engineered role selection,
chronology, named focus and patch grammar. No new fit. Both modes save
only compact typed evidence, version/overlay and focus/pending metadata.
Neither may reopen the historical prompt or older patch text.

A separately typed test control simulates a temporary service outage on a
version-current, grounded partial patch. It does not pass an expected answer
or semantic target. Both modes accept the same new raw command and track
the same resolved board, anchor, expected version, original request digest
and turn if an edit is unavailable. The control still returns the previously
committed answer. Guard instead withholds it for the matching saved scope,
based on the unchanged learned ASK anchor. The guard alone cannot make an
edit succeed, correct learned evidence or resolve an ambiguous command.
Injected service failures and natural unavailable evidence are counted
separately. This is not a measure of operational outage frequency.

An unchanged board or different anchor is unaffected. An ambiguous command
cannot name a pending scope; a stale duplicate cannot advance the version.
The registered identical successful retry clears the pending record after
both service recovery and cold restore. Service recovery alone does not
clear it. In the implemented rule a successful same-scope edit or accepted
full-board replacement clears the relevant record; the experiment does not
qualify arbitrary concurrent or superseding intent. Before emitting a
FRESHNESS_BLOCK the query head and shared composer can compute a tentative
result internally, but its stale evidence is not exposed as the answer.

The fresh nested subprocess handles exactly one new streamed turn at a time
and its own checksum-validated typed snapshot, with no session tape or labels
mounted in the child. Engine state and counters are recorded identically
except for targeted output policy. Phase timing is exploratory on frozen CPU
and includes starts; no production latency, KV-cache, free-form semantics,
durable security, sudden power loss, or complex multi-pending lifecycle
claim follows. Detached labels and independent exact replay separately
score eligibility, safe scope and successful recovery.
