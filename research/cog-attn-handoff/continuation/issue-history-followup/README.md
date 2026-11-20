# Follow-up on real GitHub issue histories

The first issue-history study found six cases where the event stream said open and the direct issue snapshot said closed. The snapshot supplied a close time for every case. I revisited each issue through the GitHub connector. All six direct responses matched the archived state, close time and updated time. The cases cover five completed issues and one duplicate. [Prior snapshot cross-check](../issue-history-study/snapshot-crosscheck.json)

## What ran

The [reconciliation policy](anchor.mjs) stages a snapshot-derived close only when the close time is later than every observed state event. It keeps two times. The reported close determines when the anchor can apply, and the time the evidence was available determines when a replay can use it. A changed issue version invalidates the confirmation. A snapshot saying open provides no timestamp for a missing reopen, so the policy defers.

| Check | Result |
| --- | ---: |
| Correct current states from timeline-only replay on the six deliberately selected discrepancies | 0/6 |
| Staged snapshot anchors matching a fresh GitHub read | 6/6 |
| Actual close-time and knowledge-time boundary queries | 24/24 |
| Additional policy invariant checks | 8/8 |

The selected discrepancies provide a regression set. Their result cannot estimate the frequency of omissions or the false repair rate. Both snapshots came from GitHub. Agreement confirms the provider's recorded state and timestamp rather than establishing what happened in any unrecorded history. The anchor does not carry an event ID or actor. These checks ran in a JavaScript isolate, and SQLite and the integrated Veritas runtime still need testing in the fresh cohort. [Cases](cases.json) · [Results](results.json)

Run `node verify.mjs` to recompute the report from the archived, non-text inputs.

## Fresh holdout

The follow-up collector logged at least 300 issues before the execution environment disconnected. Final acquisition and the Research Lab OS and TrainLab runs are unverified. The [frozen design](PROTOCOL.md) records the source selection, the comparison of same-second supervision arms, and the independent snapshot checks.

The resumed study acquired all 810 issues and completed the TrainLab comparisons and replays. [Fresh-cohort results and qualification](../cog-051-060/README.md)
