# GEN-005: Keep a generated answer current until it commits

**All four frozen checkpoints passed the preregistered test.** The system drafted an answer privately, then a separate SQLite connection committed an intervening change. Before saving the reply, the scoped check revalidated the exact requested name, alias binding, fact revision and live permission in the same transaction that saved the receipt. It retried once if the relevant evidence changed, and returned no answer if it changed again.

The figures below are **per checkpoint** on the same 288 scenarios, drawn from the same 16 independent source families reused by all four checkpoints. They are engineering outcomes, not a pooled model accuracy estimate.

| Reply policy | Stale primary receipts | Unnecessary retries | Native private drafts, including follow-ups |
|---|---:|---:|---:|
| No evidence check before commit | 144/288 | 0 | 272 |
| Check the selected fact row only | 80/288 | 0 | 320 |
| Check the entire board version | 0/288 | 16 | 416 |
| Check only the request’s dependencies | 0/288 | 0 | 400 |

Per checkpoint, the unguarded policy committed 80 stale answer receipts, including 16 same-value revision changes, and 64 stale negative receipts. Checking the selected row left 16 stale alias-answer receipts and the same 64 stale negative receipts. Neither a correct-looking word nor an empty stale reply makes the receipt current.

A row-only check missed a remapped alias when both fact rows kept their original revisions; it also missed newly available facts or identities after an initially negative lookup. The scoped check caught both. It also detected all 16 changes that returned the same word to the same record but advanced the revision. Same text did not make a stale receipt current. When the change concerned another owner or the other scope, the scoped policy kept one decode and no retry. When a second relevant write followed a retry, all 16 scoped cases stopped without emitting an answer. The second write was scheduled only in retrying cases, so controls that did not retry did not receive it.

The unchanged native 65-character head produced exact current words in **240/240, 232/240, 239/240 and 237/240** actual final fresh primary answer cases (GEN-001 seeds 1/2, then GEN-002 seeds 1/2). It generated both exact old and new words in **32/32, 31/32, 32/32 and 31/32** correction or remap pairs. Wrong native words with current provenance count as answer errors, not stale infrastructure.

After SQLite connections closed and reopened, each checkpoint allowed 16 explicitly historical replays after an ordinary correction while fresh requests reread the current value. It denied 16 revoked historical replays, redacted 32 after erase or alias remap, and rejected 16 attempts to reuse a request ID with a different question. Historical receipt blobs remained in the private test database; redaction is not deletion of stored data.

This verifies a scoped commit check under a controlled interleaving of real SQLite writes. The benchmark retained private drafts for audit but showed zero such exposures in the modeled public reply or receipt fields. It did not test an external serializer or streaming API, simultaneous duplicate requests, authenticated callers, network delivery, process restart, or a serving-speed advantage.
