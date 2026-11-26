# Independent storage and session adversarial audit

## Result

The published baseline had **three reproduced failures** in nine bounded system
experiments. Six experiments passed. The repaired working tree now passes all
nine experiments, including an amplified **20 rounds × six simultaneous OS
initializers**. That amplified check uncovered a second startup race after the
first repair; its failed run is preserved separately and the final run passes.
These results apply to `MemoryStore` and `MemorySession`, not to learned
language-model competence or a production deployment.

The baseline commit is `56cb4eeb04d68f5f6c210c756f0790f34cc2fd5f`. The final
`store.py` SHA-256 is
`3bf7561d194e19cc32a6ab4377ca4656aea86800ce6d026fd0a291ab18ec47c1`.
The exact Python source hashes and environment are recorded in
`results_baseline.json` and `results_after_fix.json`. The final run checks source
hashes before and after execution and confirms they did not change. All inputs
were synthetic, all databases lived in temporary directories, and every process
ran locally. No model training, paid compute, remote service or real user memory
was involved.

Run the complete experiment with:

```sh
python research/nanocog-001/memory-validation-002/storage/run_adversarial.py
```

The default writes `results_latest.json`, preserving the evidence snapshots.
Pass `--output PATH` to select another output location. Use
`--case CASE_NAME` to select one or more cases. A failed requirement is recorded
as `FAIL`; an unhandled experiment exception is `ERROR`. A failed requirement,
experiment error or source change makes the runner exit nonzero.

Twenty new collected storage regressions cover the failures, v1 migration,
erasure generation and checkpoint identity integrity. Together with the original
store/session tests and the parallel capacity audit's two tests, the targeted run
passed **51 tests**:

```sh
cd research/nanocog-001
OMP_NUM_THREADS=1 python -m pytest -q \
  nanocog/memory/test_adversarial_store.py \
  nanocog/memory/test_store.py nanocog/memory/test_session.py \
  nanocog/memory/test_adversarial_capacity.py
```

## Reproduced failures

### S1: Unrelated accepted facts displace the answer from memory context

**Observed:** A single source says that Orion launches Thursday and records ten
independent stock counts. Accepting `orion.launch = Thursday` initially makes
that fact appear in `MemorySession.context("When does Orion launch?")`. Accepting
the ten unrelated stock facts then makes the same query return only the eight
most recent stock facts. The target launch fact remains correctly stored at
revision 1, but disappears from the context supplied to the response callback.

**Mechanism:** Recall selects facts whose keys/values match the query **or** whose
source was retrieved. It sorts this union by modification time, then applies the
eight-record default limit. Recent facts with no direct query match can therefore
crowd out the directly matching fact merely because they share its source.

**Impact:** This is a retrieval correctness problem, not evidence of ledger
corruption. A correct durable fact can become unavailable to a fresh answer after
unrelated accepted writes. It is directly relevant to the requirement that an
irrelevant change must not alter a relevant answer's available evidence.

**Repair target:** Rank directly matching facts before broad source-related
candidates, preserving bounded context and source provenance. Add a regression
that checks the callback context, not just `facts(key)` or index membership.

Machine-readable case: `unrelated_facts` in `results_baseline.json`.

### S2: Other principals and namespaces change local retrieval ranking

**Observed:** Alice has two one-word records, `compass` followed by `beacon`. Her
query `compass beacon` initially orders them `[b, a]`. Adding fifty `beacon`
records in Bob's scope reverses Alice's order to `[a, b]`, without changing any
Alice record. The same reversal occurs when the additions belong to a different
namespace of Alice's account. A query for a foreign-only canary returns no
foreign content in both fixtures.

**Mechanism:** The shared FTS table's BM25 score uses global corpus statistics,
although rows are filtered to the current principal and namespace. Scope filters
protect returned content in these cases, but do not isolate the ranking function.

**Impact:** Another scope can change which local memory wins a bounded retrieval.
The experiment demonstrates cross-scope influence, not direct disclosure of
foreign message content. It does not establish a practical data-exfiltration
attack or measure one.

**Repair target:** Make ranking depend only on the requesting scope, or explicitly
choose a scope-independent deterministic score and verify its relevance tradeoff.
Do not label removal of content leakage alone as a ranking-isolation fix.

Machine-readable case: `ranking_interference` in `results_baseline.json`.

### S3: Simultaneous first startup can reject a legitimate initializer

**Observed:** Eight trials launched six spawned OS processes simultaneously
against a new path. Seven trials admitted all six writers. One trial admitted
five; the sixth raised `FileExistsError` in `_prepare_file` through
`_unused_database_path`. Its write never occurred. No acknowledged write was lost.

**Mechanism:** One process observes that the path does not exist; another creates
it before the first checks that the destination is unused. The first process
treats the peer's legitimate creation as a restore/path collision and fails.

**Impact:** Concurrent service startup can fail nondeterministically on a fresh
database. This is an availability failure, not corruption. The observed 1/8
trial occurrence is a reproduction, not an estimate of deployment failure rate.

**Repair target:** Coordinate initial creation without removing the guard against
stale SQLite sidecars at a genuinely new destination. A race test should use
multiple OS processes and retain both the successful and failed paths.

Machine-readable case: `concurrent_initialization` in `results_baseline.json`.

### S4: Journal initialization also raced during simultaneous startup

The first repair admitted a peer-created database correctly. Expanding the
startup experiment to 20 rounds exposed another boundary: one of six processes
in round 15 failed with `sqlite3.OperationalError: database is locked` while
changing the journal mode to WAL after its schema transaction. This is preserved
in `results_startup_lock_reproduction.json`; source hashes remained unchanged.

The final implementation checks the existing journal mode and retries only
SQLite BUSY/LOCKED conditions outside a transaction. It uses the caller's finite
wait budget and restores the original SQLite busy timeout afterward. I/O errors
and zero-budget waits fail immediately. The final **120 simultaneous startup
attempts all succeeded**, and focused fault-injection regressions cover transient
locks, persistent locks, zero timeout, timeout restoration and non-lock errors.

## Repairs and migration

Message ranking now depends on each candidate's distinct query-term overlap and
term density, then recency; foreign corpus statistics cannot alter the score.
Direct key/value fact matches are considered before source-only fallback facts.
The separate scale audit measures the cost: this local scoring is more expensive
for broad queries. Passing correctness checks is not a speedup claim.

Schema v2 adds a monotonic per-scope erasure generation. A continuation snapshots
it before inference, then `commit_checkpoint` checks it atomically before any
receipt replay or state commit. A mismatched generation rejects the work even
when the decoder has never saved a checkpoint. Other scopes do not change this
counter, and repeated erasure of an already-erased event does not increment it.
The store argument is optional for compatible direct callers; the persistent
decoder supplies it.

Current checkpoint and request-receipt identity seals bind principal, namespace,
session, request, input hash, revision and manifest to their result checksum and,
for a current checkpoint, its payload checksum. A valid payload/result plus its
checksum cannot be swapped into another request without detection. Expiry and
erasure clear old result digests and re-seal the retained receipt identity.
These are accidental corruption checks, not authentication against a database
owner who can recompute checksums.

Recognized v1 storage migrates to v2 within one transaction. Tests preserve both
principals, accepted corrections and history, message content, current decoder
state and old receipts; six simultaneous new-version initializers migrate once.
Corrupt old content aborts and rolls back the entire migration. A v1 backup can
be restored and migrates when opened by the new store. Migration verifies legacy
content hashes but cannot retrospectively establish the integrity of metadata
that v1 never sealed. Stop old v1 writer processes before upgrade: concurrent
new-version openers are supported, mixed old/new live writers are not qualified.

## Checks that passed

| Experiment | Evidence | Observed outcome |
| --- | --- | --- |
| Concurrent established database use | Four spawned writers each captured and accepted 100 records, with a fifth process running 120 read rounds | All 400 messages and accepted facts present; exact expected values; no malformed/sourceless fact observed; all five processes exited normally; SQLite integrity `ok` |
| Transitive erasure and stale response retry | Thirty cross-chat dependency links, five branches, one saved reply and a fact sourced from the last link | Root erasure removed all 36 dependent messages; current fact removed and history redacted; exact old response retry raised `DeletedError`; another namespace survived |
| Mutation during response calculation | A callback in one OS process paused after reading context while a second process corrected or erased its source | Correction raised `ConflictError`; erasure raised `DeletedError`; neither stale assistant response committed |
| Crash before and after acknowledgment | A separate writer was killed with SIGKILL immediately before COMMIT or after `append_message` returned | Before COMMIT: no message, FTS posting or dependency edge; after acknowledgment: all three survived; integrity `ok` in both cases |
| Checkpoint erasure and restore | Save checkpoint, back up, erase source, retry request, back up again and restore both snapshots | Existing checkpoint invalidated; erased receipt output stayed null; post-erasure restore retained invalidation; old backup still retained old source and state |
| Correction versus exact request replay | Answer Thursday, accept correction to Friday, repeat old request ID, then issue a new request ID | Exact old retry returned Thursday without callback; new request answered Friday |

## Limits that remain explicit

An exact committed retry is a receipt for its original answer. It does not
re-evaluate that answer after facts change; clients need a new request ID for a
fresh answer. In the fixture, the old request correctly replayed Thursday and a
new request correctly returned Friday. Whether a product should display a stale
receipt warning is a separate interface decision.

A pre-erasure backup restored the erased source and decoder checkpoint. This is
the documented retained-copy boundary: the implementation does not automatically
propagate subsequent erasures to old backups. A post-erasure backup restored the
erasure correctly. The experiment does not qualify backup rotation, remote
retention, encrypted disks or operational deletion procedures.

The checkpoint erasure check starts with an already committed checkpoint. It
does **not** cover an in-flight first decoder write without prior receipts; that
boundary is owned by the parallel decoder audit.

The temporary databases in this audit are on **tmpfs**, as recorded in the final
JSON. SIGKILL tests establish process-crash behavior at the exercised boundaries. They
do not simulate physical power loss, storage-controller failure, disk corruption,
host loss or a lifetime of operation. Reported elapsed times describe these small
fixtures only and are not a latency benchmark.

No learned response generator ran in this audit. Callback responses were
deterministic test functions. Correct persistence does not demonstrate correct
natural-language extraction, learned writer repair, semantic recall, event-time
binding or accurate answers from a trained model.
