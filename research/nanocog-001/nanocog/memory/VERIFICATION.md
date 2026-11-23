# DURABLE-MEMORY-001: engineering verification

**Historical initial verification, superseded by the broader
[MEMORY-VALIDATION-002 audit](../../memory-validation-002/RESULT.md).** The tests
below passed at the original commit, but the later audit found additional recall,
startup, execution-identity, receipt-integrity and erasure-race defects. Preserve
this record as evidence of the original testing boundary; it is not the complete
current verification or a claim that those defects were absent.

## Result

The complete local NanoCOG reference suite passed on 2026-11-23:

```text
python3 -m pytest -q nanocog
88 passed, 1 skipped in 17.11s
```

The new memory tests account for 65 passing cases: store 18, chat session 11,
decoder/codec 16, and temporal interface 20. The remaining 23 passing cases are
existing reference regressions. The skipped test is
`test_actual_archived_pulse_source_integration`, which requires the optional
`PULSE006_SOURCE` archive; it is not a CUDA or new persistence test.

The executed environment was Python 3.12.14, SQLite 3.53.1, PyTorch 2.10.0+cpu,
NumPy 2.3.5 and pytest 8.4.2. A scoped CPU CI workflow is included. This local
record does not assert a remote CI outcome; inspect the actual pull request.

The implementation is based on the NanoCOG research branch at
`b357305a8e01ee1afec96d6174aa5066711e501c`. Historical model, experiment, checkpoint,
result and corpus files are preserved. The dependency lower bound is corrected
to PyTorch 2.4 because the decoder uses RMS normalization and device-specific
autocast APIs; execution was checked on 2.10, not every permitted version.

## What was exercised

| Failure or requirement | Executed evidence |
| --- | --- |
| A new chat cannot see an earlier chat's memory | CLI records in one process; another process with a new chat ID retrieves the source under the same principal/project |
| Memory leaks across identities | Different principals and namespaces retrieve no records from the original scope |
| Model state is lost on restart | Fresh Python subprocesses resume LocalOnly, PacketMixer and PacketAndTypedProbe; logits match uninterrupted reference continuation within rtol 1e-5 / atol 1e-6 |
| A partial write corrupts the ledger | A child exits after the current-fact statement and before the history statement; reopening reveals no partial accepted fact |
| The commit succeeds but its acknowledgement is lost | A child commits and exits immediately; replaying the same request does not advance the revision twice |
| Two chats overwrite the same fact | Two connections competing from revision zero produce one accepted revision and one conflict |
| Another chat changes evidence while a response is computing | Source erasure or accepted-fact revision change prevents the stale reply from committing; retry can use current memory |
| An exact retry consumes tokens twice | Decoder receipts return saved logits without advancing the current state, including retries of older retained requests |
| Output-cache growth is unbounded | Count and byte-budget tests prune outputs while preserving applied request identity; expired outputs fail explicitly |
| Old model slots are reused with a new model | Changed weights, tokenizer identity or runtime manifest are refused; live model mutation is also refused |
| Imported state is structurally plausible but impossible | Corrupt/nonfinite tensors, malformed shapes, invalid packet occupancy and contradictory typed chronology are rejected |
| A source is forgotten but survives through its answer | Two-hop derived message chains, accepted values, historical values and search postings are erased; stale IDs cannot resurrect them |
| A restore is silently overridden by old journal data | Restore and new database creation reject paths with leftover WAL, SHM or rollback-journal sidecars |
| A damaged backup is trusted | A changed byte fails checksum validation before the restore destination is created |
| Unknown future schema is treated as empty memory | An unsupported schema version fails closed |
| The current query crowds out the only prior memory | Exclusion happens before retrieval limits; a one-record context still retrieves the prior source |
| Context becomes unbounded | The full rendered JSON budget includes metadata and escaping; large records are omitted whole |
| Labels imply timing competence | Supplied temporal bindings are checked for direction, scope identity, source spans, transitivity, ambiguity and cycles; no learned performance is claimed |

Process-exit tests exercise actual SQLite recovery, not a simulated dictionary.
They are not power-cut, disk-controller or lifetime hardware endurance tests.
All data is synthetic and inert; no user's live memory was created or erased.

## Independent review and fixes

A separate reviewer inspected the state boundary before implementation and the
store/codec/decoder/session code afterward. It reproduced two concrete defects:

1. A clean backup restored over a missing main file could be overridden by a
   leftover WAL. The fix rejects main paths and all SQLite sidecars. The reviewer
   reran the original reproduction and confirmed rejection for both restoration
   and new database creation.
2. SQLite `secure_delete` alone left a unique erased word in FTS5 postings. The
   fix enables FTS5's separate secure-delete option, requires SQLite >=3.42, and
   tests absence of the synthetic erased token in current database bytes.

Review also caught the old PyTorch lower bound. Subsequent inspection found no
additional blocking issue in the allowlisted codec, decoder manifest/CAS path,
or callback's source/revision checks. This is bounded engineering review, not a
security certification. Old backups, external copies, SSD remanence, encryption,
and host authentication remain outside the erasure/API-isolation guarantees.

## Research evidence boundaries

The user-supplied PHASE-003 private archive was kept outside the public checkout.
Its SHA-256 is
`77420248b14c6caf6230b8fe43a34464d79000e957406ee565fb1fb6c3227c23`.
The archive's internal manifests verified: 607 file hashes in SHA256SUMS,
317 source-lock entries, 18 fit entries, and 133 wire entries, with no mismatches.
These are overlapping manifest counts, not independent experimental observations.
No model training, new held-out scoring, or independent model-performance
reproduction was performed for this engineering change.

The pinned PHASE-003 report and the later SCOPE-004 and PROC-005 reports were
read. Their negative learned-selection results remain negative. The new temporal
reference consumes supplied accepted bindings and is explicitly diagnostic. Its
passing unit tests must not be reported as learned timing or writer repair.

## Readiness

The runtime and CLI are usable from a persistent local volume, and the host can
integrate the automatic chat/decoder wrappers now. This does not constitute a
deployed chat product or automatic persistence inside ChatGPT itself.

Remaining operational work belongs to the chosen host: authenticate identities,
attach persistent encrypted storage, retain backups on independent infrastructure,
exercise restore, manage growth, and maintain future schema migrations. General
writer correctness, learned retrieval quality, event binding, noun copying,
multi-machine availability, and matched-accuracy serving speed remain separate
unqualified claims. A finite test suite cannot establish literal lifetime reliability.
