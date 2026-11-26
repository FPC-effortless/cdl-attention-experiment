# Durable memory for NanoCOG

This module adds **automatic conversation capture, cross-chat record recall and
restartable decoder sessions** to the existing research runtime.
A successful operation is committed to disk
before it returns. There is no separate `save()` call to forget.

It is an application component and a working local CLI. The host must route its
chat turns through `MemorySession`, or its reference decoder calls through
`PersistentDecoder`. This does not change ChatGPT account settings, install a
chat service, supply a trained language model, or make the experimental writer
and event selector accurate.

The archive and reference decoder remain separate interfaces. `MemorySession`
passes retrieved records to the host's response callback; `PersistentDecoder`
consumes only the tokens the host supplies. In the expanded validation, changing
an accepted ledger value while holding reference-decoder tokens fixed changed no
logits; see [that measured boundary](../../memory-validation-002/decoder/REPORT.md).

`learned_archive.FactualMemory` now adds a separately opted-in experimental
factual interface over the user's frozen COG_CONNECTED_1M model. Its learned
writer and query/answer heads consume native post-board features and connect to
the accepted archive. It supports a declared eight-record categorical schema,
not arbitrary conversation facts or general language generation. All three head
seeds passed 256/256 familiar complete-history counterfactual test pairs and
scored 0/256 on fresh wording, in both matched training recipes. This mode is
**not qualified for automatic factual interpretation of unfamiliar text**.
The native language model weights and outputs remain unchanged. Full scope,
methods and retained failures are in
[MEMORY-INTEGRATION-003](../../memory-integration-003/RESULT.md).

## What is stored

| Memory | Lifetime and identity | Meaning |
| --- | --- | --- |
| Source messages | Stable principal + project namespace, across sessions | What a user, assistant or tool said, with source IDs and recording time |
| Proposed interpretations | Same scope, until explicitly accepted or erased | A candidate claim; never presented as an accepted fact merely because the model was confident |
| Accepted records | Same scope, with revisions and source provenance | An assertion accepted by the host's policy; acceptance is not independent proof of truth |
| Decoder state | One conversation + exact model/tokenizer/schema/runtime | All supported transient state needed to continue the same computation |
| Retry receipts | Request identity retained; decoder outputs have a bounded cache | Evidence that a request already committed, preventing double consumption |

New conversation IDs reset transient decoder state. They still share reusable
memory when the host supplies the same principal and namespace. Changing the
principal or namespace gives a different memory view. The local API does not
authenticate people; a service must obtain identity from its authenticated host,
never from model-generated text or a caller's arbitrary web request field.

## Run it now

Run from `research/nanocog-001`. The store, CLI and temporal reference use Python's
standard library. Python 3.12 and SQLite with FTS5 **3.42 or newer** are required.
The decoder adapter requires PyTorch 2.4 or newer; the checked environment uses
PyTorch 2.10.0 CPU. Tests additionally use pytest and NumPy.

```sh
python -m nanocog.memory --principal alice --namespace orion --session chat-1 \
  record "The spare key is in the blue drawer." --event-id key-observation

# A separate invocation, after closing the first process, with a NEW chat ID:
python -m nanocog.memory --principal alice --namespace orion --session chat-2 \
  recall "spare key"
```

The second command returns the original observation and its provenance. It does
not generate a natural-language AI answer. The default database path is
`$XDG_STATE_HOME/nanocog/memory.sqlite3`, or
`~/.local/state/nanocog/memory.sqlite3` if XDG_STATE_HOME is unset. Set
`COG_MEMORY_PATH` or pass `--db /persistent/volume/memory.sqlite3` to choose a
persistent local volume. An ephemeral container directory is not persistent
infrastructure.

Every command supports `--db`, `--principal` and `--namespace`. Available commands
are `record`, `recall`, `propose`, `accept`, `facts`, `forget`, `forget-session`,
`export`, `backup` and `restore`; use `--help` for their arguments.

## Automatic chat integration

```python
from nanocog.memory.store import MemoryStore
from nanocog.memory.session import MemorySession

# The application supplies its authenticated identity and response function.
with MemoryStore(principal_id=user_id, namespace=project_id) as store:
    chat = MemorySession(store, session_id=conversation_id)
    result = chat.turn(
        user_message,
        respond=lambda message, context: your_model_reply(message, context),
        request_id=stable_request_id,
    )
    send_to_user(result.response)
```

Each turn automatically records the input, retrieves previous memory, calls the
response function, and records the response before returning. The context is
bounded JSON with `accepted_facts`, `unverified_observations` and `truncated`.
It includes provenance and epistemic status. It is **data**, not an instruction
message; the host must preserve that boundary when constructing its model input.
This labelling cannot guarantee a model will resist prompt injection.

The default context budget is 8,000 Unicode characters and eight whole records,
including JSON/provenance overhead. It is not a tokenizer-specific token budget.
Oversized records are omitted whole. Current input is excluded before retrieval
limits, so it cannot hide the only older matching observation. Sources are found
through SQLite full-text search; accepted keys/values are indexed separately.
Direct accepted key/value matches are ranked ahead of facts linked only through
the same source. Ranking uses each record's lexical overlap and length, so adding
another principal's records cannot change this scope's order. Broad queries can
still scan and score many matching records: the output budget is not a bound on
search work. The measured correctness/performance tradeoff is in
[MEMORY-VALIDATION-002](../../memory-validation-002/RESULT.md).
This is lexical retrieval, not a trained relevance model or demonstrated
constant-time access to a lifetime archive.

The exact lexical fast path added in MEMORY-INTEGRATION-003 preserves the same
ranked results while avoiding redundant ASCII normalization. The stronger 50k
single-scope paired workload measured broad recall at about 520 ms versus
1,066 ms before; Unicode-only cost barely changed. See the
[retrieval report](../../memory-integration-003/retrieval/RESULT.md) for content,
scope and filesystem boundaries. Output length still does not bound search work.

The host must reuse the request ID for retries. A committed response is returned
without running the callback again. If computation fails, the saved input
remains available for a retry. A crash between computation and response commit
can cause recomputation. The callback should be side-effect-free: exactly-once
email, payments or other external actions are outside this transaction.

Source dependencies and fact revisions are captured before the callback runs and
rechecked during response commit. A source forgotten in another session, or an
accepted fact corrected while the reply is computing, rejects the stale reply.
The caller may then deliberately retry against current memory.

## Accepted writes and corrections

```python
store.append_message("chat-1", "user", "Orion launches Thursday.", event_id="src-1")
store.propose("orion.launch_day", "Thursday", "src-1", proposal_id="proposal-1")
# A trusted application policy supplies this decision; a confidence score is insufficient.
store.accept("proposal-1", authority="explicit-user-statement-policy-v1")

store.append_message("chat-2", "user", "Orion moved to Friday.", event_id="src-2")
revision = store.fact_revision("orion.launch_day")
store.propose("orion.launch_day", "Friday", "src-2", proposal_id="proposal-2",
              expected_revision=revision)
store.accept("proposal-2", authority="explicit-user-correction-policy-v1")
```

The store retains the prior version for explanation and correction. Two updates
based on the same revision cannot silently overwrite one another: one commits,
the stale one raises `ConflictError`. Acceptance, the current record, and its
history revision are updated in one transaction. A repeated accepted proposal
returns its original receipt and never rolls the current fact back.

No general prose extractor is hidden in this module. Source capture and recall
are automatic; choosing which interpretations deserve authoritative status is a
host policy or separately qualified model capability. Well-formed address/value
updates can still be semantically wrong, as the ANT writer studies demonstrated.

## Resume the existing decoder

```python
from nanocog.memory.decoder import PersistentDecoder

model.eval()
with MemoryStore(principal_id=user_id, namespace=project_id) as store:
    decoder = PersistentDecoder(
        model, store, session_id=conversation_id,
        tokenizer_id="sha256:<digest-of-the-real-tokenizer-assets>",
    )
    result = decoder.run(tokens, request_id=stable_request_id)
    logits = result.logits
```

Reload the same weights and reopen this session in another process to continue.
The wrapper saves the full `DecodeState`: previous embedding and every layer's
local, packet and typed state. It supports the reference `LocalOnly`,
`PacketMixer` and `PacketAndTypedProbe`. It does not pretend the older PULSE
whole-history `forward()` already has a streaming state API.

Model tensor contents, configuration, schema buffers, supported implementation
source, execution attributes, tokenizer identity, session and runtime are
fingerprinted. The tokenizer
digest is supplied by the host; this wrapper cannot verify assets it never sees.
Changed weights, vocabulary identity, source implementation or runtime cannot
silently reinterpret old slots. Start a new transient session after an upgrade;
the separate reusable message/fact archive remains available.

The adapter accepts one stream (`B=1`), immutable eval weights, float32/float64,
and no autocast, custom mixer, compiled call replacement, modified execution
methods or local/global inference hooks. Identity is rechecked
each call. Safe JSON/base64 serialization accepts only known state classes and
bounded finite tensors; it never loads Python pickle. Import checks include
shape, dtype, occupancy, typed legality, revisions and chronology. Checksums
detect damaged bytes, not an attacker who can replace both data and checksums.

State, request identity and output commit atomically using an expected revision.
The scope's erasure generation is captured before inference and checked in that
same transaction. This also rejects a first checkpoint if a source was erased
while it was computing. A retry reads an already committed receipt; deletion
cannot revoke a result that an overlapping retry has already read.
A duplicate request never consumes tokens twice. A stale competing continuation
fails explicitly. Decode output replay is retained for at most 128 recent
requests and a 32 MiB cache budget per session, with the newest receipt retained
even if larger. Older request IDs remain recorded; retrying an expired output
raises `ReceiptExpiredError` instead of executing it again against newer state.
Default raw tensor codec capacity is 16 MiB per state/output. The store caps each
JSON payload at 64 MiB and a message/proposed value at 1 MiB.

## Forget, export and recover

`forget_event(event_id)` redacts the source, recursively redacts dependent saved
assistant replies, removes accepted values sourced from those records, redacts
their value history and proposals, and removes search postings. It keeps minimal
identifier/revision erasure markers so stale request IDs cannot resurrect data.
`forget_session(session_id)` does this for a conversation and retires its ID.
Raw decoder states may encode sources implicitly, so erasure conservatively
invalidates continuation caches throughout that principal/project scope. A new
conversation ID is then required. Explicitly recorded source dependencies can be
followed; untracked copies in arbitrary documents are outside this API.

SQLite's normal secure-delete and FTS5's separate secure-delete are both enabled.
The regression suite checks that a unique erased term is absent from the current
database bytes. This is not a promise of forensic erasure from SSDs, snapshots,
old exports or old backups. Older copies still require separate deletion.

```sh
python -m nanocog.memory --principal alice --namespace orion export memory-export.json
python -m nanocog.memory --principal alice backup memory-backup.sqlite3
python -m nanocog.memory --principal alice restore memory-backup.sqlite3 restored.sqlite3
```

Export streams this scope's readable records under a consistent database snapshot.
Backup uses SQLite's online backup API and includes **all principals/namespaces**:
it is an administrative operation for the local database owner. A checksum
manifest accompanies the backup. Keep both files on independently durable
storage. Restore validates exact bytes and SQLite integrity, writes a new path,
and refuses existing files or leftover `-wal`, `-shm`, or `-journal` sidecars.
Do not restore into a path that another process is using or about to open.

Restore a post-erasure backup, or reapply later erasures before exposing an older
copy. Backups do not autonomously synchronize or rotate. Database/backup files
are created with private OS permissions, not application-level encryption.
Use a persistent encrypted disk and appropriate host access control for deployment.

Transactions use synchronous FULL. Fixed SQLite versions use WAL; older supported
versions without the 2026 WAL-reset fix use rollback journaling. WAL requires a
local filesystem, not a shared network filesystem. Unknown application/schema
versions fail closed. Recognized storage schema v1 is atomically migrated to v2,
which adds erasure generations and checkpoint identity checksums. Legacy payload
checksums are checked before migration, but v1 metadata integrity cannot be
established retrospectively. Legacy v1 backups can be restored and migrated when
opened. Readers never silently start an
empty replacement when a previous state exists but is incompatible.

The stronger decoder execution fingerprint intentionally requires a new native
decoder session for checkpoints written by the original PR 94 implementation.
Accepted facts, observations and their revisions migrate; continuing the old
native activation state transparently across this upgrade is not supported.
Unknown future formats need separately implemented migrations. Model/runtime
upgrades and stable factual records therefore have different compatibility rules.
Stop old v1 writer processes before migration. Concurrent new-version migration
is tested; running an already-connected old binary against the upgraded store
cannot enforce the new erasure-generation rule and is unsupported.

## Time cues: represent the relationship that must govern selection

`temporal.py` defines `TemporalClaim(action_id, relation, anchor_action_id, ...)`,
with exact source spans for the action, cue and anchor. A new claim is `proposed`.
Actual event order is separate from when a report was received or corrected.
The supplied-binding `TemporalReference` uses accepted relations, checks cycles,
retains provenance, and returns unknown for an unbound or ambiguous selection.

For example, “Before the alarm sounded, Amara moved the watch” supplies the
relationship `move-watch BEFORE sound-alarm`; moving the subordinate clause can
reverse those arguments. A bare label “before” does not identify the two actions.
Span matching establishes faithful quotation; it does not prove correct attachment.

An accepted outer memory record does not automatically change the inner temporal
claim's status. Host code must validate its live source and explicitly construct
an accepted `TemporalClaim` from the accepted record before using the reference.
This prevents a pending model proposal from acquiring authority just by loading.

The timing research still has an unresolved learned mechanism:

- [PHASE-003](../../cog-bind-time-phase-003/RESULT.md) improved labels without
  improving strict choices. Its comparator already used phase probabilities;
  it is inaccurate to say a correct complete phase interpretation was ignored.
- [SCOPE-004](../../cog-bind-time-scope-004/RESULT.md) changed scores but its scope
  and pooling arms made the same strict selections on every expanded case.
- [PROC-005](../../cog-bind-time-proc-005/RESULT.md) improved some locator labels,
  but swapping action/trigger roles did not change hard choices in its intervention.

A subsequent learned design must make the final route consume the bound action,
anchor and cue through its actual composition path, then pass native role-swap
and fresh-family tests against frozen W at matched competence, with Y/Z fixed.
This module supplies an inspectable interface and finite reference, **not that
learned repair**. No fit, holdout tuning or model-performance claim accompanies it.

## Verification and operational limits

```sh
OMP_NUM_THREADS=1 python -m pytest -q nanocog
```

See [MEMORY-VALIDATION-002](../../memory-validation-002/RESULT.md) for the expanded
adversarial tests, fixes, scale measurements and inference costs.
[VERIFICATION.md](VERIFICATION.md) preserves the original, narrower verification
record. The tests use synthetic inert data and temporary databases.
They exercise real fresh processes, interrupted writes, lost acknowledgements,
concurrent corrections, dependency erasure, corrupted backups, incompatible
models and bounded retrieval, as well as the existing model/state regressions.

“Across a lifetime” is an operational requirement, not a finite test result.
The runtime now supports durable continuity; continued availability still needs
persistent storage, independently retained backups, restore drills, capacity
management, and maintained migrations. The archive and small deduplication
metadata grow with use. No total-lifetime capacity, general-language correctness,
multi-machine failover or matched-accuracy speed advantage has been established.

Implementation references: [SQLite WAL](https://sqlite.org/wal.html),
[online backup](https://sqlite.org/backup.html), and
[FTS5 secure deletion](https://sqlite.org/fts5.html#the_secure_delete_configuration_option).
