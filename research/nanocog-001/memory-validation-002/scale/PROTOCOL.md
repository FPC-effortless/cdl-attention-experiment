# MEMORY-VALIDATION-002 — scale, CLI and temporal protocol

Registered before the first workload execution, 2026-11-23 UTC. This is an
infrastructure diagnostic of the durable-memory implementation, not a learned
writer, a language-model benchmark, a claim of lifelong capacity, or a comparison
against attention. No fitting, paid compute, network inference or personal data.

## Fixed workload and acceptance

Seed `271828`; sequential stages of 1,000, 10,000 and 50,000 synthetic source
messages in one SQLite database. Eight principal/project scopes share the
database; each scope rotates through 200 conversation IDs. Every 31st source
proposes and explicitly accepts one of 32 per-scope keys, correcting its previous
value with an exact expected revision. Other writes are nuisance observations.
Unique exact-word needles identify each source. A canonical input stream hash
excludes nondeterministic timestamps.

At each stage:

* Probe early, middle and late sources in all eight scopes, plus seeded random
  positions. Every requested unique exact token must return its source. Repeat
  the same probes in a fresh Python process against the same database, then
  against a verified backup restored to a new file.
* Compare all current facts and revisions with the deterministic oracle. Probe
  current keys and values through recall; correction history must remain ordered
  and obsolete values must not reappear as current accepted facts.
* Foreign-scope source probes must return no source. Check actual rendered context
  length against 256-, 1,024- and 8,000-character limits. These are character caps,
  not token caps or archive-cost caps.
* Measure wall time, process CPU, maximum resident set, ingestion operations,
  per-call recall latency, full-database backup, scope export, restore, and sizes
  of the main file and journal sidecars. Report cumulative as well as incremental
  ingestion cost. No OS cache flush is performed; a process restart is not a cold
  storage-cache test. Backups use the same temporary local filesystem and do not
  simulate disk loss.

A literal `car` record queried only with `automobile` is a predeclared lexical
limitation control. Failure to retrieve it is recorded separately, not a
violation of the advertised lexical contract. Exact-token success does not imply
successful natural-language recollection or useful answer influence.

## Integration diagnostics

Run actual CLI subprocesses for all commands, explicit database paths, configured
`COG_MEMORY_PATH`, and absolute `XDG_STATE_HOME`. Do not change `HOME`. Validate
cross-invocation recall, proposals and corrections, authority provenance,
scope filtering, export, backup, restore, erasure, error exit codes, and generated
session/event IDs. Invalid command-line syntax is argparse behavior, separate
from structured runtime errors. Existing-target backup/restore must fail without
changing existing contents.

Construct a 10,000-edge temporal chain using explicit accepted action/cue/source
spans. Shuffle source/claim arrival order; test both directions, a unique
selection, ambiguity, unbound alternatives, a cycle, and invalid source spans.
Persist and reload a small subset through ordinary accepted JSON facts across a
process restart, while preserving the distinction between host fact acceptance
and the temporal claim's own accepted status. This diagnoses an explicit
reference interpreter, not learned temporal binding or PHASE-003 qualification.

## Execution limits and reporting

Stop before the next stage if cumulative CPU reaches 300 seconds or wall time
reaches 540 seconds; report a skipped/censored stage explicitly. A subprocess has
a bounded timeout. Retain compact JSON results, the runnable harness, source
hashes, environment and operation-stream hashes. Temporary databases are removed
after verification. Any production source change during a run is a protocol
deviation and must be identified; rerun affected checks against a pinned version.
Report failures with reproduction details. Do not silently discard a workload,
average a failed test into a pass, or generalize these finite synthetic timings
to asymptotic scaling, a production deployment, or model inference efficiency.

### Amendment before final paired runs

The first baseline execution completed all three stages without failure in
19.44 seconds. Before running any repaired implementation, extend the same
workload to check every accepted historical value and source against the
deterministic generator, to time common-word bounded context separately, and to
exercise CLI fact correction and cross-project filtering explicitly. Rerun the
unchanged baseline snapshot as well as the repaired package with this identical
expanded harness. The original baseline result is retained as
`baseline-initial-results.json`. Runtime packages are copied to isolated temporary
directories before execution so concurrent implementation work cannot mix source
versions within a run. Source hashes, rather than the mutable working tree name,
identify the tested implementation.

### Filesystem qualification before overlay runs

`df -T` identified the default `/tmp` workload directory as `tmpfs`, and the
workspace as `overlay`. Retain the complete paired tmpfs runs with that label.
Run the same registered stages again with `TMPDIR` on workspace overlay, comparing
baseline and repaired implementations only within a matching filesystem. The
overlay results are the primary available filesystem timing evidence. Neither
filesystem test establishes survival of host/disk loss, power failure, or a
real-world service lifetime. This amendment changes the storage location, not
the generated workload, acceptance criteria, or implementation.
