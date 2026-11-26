# MEMORY-INTEGRATION-003: exact archive retrieval cost

## Result

The exact lexical fast path reduced the measured broad-query cost materially.
On the reused 50,000-source workload (eight scopes, 6,250 sources per scope), the
same-process median broad recall at an 8,000-character budget fell from
**215.406 ms to 88.003 ms**, a **2.45×** speedup. Both builds passed all
**14,630 checks**, including fresh-process and backup-restored reads. Database
size was identical at **44,023,808 bytes**.

A stronger interleaved comparison used the **same database containing 50,000
sources in one scope**. The broad ASCII query fell from **1,066.248 ms to
519.776 ms**. All **54 paired comparisons** across nine query forms returned
exactly equal complete recall objects. This reduces a measured serving cost; it
does not qualify language understanding or establish constant-time retrieval.

## Mechanism and preserved behavior

The previous ranker normalized and scanned every candidate character in Python,
even when the case-folded text was entirely ASCII. Unicode NFD is the identity on
ASCII, and ASCII combining classes are zero. The new `_word_tokens` therefore
applies the unchanged token regex directly to ASCII text and retains the original
normalization algorithm for non-ASCII text. The regex is compiled once.

FTS still supplies every matching candidate. The score, tie order, scope filter,
fact priority, context construction, provenance, erasure and persistence paths
are unchanged by this optimization. No approximate top-k, new index, result
cache, schema migration or extra write is introduced. Schema remains version2.
The unchanged lexical negative control still finds `car` and misses `automobile`.

This small change was sufficient to improve the measured bottleneck. A persistent
token index would add write, migration, corruption and erasure responsibilities;
it was unnecessary for this result and was not implemented or benchmarked.

## Exactness and regression evidence

- **71 storage/regression tests passed**, comprising the existing 51 tests and
  20 new cases. The new retrieval cases compare the literal old tokenization and
  scoring algorithm on accents, combining marks, case-fold-to-ASCII characters,
  scripts, punctuation, 1,000 generated mixed strings and 1,000 arbitrary Unicode
  strings. Actual database comparisons cover limits, session and exclusion
  filters, small context budgets, foreign writes, corrections, erasure and reopen.
- Both independently frozen packages passed **14,630 scale assertions each**:
  294, 1,543 and 3,022 per process mode at 1k, 10k and 50k sources, each checked in
  the current process, a fresh process and a backup-restored process; plus 43 CLI
  and 10 temporal checks. There were no skipped stages, failures or source drift.
- The paired single-scope run checks the entire returned object, including
  ranked records, accepted facts, context and provenance, on every warmup and
  measured pair. The database integrity check returned `ok`.
- `verify_results.py` independently rejects missing stages, changed source
  hashes, incorrect check counts, failed probes, differing input manifests,
  budget violations and mismatched schema/backup receipts. An old runner's zero
  exit status alone is not treated as evidence of success.

## Cost measurements

All database files were on the workspace **overlay filesystem**, not `/tmp`
tmpfs. The sequential scale runs reused the unchanged prior runner and the same
deterministic operation stream. These are shared-host measurements; the paired
interleaved run provides the stronger timing comparison.

### Prior workload, eight scopes

Broad times are medians over all eight scopes at an 8,000-character context
budget, in the same process. Raw files also retain fresh-process and restored
measurements. The 50k broad medians in those two modes were respectively
213.800→87.519 ms and 198.803→134.177 ms.

| Total sources | Sources per scope | Broad baseline (ms) | Broad candidate (ms) | Checkpointed database bytes, both builds |
|---:|---:|---:|---:|---:|
| 1,000 | 125 | 9.558 | 8.253 | 1,159,168 |
| 10,000 | 1,250 | 49.215 | 26.269 | 9,011,200 |
| 50,000 | 6,250 | 215.406 | 88.003 | 44,023,808 |

At 50k, exact-source median recall was 0.247→0.236 ms and accepted-fact median
recall was 0.204→0.194 ms. These small differences are not a separate speed claim.

| Other measured cost at 50k | Baseline | Candidate |
|---|---:|---:|
| Cumulative ingestion wall time | 61.190 s | 53.735 s |
| Cumulative ingestion CPU | 25.706 s | 24.031 s |
| Backup wall time | 0.611 s | 0.672 s |
| Restore wall time | 0.674 s | 0.884 s |
| Whole scale run wall time | 95.462 s | 73.983 s |
| Whole scale run parent + child CPU | 58.716 s | 43.729 s |
| Maximum parent RSS | 63,192 KiB | 62,756 KiB |

Writes and backup/restore code were identical in the frozen builds, so their
wall-time differences are reported as run variation, not credited to the lexical
optimization. The unchanged schema and exactly equal final byte counts avoid a
hidden disk-space tradeoff. This is not a hardware power-loss test.

### Same database, 50k sources in one scope

The fixture has 200 ordinary sessions, 50 accepted facts, and 5,000 sources with
non-ASCII additions. Each query has one warmup and five measured trials. Arm
order alternates; the table reports medians, with all samples retained in JSON.
The label `all-records-ascii` describes the query: 10% of matching sources contain
non-ASCII text. The single-scope database occupies 38,350,848 bytes.

| Query form | Baseline (ms) | Candidate (ms) | Observed ratio |
|---|---:|---:|---:|
| Early unique token | 0.162 | 0.157 | 1.03× |
| Middle unique token | 0.476 | 0.233 | 2.04× |
| Late unique token | 0.170 | 0.165 | 1.03× |
| All-records ASCII query | 1,066.248 | 519.776 | 2.05× |
| All-records multiple terms | 1,053.979 | 538.400 | 1.96× |
| Selective phrase | 14.278 | 8.257 | 1.73× |
| Unicode-only subset | 165.527 | 162.694 | 1.02× |
| Mixed broad query | 1,021.564 | 507.211 | 2.01× |
| Absent token | 0.352 | 0.235 | 1.49× |

The fresh-connection first broad query was 1,052.037→511.768 ms. OS pages were
not flushed, so this is not a cold-device benchmark. The Unicode-only subset
shows why the gain must not be generalized to all languages or content. Broad
recall still examines matching candidates and remains about half a second in
this 50k single-scope workload. Further scaling work remains necessary.

The three measured jobs consumed **149.943 seconds of aggregate parent/child
CPU** and **239.647 seconds of aggregate wall time**, below the registered
600/1,200-second limits. No training, paid service or remote compute was used.

## Atomic exact-key snapshot for model integration

After the retrieval candidate was frozen, the learned archive adapter needed a
single-snapshot exact-key lookup. `fact_snapshot(key)` returns `key`, `revision`
and a fact `record` or `None` from one scoped SELECT. Absent keys use revision0;
erased keys retain the deletion revision. Accepted JSON null remains a real
record whose value is null, distinct from absence.

`append_message(expected_fact_revisions=...)` now permits an answer based on
stable absence. It rejects the answer if insertion, erasure or recreation changes
the observed revision before commit. Existing live dependencies still reject
erasure because erasure increments the revision. Three new tests exercise these
races, accepted null and cross-principal isolation. Root reviewed the revision
semantics before final integration.

This helper does not alter the timed lexical paths. The verifier compares their
ASTs in the frozen timing candidate and final combined store. The final combined
store passes the 71-test regression suite, but the separate frozen-build timing
hash is reported honestly below. Model-level integration is reported separately.

## Source identity and reproduction

| Build | Store SHA256 |
|---|---|
| Baseline commit `ffb842adc176496d8838b774d0fe412aec5d1d20` | `3bf7561d194e19cc32a6ab4377ca4656aea86800ce6d026fd0a291ab18ec47c1` |
| Frozen retrieval-only timing candidate | `3e701337c26bfad404bb98c882bfe0638efb88b826576b9725e6be715c428f3c` |
| Final store with atomic fact snapshot | `89dc05ae971c52b73b65bf37df1349ef16473528d1f4a8c695a4c891f0ea6085` |

The baseline was extracted with `git archive` from the named commit. The timing
candidate copied that package and changed only `store.py`; `retrieval-only.patch`
preserves the exact patch needed to reconstruct it. Every other memory-package
file was identical. `environment.json` records the local frozen runtime paths.

From the repository root, each scale arm is reproduced by setting
`NANOCOG_VALIDATION_RUNTIME_ROOT` to its frozen `research/nanocog-001` directory,
setting `TMPDIR` to the recorded workspace directory, and invoking
`research/nanocog-001/memory-validation-002/scale/run_scale.py --output <new-file>`.
Never overwrite the original evidence when repeating a measurement.

The paired harness takes `--baseline-root`, `--candidate-root`, `--work-dir` and
`--output`. `verify_results.py --candidate-root <frozen-candidate-runtime>` validates
the retained evidence and writes `verified-summary.json`. It fails if the final
store no longer has the reviewed hash. The exact pytest invocation and observed
result are recorded in `test-receipt.json`.

Evidence files: `baseline-scale.json`, `candidate-scale.json`,
`paired-single-scope.json`, `verified-summary.json`, `test-receipt.json`,
`environment.json`, `PROTOCOL.md`, `retrieval-only.patch`, `run_compare.py` and
`verify_results.py`.
