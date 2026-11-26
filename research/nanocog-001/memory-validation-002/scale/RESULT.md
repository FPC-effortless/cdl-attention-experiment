# MEMORY-VALIDATION-002 — archival scale, CLI and temporal evidence

## Result

The registered 1k/10k/50k workload completed against both the original implementation
and the repaired implementation on both available filesystems. Each expanded run
has **14,630 assertions and 0 failures**, including **14,577 storage assertions,
43 CLI assertions, and 10 supplied-temporal-reference assertions**. No stage was
skipped, no implementation changed within a run, and all four runs used the same
harness and identical generated operation-stream hashes. These are synthetic
infrastructure checks. No trained writer, reader, temporal binder, or language
model supplied any prediction in this workload.

The broader validation found defects elsewhere; a clean exact-token workload did
not mean the original implementation was correct in every adversarial case.
This report measures the archive behavior and costs of the subsequently repaired
store without substituting those results for model quality.

## Workload and replay accuracy

The final stage contains 50,000 ordinary source messages plus one lexical-control
message, across eight principal/project scopes and 1,600 ordinary conversation
identities (plus the control conversation). Explicit host acceptance creates
1,613 historical fact revisions across 256 current keys: 1,357 revisions are
corrections after the first revision of each key. There are 53,227 mutating API
calls in total: source appends, proposals, acceptances, and the control append.
The live database after checkpoint is 44,023,808 bytes;
the remaining shared-memory sidecar is 32,768 bytes
and the WAL is truncated to zero. Storage size includes indexes and metadata.
At 50k total messages, each queried scope holds 6,250 ordinary messages; this
workload does not measure a 50k-message single-user/single-project archive.

Across the three stage sizes and three access modes (same process, fresh process,
and fresh process using a restored backup), the final overlay run produced:

* **432/432 exact source-retrieval probe instances**. Probes cover early,
  middle, late and seeded-random positions in all scopes. These include repeated
  queries across restarts and are not independent language-understanding examples.
* **1635/1635 current-fact recall instances**, with exact value, source and
  revision checks. Every current value also passes a value-query check.
* **5,907 historical-revision checks** against generated values, source contents,
  proposal IDs, scope/key mapping, revision order and host authority. Replays
  repeat histories, so this count does not mean that many distinct revisions.
* **216 foreign-scope probes** returning no source or fact, and **216 rendered
  context-cap checks** within the configured character limits. Scope filtering
  is a local API property; this does not evaluate user authentication.

The literal `car` control is recalled by `car` and missed by `automobile` in every
stage/access mode. This is the predeclared lexical limitation control. Successful
exact-token lookup therefore cannot be described as reliable natural-language
recollection. The source and facts are generated together from known synthetic
inputs; this says nothing about whether a learned model interpreted a sentence
correctly before writing it.

## Primary performance evidence: workspace overlay

Cells showing two measurements list **baseline / final repaired**. Latencies
below are same-process observations, without flushing OS caches. Source latency
uses 48 unique-token probes per stage; the common-word context median uses eight
scopes and a query matching every ordinary observation in its scope.

| Sources | Cumulative ingestion wall, s | Exact-source median, ms | Common-word 8k-char context median, ms | Final DB bytes |
|---|---|---|---|---|
| 1,000 | 0.98 / 1.25 | 0.214 / 0.279 | 6.75 / 10.42 | 1,159,168 |
| 10,000 | 8.71 / 10.82 | 0.169 / 0.250 | 12.79 / 39.88 | 9,011,200 |
| 50,000 | 47.40 / 54.47 | 0.200 / 0.303 | 41.30 / 202.17 | 44,023,808 |

The scope-independent ranking repair carries a retrieval cost. At 50k, broad-query
context latency changed from 41.30 ms to 202.17 ms
in the matched overlay runs. The repair's Python lexical scoring examines matching
candidate content; removing foreign-corpus ranking influence does not make that
work free. These single-pass measurements are descriptive observations, not an
isolated statistically controlled speed comparison or an asymptotic guarantee.
Other validation processes could share the host.

The 256-character context cap returned only the 86-character empty envelope
in every scope at 50k, because no complete source-labelled record fitted. Its
median lookup time was still **199.44 ms**. A small output cap does not guarantee
that a useful record fits and does not bound the work spent finding candidates.
The 8,000-character cap is a character budget, not a tokenizer or inference budget.

Final 50k-stage extra operations are measured separately below. Backup is the
entire database with all eight scopes; JSON export is only scope zero.

| Operation | Wall seconds | CPU seconds |
|---|---|---|
| Full database backup | 0.664 | 0.638 |
| Verified restore to a new path | 1.019 | 0.982 |
| One-scope JSON export | 0.126 | 0.123 |
| Complete 50k-stage validation in parent | 5.393 | 5.358 |
| Fresh-process complete validation | 5.166 | Included in total child CPU |
| Restored-database complete validation | 6.568 | Included in total child CPU |

The backup file is 44,023,808 bytes and the one-scope JSON export is
3,115,711 bytes. Backups and restored databases share the test
filesystem; this is verified copying and reopening, not independent off-device
disaster recovery. Temporary data files are removed after each run.

## Full workload costs and filesystem qualification

`df -T` identifies `/tmp` as **tmpfs** and the workspace as **overlay**; the raw
command outputs are retained in `environment.json`. tmpfs timing measures a RAM
filesystem, not durable-disk fsync performance. Overlay timing measures the
available container filesystem, not a specified SSD or proof against power loss.
Python is 3.12.14, SQLite is
3.53.1, journal mode is wal,
and SQLite synchronous is 2 (`FULL`).

The full run includes ingestion, oracle checks, recall, context generation,
scope export, backup, restore, fresh-process checks, CLI subprocesses, temporal
reference work and temporary-data cleanup. CPU totals include child-process CPU.

| Run | Total wall, s | Total CPU, s | 50k ingestion wall, s | 50k common context median, ms | Assertions / failures |
|---|---|---|---|---|---|
| Baseline overlay | 59.69 | 32.87 | 47.40 | 41.30 | 14,630 / 0 |
| Final overlay | 84.27 | 56.29 | 54.47 | 202.17 | 14,630 / 0 |
| Baseline tmpfs | 21.91 | 21.86 | 10.54 | 41.33 | 14,630 / 0 |
| Final tmpfs | 33.79 | 33.66 | 10.09 | 180.58 | 14,630 / 0 |

For the final overlay run, parent maximum RSS is
61,828 KiB; the largest child-process reported maximum is
61,828 KiB. These high-water marks are not summed and do not
measure filesystem page cache. The source payloads, SQLite pages, indexes,
backup bytes, generated contexts and interpreter computation all contribute
different costs. No model inference or training cost appears here.

## CLI coverage

Actual subprocesses exercised `record`, `recall`, `propose`, `accept`, `facts`,
`forget`, `forget-session`, `export`, `backup`, and `restore`. Checks cover
cross-invocation retrieval, a revised fact accepted with its expected revision,
authority provenance, cross-project filtering, source erasure, preservation of
the newer independently sourced correction when its older source is erased,
generated IDs, new-target backup/restore behavior, refusal to overwrite existing
backup/restore files, malformed/nonfinite JSON, invalid retrieval limits, explicit
database paths, `COG_MEMORY_PATH`, absolute `XDG_STATE_HOME`, and explicit-path
precedence. These 43 assertions passed. This is a local command-line
integration check, not a deployed service, authentication flow or browser test.

## Supplied temporal bindings

A 10,000-edge chain with shuffled claim order and reversed source-map order
remained consistent. Forward and reverse queries returned the expected relation
and all 10,000 provenance sources. A unique selection worked; competing matching
candidates, an unbound alternative, and a cycle did not produce a guessed event.
Invalid source quotations were rejected. Construction took
109.99 ms; the forward query took
10.27 ms on the final overlay run.

Twenty bindings were also persisted through ordinary accepted JSON facts and
reloaded in a fresh process. Outer host acceptance alone left the internal
temporal claims `proposed`, so the reference returned `UNKNOWN`. Explicitly
accepting those internal claims after source-span validation yielded `BEFORE`
with all 20 sources. This preserves the two distinct policy decisions. All ten
temporal assertions passed; none learns cue attachment or improves the learned
PHASE-003 selector.

## Reproduction and source identity

Run from `research/nanocog-001`:

```bash
python memory-validation-002/scale/run_scale.py --output /absolute/path/results.json
```

Set `TMPDIR` to an existing directory on the filesystem being measured. The harness
uses bounded temporary databases and deletes them on exit. The registered seed
is `271828`. `NANOCOG_VALIDATION_RUNTIME_ROOT` optionally selects a pinned copy of
the `nanocog` package. The retained result JSON includes every source hash,
environment, operation-stream hash, probe count, failure detail, latency summary,
resource total, and stage-specific file size. The full stage output is larger than
this summary and remains the primary numerical evidence.
The benchmark preserves diagnostic JSON even on a failed assertion or execution
error; successful process exit alone is not its acceptance signal. Inspect the
failure lists and execution-error field. `write_report.py` refuses to summarize
the four named final comparison files as passing if any contains a failure,
skipped stage, mixed source hash, or mismatched generated inputs.

Baseline `store.py` SHA256:
`fe694bcd923a4167a97c237e5ae4bfb279e7cccc92b54d542943258ef413e3d7`.

Final tested `store.py` SHA256:
`3bf7561d194e19cc32a6ab4377ca4656aea86800ce6d026fd0a291ab18ec47c1`.

The baseline is the published PR head
`56cb4eeb04d68f5f6c210c756f0790f34cc2fd5f`; the repaired working-tree implementation
is pinned by the hashes above and in the JSON, rather than by assuming that a
mutable branch name identifies exact bytes. The final temporal and session files
are unchanged from baseline.

The initial baseline, intermediate repaired tmpfs run and pre-startup-retry
repaired overlay run are retained for audit
but superseded by the four expanded comparison runs listed above. The initial
baseline used the narrower registered first-pass checks; the amendment added
historical value/source checks and explicit CLI correction/filtering before the
paired final runs. The filesystem amendment is recorded in `PROTOCOL.md`.

## Boundaries of the result

This workload establishes tested finite behavior at up to 50k generated messages.
It does not establish lifetime capacity, external backups, failure of the host or
disk, concurrent production load, natural-language memory selection, trained
writer correctness, useful influence on an actual generated answer, privacy
against an adversary controlling the database, or superiority to attention.
Whole-history correctness here means exact storage and retrieval of supplied
structured facts; the earlier learned writer problem remains a separate
qualification task.
