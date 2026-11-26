# MEMORY-INTEGRATION-003 retrieval cost protocol

## Question and fixed semantics

Can the scope-independent exact lexical ranking introduced in MEMORY-VALIDATION-002
be served more cheaply without changing any ranked result, candidate set, source
provenance, context budget, scope rule, write, erasure or persistence behavior?

The baseline is commit `ffb842adc176496d8838b774d0fe412aec5d1d20`, storage schema 2.
Its source is frozen outside the repository before production edits. This is an
engineering optimization, not a learned model or semantic-retrieval experiment.

The first candidate targets a specific unnecessary cost: `_word_tokens` performs
Unicode decomposition and a Python combining-character scan for every candidate
even if the case-folded string is entirely ASCII. For an ASCII string, Unicode
NFD leaves every character unchanged and every combining class is zero. Therefore
the same token regex can be applied directly, with exactly the same result.
Non-ASCII inputs keep the old algorithm. Compile the unchanged token regex once.

There is no new index, result cache, schema migration, shortened candidate list,
different matching rule, changed ranking score or weaker integrity check.
If this small exact optimization is insufficient, its results must be reported
before choosing a different mechanism; do not quietly add an approximate top-k.

## Verification before performance interpretation

- Compare token sequences and score results with the literal baseline algorithm
  for ASCII, case-fold-to-ASCII characters, combining accents, non-Latin scripts,
  punctuation, underscores, repeated words and deterministic random Unicode.
- Compare actual ranked messages and full recall results with the frozen baseline
  on the same database across query limits, scope filters, session filters,
  excluded events, accepted-fact corrections, source deletion and reopening.
- Preserve the original memory regressions, including foreign-corpus invariance,
  direct-fact priority, rollback and erasure. A speedup cannot excuse a mismatch.

## Cost measurements

Use the unchanged prior scale runner and its deterministic 1k/10k/50k operation
stream, with baseline and candidate packages frozen independently. Run on the
workspace overlay (`TMPDIR` explicitly under `/workspace`), not default tmpfs.
Record source/harness hashes and filesystem identity. Check all original sources,
facts, historical revisions, fresh-process and backup-restored results, all CLI
commands and the supplied temporal reference. Compare ingestion, exact/broad
recall, total CPU/wall time, backup, restore and database size. The writes and
schema are unchanged; do not hide their measured costs.

Additionally measure repeated, interleaved baseline/candidate queries against
the same populated database, including a 50k-message single scope, mixed queries,
and a Unicode-heavy subset. Use one untimed warm-up and five measured trials per
query; alternate order to reduce drift. Include exact returned-record equality
on every trial. Preserve raw times and report medians, not selected best samples.
Report fresh connection first-query timing separately; do not call warmed reads
an uncached disk benchmark.

This is a bounded local CPU study, with no paid services, model training or remote
jobs. Target no more than 600 seconds CPU and 1,200 seconds wall for all new
measurements; report any skipped gate. Shared-host timings are descriptive, not
a universal or matched-language-quality speed claim. No claim of lifelong
capacity or synonym understanding follows from exact lexical ranking.

## Coordinated integration helper

After the retrieval candidate was frozen for timing, the learned archive adapter
requested an atomic exact-key read. Add `fact_snapshot(key)` using one scoped
SELECT and allow expected fact revisions to represent stable missing (revision0)
or erased (retained deletion revision) states. An answer based on absence must
fail if insertion or recreation changes that revision before commit. Existing
live dependencies still reject erasure, because deletion increments revision.
This helper does not change lexical ranking, schema, matching candidates, or
ordinary writes without fact dependencies. Validate it separately with absence,
insertion, erasure, recreation, accepted-null and cross-scope tests. Report its
final combined source hash distinctly from the frozen retrieval-only timing
candidate; do not pretend they are byte-identical builds.
