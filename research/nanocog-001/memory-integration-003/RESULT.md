# MEMORY-INTEGRATION-003 — real model integration, failed fresh-wording qualification, lower infrastructure cost

## Outcome

The archive now has a working **bounded learned factual interface** over the
user's trained COG_CONNECTED_1M model. It survives a fresh process and a new chat,
and relevant accepted facts change its answers when the learned question route
is correct. Its complete learned memory path **does not qualify on fresh wording**.
The retrieval and reference-decoder optimizations separately reduce measured
overhead while preserving the checked outputs and safety invariants.

| Requested outcome | Finding | Decision |
|---|---|---|
| Accepted facts influence actual learned answers | Familiar causal panel passes; a fresh passive question form routes three of eight questions incorrectly even with a correct ledger | Keep the connection, retain explicit scope and query failures |
| Whole learned history and both counterfactual answers correct, familiar accuracy preserved | Every arm/seed scores 256/256 familiar pairs and 0/256 fresh pairs; candidate loses zero control-correct familiar pairs | **Fresh-wording qualification fails; do not promote the writer** |
| Lower serving cost without correctness loss | Broad retrieval improves about 2× on the stronger single-scope test; reference warm persistence improves in the prespecified quiet pass, with all prior checks retained | Keep the measured engineering improvements; no overall trained-model efficiency claim |

This is a new experiment on the current trained Connected COG checkpoint, not a
numerically comparable continuation of the older PULSE/ANT panels. It does not
repair the separate noun-selection or temporal-binding experiments by inference.
The baseline repository commit is
`ffb842adc176496d8838b774d0fe412aec5d1d20`; old evidence is unchanged. The work
continues on the existing draft `feat/durable-memory-001` branch / PR #94.

## What was built

The general archive still stores scoped observations, proposals and accepted
facts with source provenance and revisions. The reference `PersistentDecoder`
continues to consume its supplied tokens; this work does not secretly inject
archive contents into that model's language stream.

A separate experimental `FactualMemory` mode now connects the archive to trained
decisions. For each raw event, the actual Connected COG backbone and record board
produce character features. A 32-unit GRU and joint operation head choose NOOP or
a record/value write. A host policy may accept a valid typed prediction; acceptance
means the policy accepted it, not that the interpretation is true. For each raw
question, a learned route selects a record; exact typed traversal obtains its
owner value, and a learned answer head chooses a name or `unknown`. The archive
is therefore a causal input to this answer mode.

| Component | Behavior | Parameters |
|---|---|---:|
| Complete admitted Connected COG language model | Frozen native features through all six blocks, connected board and final normalization | 993,574 |
| New GRU and writer/query/answer heads | Learned event interpretation, question route and answer label | 18,567 |
| Total model | All native weights remain present | **1,012,141** |
| SQLite, schema and traversal | Deterministic infrastructure; no learned parameters | 0 |

The declared experiment has four boxes, four items, eight possible people and
eight record keys. The writer jointly scores 97 possibilities: NOOP plus eight
addresses times twelve values. Only 49 possibilities are schema-valid. Invalid
proposals are rejected, never repaired using a gold label. Inputs are one to 256
characters from the admitted 65-character tokenizer. Native transient features
reset for each event/question; the durable ledger carries event history. Native
latent slots are not treated as semantic record identities. No diagnostic route,
commit or value override supplies inference labels.

This is a categorical experimental interface, not an open-vocabulary chat memory
system. The general store can hold other keys and values, but this adapter cannot
interpret arbitrary facts. The original language-generation path is unchanged.

### Transaction and identity behavior

New `fact_snapshot(key)` reads a scoped record and revision from one SQL snapshot.
Absent keys use revision zero; deletion retains an incremented tombstone revision.
An answer records every actual read dependency, including absence, and revalidates
all revisions and source liveness in its commit. A correction, insertion, erasure
or recreation during computation cannot silently produce a newly acknowledged
stale answer.

The writer captures the bounded set of candidate record revisions before model
inference. Its chosen key must still match the original revision when accepted.
An immutable decision receipt lets an interrupted request retry the same decision
without rebasing old text onto a newer fact. This is compare-and-set concurrency,
not a total event-time order across concurrent chats. A decision receipt records
the requested operation; only a successful acceptance and returned WriteResult
establish that the write committed.

The fitted adapter binds the native source/checkpoint/config/tokenizer admission,
ontology and all model tensor bytes. Full identities are checked before and after
inference and before receipt replay; tensor mutation, training mode and local or
global inference hooks are rejected. Admitted host code remains a trust boundary.

Independent review first reproduced a global-hook bypass and a revision captured
too late in the writer. Both fail against the retained original source and pass
after repair. The exact review and fixtures are in [review/REVIEW.md](review/REVIEW.md).

## 1. Accepted facts now affect actual model behavior — within the measured scope

An independent audit used the prospectively selected **candidate seed17**, not
the best-performing seed. It loaded the real fitted checkpoint and exercised
`FactualMemory`, including fresh processes. The causal panel supplied a correct
accepted ledger to isolate query/answer use; it is not writer-accuracy evidence.

| Question panel | Correct routes / owner-change pairs | Unrelated-answer invariance | Correct item-link-change pairs |
|---|---:|---:|---:|
| Three exposed forms combined | 24/24 | 24/24 | 12/12 |
| Fresh passive form | 5/8 | 7/8 | 1/4 |
| Other three fresh forms combined | 24/24 | 24/24 | 12/12 |
| Total | 53/56 | 55/56 | 25/28 |

The unrelated readset counts match the answer-invariance counts. The passive
question “By whom is the container holding the coin owned?” chose the **key**
record instead of the coin. Changing the coin's relevant owner therefore left
the wrong answer unchanged. For the analogous watch question, changing the key's
owner—irrelevant to the intended question—changed the answer. Storage supplied
exactly the records requested; the question route was wrong.

An independently hand-authored eight-event transfer/cancellation pair passed
every learned operation, every post-event ledger and both answers. This is a
useful integration example, not a statistical generalization result. Three race
tests executed the real fitted answer head and then applied a verified concurrent
correction, erasure or insertion: **3/3 stale replies were rejected before saving**.
A fresh process/new chat matched the corrected answer and readset. Reusing the
same request returned its original committed historical receipt as designed;
a fresh request observed the correction.

See [integration/RESULTS.json](integration/RESULTS.json) and the independent
[audit script](integration/verify_fitted.py). This audit ran after candidate seed 17's
checkpoint was frozen while the other fixed-schedule fits continued. Its findings
did not change any remaining fit, selection, objective or evaluation threshold.

## 2. Complete learned-memory qualification failed

### Prospective design

The [protocol](learned/PROTOCOL.md), [data manifest](learned/DATA_MANIFEST.json),
[source lock](learned/LOCK.json) and [train-only calibration](learned/CALIBRATION.json)
were saved before scored fitting. Both recipes used the same native checkpoint
and matched initial head seeds **17, 29, 41**. Each completed exactly **2,048
updates** with batch128 and identical optimizer settings. The control saw
canonical phrasing; the candidate saw canonical, recipient-first and
completion/cancellation views with consistent role/value permutations.

There were 512 TRAIN, 128 DEV, 256 familiar TEST and 256 fresh TEST semantic
lineages. Every lineage has two counterfactual histories of 8, 24 or 64 events.
All views of a lineage stay together. The pair differs in one completed versus
cancelled transfer and requires different answers. Fresh TEST has four held-out
wording families with 64 pairs each. Historical ANT/PULSE TEST panels were not
used for selection. TEST was scored after the six fits froze; no repair fit
followed the results.

The strict endpoint requires every predicted operation and every post-event
ledger across both histories, plus both final answers, to be correct. Correct
answers alone, final-state recovery after earlier errors, or better route scores
cannot pass it. The scorer uses a typed in-memory ledger; an additional durable
sample verifies that actual full-native and SQLite execution match those scores.

### Primary results

| Recipe | Head seed | Familiar DEV strict | Familiar TEST strict | Fresh TEST strict | Fresh both answers | Fresh both query routes |
|---|---:|---:|---:|---:|---:|---:|
| Canonical control | 17 | 128/128 | 256/256 | **0/256** | 0/256 | 136/256 |
| Matched candidate | 17 | 128/128 | 256/256 | **0/256** | 9/256 | 232/256 |
| Canonical control | 29 | 128/128 | 256/256 | **0/256** | 0/256 | 152/256 |
| Matched candidate | 29 | 128/128 | 256/256 | **0/256** | 0/256 | 192/256 |
| Canonical control | 41 | 128/128 | 256/256 | **0/256** | 0/256 | 80/256 |
| Matched candidate | 41 | 128/128 | 256/256 | **0/256** | 0/256 | 160/256 |

Every recipe/seed also scores 512/512 canonical TRAIN pairs. The candidate loses
zero control-correct familiar TEST pairs, so familiar preservation passes this
matched comparison. Fresh qualification fails in every seed, history length and
grammar subgroup. A 0/256 result has a per-seed Wilson95 interval of 0–1.48%; the
three seeds reuse the same families and must not be pooled as 768 independent
examples. These are three adapter initializations on one pretrained backbone.

A separate post-fit replay of all exposed candidate TRAIN views clarifies fit
quality. Seeds 17 and 41 score 1,536/1,536 strict pairs across the three trained
forms. Seed29 scores 512/512 canonical, 388/512 recipient and 452/512 completion
pairs: it additionally underfits part of its own augmented training data. Thus
the failure cannot be described as only underfitting across all seeds, nor can
seed 29's weakness be hidden by the perfect canonical panel.

### What breaks

An independent pure-Python replay reconciled **all 6,912 original pair rows**
exactly against frozen exported predictions. It added diagnostics without new
model calls, fits, selection or oracle correction. For fresh candidate seed 17, across
16,336 event decisions in 512 histories:

| Mutually exclusive wrong-operation category | Canonical control | Matched candidate |
|---|---:|---:|
| Required write omitted | 11,695 | 5,573 |
| Write proposed when no write was required | 262 | 997 |
| Both propose a write, wrong address only | 260 | 883 |
| Both propose a write, wrong value only | 61 | 1,249 |
| Both propose a write, wrong address and value | 526 | 3,050 |

The candidate misses fewer writes but introduces more of the other mistakes.
No schema-invalid proposals occurred. Schema rejection therefore does not explain
the failure. Candidate first-event errors occur in **386/512, 464/512 and 366/512
histories** for seeds 17/29/41. Initial fact establishment is already unreliable;
long-history accumulation is not the sole explanation.

Later writes also damage previously correct, unchanged records: candidate counts
are 698/834/811 compared with control 108/145/16. These are event-record counts,
not exposure-matched rates: a method must first establish a correct record before
it can damage one. The diagnostic retains the corresponding history positions
and denominators.

Candidate seed 17 has nine pairs with both required final chains correct and nine with
both answers correct, but their **intersection is only eight**. Equal marginal
counts do not identify the same cases. Its strict complete-history result stays
zero. Full subgroup/component evidence is in
[review/DIAGNOSTICS.md](review/DIAGNOSTICS.md) and
[learned/RESULTS.json](learned/RESULTS.json).

Fresh forms change task vocabulary as well as syntax. These results locate a
wording-transfer failure; they do not isolate word order, lexical semantics or
binding as its sole cause. They also do not establish that Connected COG is
better or worse than a matched simpler encoder, which was not tested here.

### Native-language regression and real durable parity

The admitted kit's 10 manifest entries and research archive's 218 entries passed
byte/hash verification. The selected native checkpoint is SHA256
`d8bab3d1d99a43dfb6cca2257f2caae4d3a7282d62b079c92874ce736a44ec3c`.
An independent full validation reproduced **1.595178123236727 nats/character**
exactly on all 55,769 already-exposed targets. Separate original and attached
native model objects had bit-identical logits and complete recurrent states.
All six fitted checkpoints retain every native tensor unchanged. This preserves
the original language path; it does not make its original validation a fresh test.
See [native/validation.json](native/validation.json).

The prespecified candidate seed 17 durable sample covered 16 pairs, **1,152 event
decisions and 32 answers**. Cached-feature scoring, uncached native inference
and actual SQLite execution agree on every sampled path. All eight familiar
pairs pass; all eight fresh pairs fail. Reopening a connection/new chat preserves
the same ledger. Correct persistence faithfully preserves incorrect learned
decisions too. The root audit supplies the separate fresh-process check.

## 3. Infrastructure costs fell, but efficiency remains bounded

### Exact archive retrieval

ASCII text no longer undergoes redundant Unicode decomposition and per-character
combining scans. Non-ASCII retains the prior normalization. Candidate sets,
ranking, tie order, scope, fact priority, provenance, erasure and context budgets
remain unchanged; there is no approximate top-k, cache, index or schema change.

| Workload | Baseline median | Candidate median |
|---|---:|---:|
| Original 50k total / eight scopes, broad 8,000-character recall | 215.406 ms | 88.003 ms |
| Same database, 50k sources in one scope, broad ASCII query | 1,066.248 ms | 519.776 ms |
| Same single scope, multiple broad terms | 1,053.979 ms | 538.400 ms |
| Same single scope, Unicode-only subset | 165.527 ms | 162.694 ms |

Both frozen scale builds pass **14,630 assertions**; all 54 paired same-database
comparisons across nine query forms return exactly equal full objects. The original scale database has the
same 44,023,808-byte size. The Unicode result limits the speed claim. Broad recall
still examines matching candidates and takes about half a second at 50k sources
in one scope. See [retrieval/RESULT.md](retrieval/RESULT.md) for raw samples,
ingestion/backup/restore costs and the separately identified snapshot helper.

### Reference-decoder persistence

Full tensor bytes are still hashed before/after inference and before replay.
Canonical byte views avoid copies, live metadata traversal avoids redundant
lookups, and attachment verifies finiteness. A later complete matching byte hash
establishes the same finite contents under the existing checksum assumption;
tensor version counters are not substituted. `.data` changes, NaN/Inf mutations,
descriptors and inference hooks remain tested. Manifest-v2 bytes and old-session
continuations stay compatible.

At 1,024 prior tokens plus one new token, the quiet confirmation measured:

| Reference fixture | Baseline full persistent call | Candidate full persistent call |
|---|---:|---:|
| Local | 46.815 ms | 31.733 ms |
| Packet | 48.091 ms | 46.530 ms |
| Packet + typed | 73.512 ms | 45.449 ms |

Warm timings exclude process/adapter startup and prefix construction. Each case
has one warm-up and five measured trials on one shared CPU host. Cold timings
are single new-process observations with OS caches unflushed.

All 12 warm-case medians improved in that prospectively specified confirmation;
the original pass, including two regressions, is retained. A typed cold-process
observation regressed. All 12 complete output/checkpoint comparisons per pass and
all three old-session continuations match. These seeded untrained reference
fixtures measure engineering overhead, not the trained Connected COG mode.
Durable calls remain more expensive than RAM-only continuation. See
[serving/RESULT.md](serving/RESULT.md).

### Actual trained factual mode

The separate overlay-filesystem sample includes full native feature computation,
identity checks, provenance, receipt handling and synchronous persistence. It
uses the same fitted model/output path for raw and durable execution. The four
familiar pairs at each length are all strict-correct:

| Total history length | Raw event mean | Durable event mean | Raw answer mean | Durable answer mean |
|---|---:|---:|---:|---:|
| 8 events | 9.07 ms | 22.08 ms | 6.72 ms | 17.49 ms |
| 64 events | 9.03 ms | 22.67 ms | 8.23 ms | 19.01 ms |

Event means cover all events in each history; answer means are measured after
the final event.

This exposes substantial remaining overhead. It is not a trained-model speedup
against an alternative architecture. The sample is small and raw-then-durable
order is fixed. Fresh cases are slower and fail correctness; they cannot support
a matched-accuracy efficiency claim. See [learned/DURABLE_RESULTS.json](learned/DURABLE_RESULTS.json).

All six scored optimizer fits consumed 223.0075 CPU seconds. Calibration/extraction,
full fit phase, evaluation/remaining extraction, trace export and durable sample
together consumed 299.514 CPU / 304.105 wall seconds. Native extraction itself was
15.325 CPU seconds. Those measured phases exclude Python process startup and
initial dataset generation (6.681 wall seconds measured, CPU not instrumented),
as well as independent root/review checks and
the separate infrastructure studies. Caches are not counted as free inference.
The later augmented-TRAIN diagnostic replay adds 1.250 CPU / 1.255 wall seconds
outside those five phases.
No paid/remote training or deployment was performed.

## Final engineering gate and handoff

- **189/189 complete local package tests pass**, with zero failures, errors or
  skips. The optional archived PULSE source-only test validates its pinned source
  hashes; it is not a trained PULSE accuracy result.
- **9/9 system scenarios pass**, including concurrent opens/processes, interrupted
  transactions, correction/erasure and backup/restore behavior.
- **3/3 independent review regressions pass** after preserving their original
  failures. **4/4 fitted CLI process checks pass**, including a new chat reading
  a fact and a later correction in a different process.
- Source hashes are identical before and after the final gate. Exact local logs,
  JUnit and source manifest are in [verification/RESULTS.json](verification/RESULTS.json).
- The native validation, independent causal audit, learned qualification and cost
  studies have separate evidence. No test count is substituted for model accuracy.

The implementation is in `nanocog/memory/learned_archive.py`; the strict private
checkpoint loader and opt-in `factual_cli.py` are in `learned/`. Private fitted
weights, generated histories and detailed diagnostic traces are retained outside
public git. The public branch contains code, protocols, aggregate evidence and
reproduction scripts. The original default memory/native interfaces do not enable
this unqualified factual mode automatically.

The engineering changes are suitable for review. The learned candidate is not
qualified for unfamiliar wording, unrestricted conversations or lifelong autonomous
memory. Further model work should isolate event interpretation and question
reference failures with a fresh held-out gate, preserving the familiar baseline;
more storage, retrying the same prediction, or selecting a better internal score
would not repair the demonstrated failures. No merge or deployment was performed.
