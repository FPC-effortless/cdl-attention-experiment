# Changing State 001 — prospective development diagnostic

Date: 2026-11-13. Status: local prospective protocol, pending independent review;
not an admitted cross-architecture campaign or scientific qualification.

## Question and falsifier
Does the recovered CSM-ADDRESS-002 controller retain correct current bindings
when repeated changes replace its training distribution's single overwrite?
Frozen weights, original vocabulary, four active identities and eight values.
Failure means any previously successful supervised seed falls below 95% answer
accuracy or 90% complete-world accuracy on any new stress panel. These are
engineering continuation screens, not significance tests or novelty claims.
A pass supports only repeated SET under this grammar, not deletion/composition.

## Fixed diagnostic before scoring
- Exact published parent: 3edc4f1a541a3221390e4c0a21fe0c12004f1ca5.
- Recovered archive SHA256: e4a3d3e39e6e8aa3e6f4f3c6c76b705dacf310be4adb8fc82f068712e678b9f2.
- Seeds: original checkpoints 101, 202, 303. No fits or parameter updates.
- New public DEVELOPMENT worlds, RNG 131126; 64 families, paired across depths
  1, 4, 16, 64, 256 additional SETs after four distinct initial values.
- Two regimes: cycle through all four identities; repeatedly change one identity
  while preserving the other three. Every overwrite must change its current value.
- Random identity order. Each eight-family block shifts every value by 0..7,
  giving exactly balanced answers. Four independent query replays per world,
  four NOP tokens, original encoder. Query replays are NOT a persistent multiquery
  session. Eight value shifts share one base block; there are only eight base blocks.
  Use base block as the independent data cluster; depths/regimes are paired.
- Exclude complete register-renaming-canonical histories present in the archived
  TRAIN bank. No protected TEST generation, inspection or tuning.
- Arms: frozen supervised CSM, raw CSM and supplied-controller CSM. Supplied
  routes are oracle diagnostics. Historical Transformer only on <=128-token
  panels for this bounded diagnostic; longer panels explicitly unscored, not
  failed or silently truncated. Its answer-only training is not equivalent.
- Reference: perfect structured current-value store, explicitly an oracle parser
  diagnostic. First-value and last-global-value controls detect metric weakness.
- Inputs to learned arms: token IDs only. Targets and semantic metadata remain
  outside forward except the explicitly supplied-controller arm.
- Metrics: per-seed answer accuracy; all-four-query world correctness; stale
  immediate-predecessor errors on changed identities; untouched-query accuracy;
  destination and admission diagnostics. Report denominators and all seeds.
- Runtime receipt: process, Python/Torch/NumPy, thread count, source/data/checkpoint
  hashes, weights unchanged, elapsed evaluation time. Instrumented batch replay
  is NOT session-cost benchmarking. No inference speed or peak-memory claim.
- Continue only after semantic tests and independent pre-scoring code review.
  Stop on nonfinite outputs, changed source/checkpoint, or malformed fixture.
- Any post-freeze changes or failed attempts must be retained and described.

## Shared lifecycle contract implemented in this increment
Typed records: VALUE(v), LINK(k), or absent. SET replaces a record; DELETE removes
it; LINK replaces it with a live reference; COPY resolves the source at that
instant and stores an independent value snapshot (absent if unresolved). NOP
has no effect. QUERY follows live links; absent targets or cycles yield UNKNOWN.
COPY of an unresolved source removes the destination. Self-COPY resolves before
mutation. Queries have no semantic mutation. A live dangling link can resolve
again if its target is recreated; a copied UNKNOWN cannot. These choices are
explicit benchmark semantics, not assertions about all applications.

The pure state-map implementation is checked against a separately implemented
backwards event-history interpreter. This is a benchmark sanity check, not a
learned model result. The original CSM has no DELETE/LINK/COPY training in this
checkpoint; it is not scored on those operations here.

## Next admitted comparison (not executed by this diagnostic)
CSM, supervised MemNet, tuned cached Transformer and current-state store with a
shared learned extraction interface; later add a compact delta-rule/SSM control.
Two tracks: answer-only and equivalent semantic process labels. Give all models
entity, valid-update, support and supersession information in the process track;
never grant CSM physical-slot labels while denying equivalent binding targets to
competitors. Oracle parser/route arms remain separate. Record tuning and failed
runs. Freeze a concrete budget, all seeds, splits and resource admission before
new fits; do not describe this outline as a completed campaign preregistration.

Vary history length independently of active state. Score stale answers, deletion,
complete state, untouched preservation, COPY-vs-LINK consequences and dependent
queries. Run joint/staged/staged-plus-joint at equal total exposure. Future held-out
families must be private/disjoint and never derived by renaming these public dev
families. Bootstrap by history family within seed; do not pool correlated queries
or treat three seeds as enough for broad population claims.

Cost: time full sessions from input encoding through writes, queries, maintenance,
verification and retries; account for cache construction and warmup; record cold
and warm runs, retained bytes and peak memory separately. Prospective quality
floor: >=99% answers, <=1% stale errors and >=99% untouched preservation in EVERY
required stratum/seed before claiming lower cost. Comparators need saturated or
adequately characterized learning curves, a declared tuning budget and repaired
implementation failures. No qualified advantage is possible from this diagnostic.

Restore/ownership admission stays with MEM-ADMISSION-038 and DPSL-RECEIPT-039:
wrong restored state/shared backing/partial mutation is a failure, missing native
observation is UNKNOWN, and read/association/output are separate. This benchmark
does not replace their native observer or authenticate their private receipts.
