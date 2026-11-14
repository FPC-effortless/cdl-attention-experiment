# CSM learns this lifecycle task better, but value reliability remains insufficient

LIFECYCLE-LEARNING-001 · 14 November 2026 · bounded supervised development diagnostic

Six fresh fits completed: CSM and a conventional pointer-capable slot model, seeds 101/202/303,512 updates ×32 answers per fit. Training data, seed-specific sampling schedules, process labels, loss weights and optimizer settings matched. CSM has 18,359 parameters; the control18,357. All fits closed before validation inference. No infrastructure implementation changes were made.

| Model | Seed101 query / complete map | Seed202 query / complete map | Seed303 query / complete map | Mean query | Mean complete map |
|---|---:|---:|---:|---:|---:|
| CSM |89.62% /61.54%|94.23% /80.77%|90.38% /69.23%|91.41%|70.51%|
| Pointer-slot control |85.58% /56.15%|75.58% /21.54%|79.81% /39.23%|80.32%|38.97%|

CSM is higher in every paired seed on this frozen development panel, by 4.04,18.65 and10.58 percentage points of query accuracy. This is evidence for a fixed-budget learning difference on the five supplied mechanisms, not a general architectural ranking or a tuned-baseline win. Neither model passes the preregistered95%-in-every-template/every-seed development screen.

## What actually learned

Both models answered every UNKNOWN-target query correctly:273/273 per arm across the three seeds. This includes deleted aliases and absence snapshots retained after source recreation. Both preserved all user-record values and pointers exactly during every normal evaluated query:1,560/1,560 per arm. Preservation measures a query's state effect; it does not imply that the stored values were correct before the query.

The remaining errors all concern concrete values. CSM correctly answered1,153/1,287 concrete-value queries(89.59%); the pointer-slot control980/1,287(76.15%). Overall accuracy includes the easier UNKNOWN cases.

| Known mechanism | CSM mean query accuracy | Pointer-slot mean query accuracy |
|---|---:|---:|
| Overwrite |88.78%|73.40%|
| Query during source absence |95.51%|87.50%|
| Save absent root, then recreate source |87.18%|81.41%|
| Detach alias by overwriting it |91.35%|77.24%|
| Delete alias |94.23%|82.05%|

A whole mechanism's accuracy includes all four records. Thus100% on a deleted-alias target or saved-absence target can coexist with errors on the source, copied value or unrelated fact. Each complete map consists of four independently replayed queries, not one unbroken four-query session; query preservation was separately checked.

## Failure localization

| Frozen-weight inference condition | CSM mean accuracy | Pointer-slot mean accuracy |
|---|---:|---:|
| Learned decisions, raw tokens only |91.41%|80.32%|
| Stored value vectors erased before query |6.47%|10.45%|
| Pointers replaced by identity links before query |85.58%|74.81%|
| Correct routes/edits/full gain supplied |91.47%|80.58%|

Erasure strongly reduces performance, supporting dependence on retained values. Pointer ablation also changes performance. The categorical source, destination, direct-second, link-use and link-edit decisions match supervised targets on all measured validation decisions. Value-edit decisions are100% except CSMseed 303(99.86%). The edit totals include many easy noncommit zeros.

Supplying correct controls also overrides continuous overwrite gain, yet produces almost no improvement. This points to the learned value-writing/copying/query-transformation/decoding path as the main remaining bottleneck. It does not yet distinguish poor stored representations from a poor query readout, nor prove that CSM's operator bank causes its advantage. The models differ in both the state-conditioned controller/operator machinery and transform width.

## Scope and method

Four identities, eight concrete values plus UNKNOWN; COPY and LINK sources are roots; QUERY follows at most one link. No COPY-through-alias, cycles, long chains, new identities or language. Training explicitly supervises semantic control decisions; normal evaluation receives only raw tokens. No answer-only result is established.

128 role/value assignments produce1,280 worlds. Four identical event-history collisions join two pairs of assignments, leaving126 subjects. One fixed Forge split gives1,000 TRAINworlds/100 groups and130 validation worlds/13 groups. Every validation world contributes four queries. All five causal templates appear in TRAIN and validation. These are correlated examples of five known mechanisms, not130 independent tasks. The third bucket was not used for model inference, tuning or selection. Generated oracle semantics were verified before fitting. The old failed046/047 comparison is still unadmitted; its coverage thresholds were not changed.

The control is a fairer successor to the earlier SET-only slot model: it has source reads and pointer rows, so it can express live references. It drops CSM's active-state controller and operator bank, but widens its transform from 64 to 98 hidden units to match parameter count. Matching examples, steps and parameters does not match FLOPs or elapsed compute. No inference-cost advantage was measured. Training wall time is descriptive only and varied on the shared host.

## Verification

Six source tests passed. Native CSM source is byte-identical to the recovered implementation, and the instrumented forward matches it exactly. Independent prefit checks verified5,124 privileged transport/query-preservation cases per arm; these check representability, not learned success. Independent postfit review rescored all 12,480 saved predictions, verified exact schedules/receipts/checkpoints and replayed960 predictions across both arms, all seeds and conditions. No reviewer optimizer steps were taken.

The existing Forge → TrainLab → Veritas → Research Lab interfaces supplied data release, execution receipts, semantic checks and evidence links. The trained models, not new infrastructure, are this experiment's output.

## Next discriminating research

First freeze these checkpoints and run a small SET→QUERY diagnostic over all 32 key/value combinations, followed by a separate overwrite panel. Some payload classes are particularly weak in this development result: pointer-slot seed 202 answers v2 correctly 0/43 and v3 correctly 1/66; pointer-slot seed 303 answers v6 correctly 0/59. CSM seed 101 also struggles with v6 (18/59). These are correlated query counts, not independent tasks, and the diagnostic must check routes/occupancy effects before attributing failure solely to value coding.

If basic value competence is weak, compare a prospectively frozen competence-first stage against joint training with identical total examples, supervision and optimization budgets. If basic competence is strong, probe frozen stored records to distinguish representation/transport failures from query/readout failures. A failed linear probe alone would not prove information loss. This tests the proposed training-order idea without assuming it works or adding another architecture.
