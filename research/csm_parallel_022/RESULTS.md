# CSM-PARALLEL-022: learned parsing plus exact parallel memory

17 November 2026. Six completed fixed-budget CPU fits, three seeds, two learned parsers. This is a separately authored experiment, not a reproduction of the unavailable CSM-PARALLEL-001 evidence package.

## Decision

Keep the exact typed memory executor and instruction batching. Do not promote the Transformer parser or claim a general parallel CSM repair. Both parsers learn familiar instruction semantics and preserve COPY/LINK/conditional behavior in that setting, but neither meets the complete qualification gate on unfamiliar wording, identities, cross-bank instructions or independent multi-region composition. The immediate bottleneck is robust learned instruction/role selection, not evidence that the typed memory rules need more capacity.

A simpler GRU parser with batched instruction processing is the stronger practical reference here. Its familiar-task accuracy is higher and its measured complete offline sessions are faster than the Transformer parser. This is not a universal architecture ranking or a compute-matched optimization study.

## What was actually tested

Each parser receives identical instruction bytes, operation/span supervision, sampled example order, optimizer and 2,048 updates at batch 32: 65,536 sampled instruction exposures per fit. The TRAIN roster contains 32,768 instructions, approximately two expected reuses with replacement, not proof of convergence. The Transformer has 67,919 parameters; the GRU 68,927. Both can use the complete framed instruction, including later text, for role selection. Neither receives gold roles or an ideal teacher memory state at learned inference.

After learned operation and role selection, a shared engineered executor performs exact SET, snapshot COPY, live LINK, DELETE, QUERY and conditional SET. It preserves per-bank order and uses real OpenMP bank concurrency only when all dependencies are bank-local. Cross-bank dependencies trigger conservative whole-program serial fallback. The controller is learned; storage laws, literal conversion, rendering and conditional comparison are engineered.

This is **offline ingestion of an available instruction session**, not an online-arrival or arbitrary native-CSM token-parallel experiment. Input event boundaries are supplied; payloads are fixed three-byte ASCII values. Conditional sources are always-present snapshots in this task family. The experiment does not establish arbitrary natural-language, arbitrary conditional-source or unbounded-memory competence.

## Complete-session accuracy

Each row contains 384 session-evaluations: the same 128 sessions / 64 pairs evaluated under three seeds, not 384 independent subjects. Independent-region sessions contain 24 queries; other sessions contain six. All required answers must be correct for a session to count.

| Development axis | Transformer | GRU | Both meet 95% session and paired-first gate in every seed? |
|---|---:|---:|---|
| Familiar wording and structure |382/384|384/384|Yes|
| Held-out composition pattern |379/384|384/384|Yes|
| Longer distractor delay |382/384|384/384|Yes|
| Unfamiliar identities |37/384|24/384|No|
| Untrained wording |0/384|0/384|No|
| Cross-bank instructions |0/384|0/384|No|
| Four independent memory regions |0/384|0/384|No|

Neither model passes the full gate. Partial answers and aggregate accuracy do not rescue a failed axis. These are exposed constructed development results, not protected final-test or broad language generalization evidence. A multi-region pair is counted as one independent paired session, not four independent regions.

Training losses fell across all four training quarters; the per-seed trends are retained in AGGREGATE.json and raw update traces. A decreasing training loss does not remove the observed transfer failures, and a fixed-budget failure does not establish that an architecture could never learn the task.

## Complete offline-session costs

The fixed timing subset uses four sessions per axis, three cyclic schedule orders and three seeds. Below are medians of nine per-four-session mean observations, in milliseconds/session. Same-checkpoint outputs were checked for exact parity between schedules. Timers include parsing, span interpretation/registry, dependency planning, native executor and payload rendering; model load and offline scoring are excluded.

| Workload/parser | One instruction per parser call, serial executor | Chunk 32, serial executor | Chunk 32, parallel-enabled executor |
|---|---:|---:|---:|
| Familiar/Transformer |32.873|27.361|28.015|
| Familiar/GRU |74.080|13.809|13.324|
| Independent regions/Transformer |138.937|157.747|157.173|
| Independent regions/GRU |344.160|54.374|53.379|

Most of the practical gain comes from batching instruction parsing, not demonstrated large executor-thread speedup. Parallel-enabled execution is not necessarily multi-threaded: single-bank cases stay serial, and cross-bank hazards fall back. The learned multi-region cases fail quality, so their timing cannot support a useful quality-matched speed claim. Larger chunks are not uniformly faster; all schedules are retained rather than selecting a winning chunk after the fact.

The supplied-gold-operation executor control used actual four-thread teams for all 384 independent-region sessions and preserved serial state/answers exactly. Its median complete supplied-program session time was 0.358 ms serial versus 0.303 ms parallel-enabled. These control runs were fixed-order single measurements per session, not randomized repeated microbenchmarks, so the difference is descriptive and does not establish a reliable 1.18× speedup. Gold-operation control correctness is engineered, not a learned-parser result.

Successful scientific native stages total 2,189.637 seconds (36.49 minutes), including training, evaluation, replay, timing and bookkeeping. Parser training totals 566.366 seconds Transformer and 521.796 seconds GRU. Earlier software calibration/preparation costs are separate. No paid compute, GPU or protected TEST was used. Peak memory, state-byte accounting and isolated model startup were not measured; no memory-efficiency claim follows.

## Verification and mathematical boundary

The implementation maintains explicit absent/value/reference states. Snapshot COPY reads the pre-operation effective value; live LINK preserves identity dependence; SET detaches an outgoing alias; cycles/dangling references resolve to absence. Bank isolation provides a sufficient commuting-operation argument while order remains fixed within each bank. Cross-bank/transitive-link, cycle, self-copy and conditional hazards have dedicated tests and serial fallback. This is a conditional mathematical correctness argument plus differential tests, not a Lean proof or a proof of learned role selection.

Before fitting, independent review fixed two substantive issues: an additive shared role-context term that canceled in relative span scores, and a scorer that trusted derived receipts rather than reconstructing them. The final multiplicative role head and fail-closed raw-action reconstruction were tested and frozen before scientific execution. A reference-path portability issue was also repaired;46 tests passed in relocated scientific bundles. Historical calibration/source and failed reviews remain retained, not rewritten as if those issues never occurred.

Independent auditing checks archive integrity, pinned source/runtime/data, actual checkpoint/update/parameter counts, matched sampled IDs/bytes, raw action rescoring and serial/fresh-checkpoint replay. It is not independent retraining or a new independently executed checkpoint-inference reproduction. Detailed final audit receipts accompany the evidence.

## Next engineering choice

The next repair should target reusable identity and role binding with enough linguistic and multi-region training coverage, keeping the exact executor and batched GRU as controls. Do not add memory complexity or increase backbone size solely because unfamiliar instruction parsing failed. A later study must prospectively separate learned boundary detection, role generalization and genuinely online commit behavior, then demonstrate acceptable quality before claiming inference superiority.

Reviewed source and resumable instructions are published in draft PR 69. Private checkpoints, constructed data, native receipts and raw predictions are retained separately in the owner's Library and will be indexed in ChatGPT Space after final package verification. Another agent needs access to those private files as well as the public repository to resume without retraining; public GitHub alone does not contain them.

## Post-audit interpretation clarification

See [compound development shifts addendum](PANEL_INTERPRETATION_ADDENDUM.md). The cross-bank and independent-region panels change identifier formats and other structural factors simultaneously. Their failures do not isolate bank interactions, interleaving, capacity or unfamiliar identifier formats. Original scores, gates and evidence remain unchanged.
