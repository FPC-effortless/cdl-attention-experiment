# CSM-PARALLEL-022: learned parsing with exact, dependency-aware memory execution

Status: prospective draft, no scientific fit admitted. This is a new study, not a reproduction of the unavailable CSM-PARALLEL-001 archive. The supplied report alone does not verify its 2,709 checks or timing claims.

## Practical target
Read explicitly framed text instructions, learn operation and argument spans, execute SET/snapshot COPY/live LINK/DELETE/QUERY/conditional SET, and return exact retained payloads. Compare one learned checkpoint under serial versus chunked controller scheduling and safe executor scheduling. Preserve multihop, deletion/recreation, cycles and cross-region dependencies rather than deleting hard cases for speed.

This is a structured-memory architecture change with learned text interpretation, not an equivalent implementation of native state-conditioned CSM, a language foundation model, or unrestricted token-parallel recurrence. The controller sees each complete instruction, not future instructions. Instruction framing is supplied. The IF predicate is executed against actual current memory after learned argument selection; learned parsing is not a learned state-dependent guard.

## Arms and attribution
1. Small byte Transformer instruction controller, learned operation plus four byte-span roles.
2. Byte GRU instruction controller with the same text/role supervision.
3. Supplied correct operation program through the same reference executor, a labeled engineered correctness/cost control, not a trained model.
For each learned checkpoint, compare chunk sizes 1, 8, 32, 128 with serial execution, then the conservative dependency-aware executor. Identical predicted actions/answers/state are a prerequisite for scheduling cost comparisons. Chunk-parallel parsing alone is distinguished from parallel memory commits. No claimed MemNet comparison unless a separately specified, trained memory-network control is actually admitted and run.

Both learned models are trained on identical sampled instruction IDs with exact counts and fixed seed roster 11/23/37. Parameter counts and actual train cost are reported; equal exposure does not imply equal compute. No outcome-selected seeds or stopping. Final training budget will be set after bounded software calibration and independent review, before inspecting capability results.

## Data and qualification
Prospective default: 1,024 TRAIN pairs / 2,048 sessions; 64 fresh pairs / 128 sessions on each of seven DEV axes: familiar, wording, names, composition, delayed dependencies, cross-bank interaction, independent regions. Six queries/session for six axes; the independent-region axis interleaves four dependent programs across sixteen records and has twenty-four queries/session. It remains sixty-four independent paired sessions, not four times that number. Counterfactual histories differ in early facts; paired observations are not independent sessions. All pair IDs globally unique with exactly variants 0 and 1. Exact rosters, split hashes, role/payload coverage and generator version freeze before fits. No protected TEST.

Payloads are copied exactly from learned spans into a reversible natural-number representation; no neural raw-byte reconstruction is needed. Gold roles/actions never enter inference. Invalid spans or operations are explicit failures, never silently repaired from labels. State is fresh per session, then persistent through every instruction/query. Conditional outcomes use live generated state. Session scores require every answer, not merely correct absence outputs.

Report per-seed/axis: operation and role exactness, exact nonabsence payload answers, absence answers, complete sessions, paired first-answer correctness, conditional decision correctness, and invalid-action counts. Qualification requires >=95% complete sessions and paired correctness in every declared seed/axis, no scheduling-induced regressions. No pooled aggregate rescue.

## Semantics and concurrency contract
Pinned 019 reference: absent/value/ref records; SET detaches outgoing reference; COPY resolves source before mutation and snapshots; LINK stores live identity; DELETE marks absent; read-only QUERY; cycles and dangling chains resolve absent. Capacity exhaustion rejects, never silently aliases identities. Payload encode/decode is bijective over the declared bytes.

For fixed-route edits A,B, sufficient snapshot-parallel safety is disjoint writes and no write/read intersection in either direction. Read footprints must include transitive live references. Different destinations alone are insufficient. Conservative executor may force LINK/QUERY/IF and COPY with live references to barriers. Such barriers are measured, not removed from the benchmark. Same-checkpoint controller batching preserves functions because instruction features have no shared state; floating-point/argmax parity is checked rather than assumed.

## Costs and recovery
Measure tokenizer/span validation, neural controller, identity registration, dependency planning, executor, query rendering, total session time, startup, parameter/state/peak memory, and training separately. CPU threads fixed and reported; actual concurrent execution distinguished from vectorization/cache reuse. Batch-1 session latency and chunk throughput are distinct. Include every method's equal reuse opportunity and rejected-program handling.

Free resources only. Last verified T4 balance 9,222.760s, no reservation, paid scaling false. Refresh before admission; preserve 25% remaining quota recovery reserve. Use CPU when practical, bounded calibration before fits. Public source backup and private exact-input backup precede admitted training. Native Forge/TrainLab/Veritas/ResearchLab evidence, checkpoints/RNG/optimizer/source/runtime bindings and independent saved-result scoring/replay are required. Failures remain retained. Do not launch from this draft.

### Availability and latency qualification
Chunk batching assumes the framed instruction block is already available. It does not allow reading future interactive user instructions, nor claim that a live user can receive a response before its dependencies arrive. The measured session is an available-input prefill workload with ordered query answers; one-instruction online latency is reported separately if actually measured. Parsing future instructions in a supplied block cannot alter earlier actions because the parser has no cross-instruction features, and exact executor ordering preserves earlier queries.

## Frozen scientific execution proposal after CPU calibration

CPU-only, one Torch intra-operation thread; OpenMP executor explicitly four threads. Three seeds 11/23/37 and two parser architectures per seed. Each fit uses 2,048 AdamW updates, batch32, learning rate0.0003, weight decay0.01, betas(0.9,0.999), epsilon1e-8, clipping norm1; fixed dropout0. Operation and required-role endpoint CE supervision. There are32,768 TRAIN instruction rows from2,048 sessions. Each fit samples65,536 instruction exposures with replacement (~2 expected roster reuses, not two complete epochs). Total6fits/12,288updates/393,216 instruction exposures. Fixed budget is not a convergence guarantee; report retained loss trend rather than asserting optimization finished.

Each seed runs all7 DEV axes through both models:1,792 primary sessions/15,360 primary queries. Primary chunk32/four-thread executor is compared with same-checkpoint chunk1/serial executor on the entire roster, and replayed from a separately loaded saved checkpoint. Identical raw actions, registry, errors, answers, renderings, conditional decisions and final state are required. Replay is not independent new capability evidence. An independent reviewer rescoring retained raw actions is not an independent training or inference run.

Diagnostic timing uses the first two complete pairs per axis, fixed before training, with schedules(chunk1/8/32/128,serial1thread) and(chunk32,parallel4threads), three cyclic order rotations. Raw timed rows are retained and parity checked. Measurements are correlated repeated executions, not extra generalization examples. Supplied-program exact controls are actually executed across the full roster under serial and parallel schedules, including command construction and exact answer rendering. They receive gold typed-operation supervision at inference and must never be called learned results.

A conservative wall budget of90 minutes for the whole CPU campaign covers training, complete validation/replay, timing, cold compilation and evidence packaging; each seed is bounded to1,800seconds model work, with native wrapper overhead separately recorded. This is an estimate from bounded CPU software calibration, not a runtime guarantee. CPU execution does not spend the remaining T4 quota. If a stage times out, retain partial evidence and review recovery; no silent reduction in model counts, updates or panels.

This implementation parses and registers the whole available session before execution, and dependency planning can inspect later supplied instructions. It is an **offline/batch ingestion** comparison, not an online-arrival commit algorithm or first-query latency measurement. No input after the available session is accessed, and stable in-bank execution preserves ordered answers. IF source is a present snapshot in this task design; arbitrary or absent conditional-source generalization is not tested.

Scientific scoring independently reconstructs typed commands from raw text and predicted spans, replays the Python reference and checks every retained answer, guard, rendering and final state before metrics. This prevents forged/intermediate wrong outputs from being hidden by a correct final state. Gold roles are used only for scoring and the explicitly labeled supplied-program control.

### Exact measured cost scope (clarification before admission)
The earlier broad cost checklist is limited for this CPU campaign: retained metrics are actual parameter counts, training/process wall times, cold executor construction, and complete timed session components. Peak resident memory, allocated tensor/Python-state bytes and isolated model-load/import startup are **not measured**. Native outer wall time includes process overhead, but cannot attribute that overhead to individual models. No memory-efficiency or isolated-startup superiority claim is permitted. Registry size is not total memory usage.
