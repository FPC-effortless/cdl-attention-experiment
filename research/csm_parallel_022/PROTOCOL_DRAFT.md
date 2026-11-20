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
