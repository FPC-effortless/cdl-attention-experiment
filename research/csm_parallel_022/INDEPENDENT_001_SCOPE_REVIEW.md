# Independent review of the supplied CSM-PARALLEL-001 report

17 November 2026. This is a report/source-boundary review, not a reproduction of the unavailable 001 archive.

## Evidence recovered
The owner's pasted report describes 2,709 correctness comparisons and CPU timings, but the message has no Library attachment; its visible attachment is a GitHub reference. Alternate Library title/content searches and the joint research Space index did not locate the archive. This does not establish that it does not exist elsewhere. No original C++ source, workload generators, timing repetitions or correctness logs were inspected.

The cited Git object `c5b53d973192d3f1842185a4ec51d55c7e958b38:research/csm_conditional_019/reference_executor.py` is present in the retained csm-resume-clean repository. Its SHA256 is `bec9d5c1a05f6c50b43ed68fc3e6040064f76036d9d2fbecb024eb2dd81a7059`. An older checkout's untracked copy is not evidence that the pinned path is missing.

## What the reference actually specifies
SET replaces with an immutable natural-number value and detaches a reference. COPY resolves the source before mutating its destination, including self-copy, and creates a snapshot. LINK stores a live identity. DELETE makes the record absent; an incoming link can observe later recreation at the same identity. QUERY is read-only. Cycles and dangling chains return absence. These are supplied software semantics, not learned language interpretation or a learned memory algorithm.

## What the pasted numbers support, conditionally
Reported 15.764 / 4.444 = approximately 3.55 times improvement is a same-partition implementation one-versus-four-thread comparison. Reported 14.697 / 4.444 = approximately 3.31 times is versus the original serial implementation. Both apply to independent linked regions, not arbitrary dependent operations. The reported 17.263 / 0.217 = approximately 79.55 times from rounded values concerns caching, not parallel execution; the reported 79.68 may use unrounded data, which is unavailable. Neither figure is independently verified. Most other shown workloads lose to serial execution. No learned-controller accuracy, full natural-language session cost or GPU advantage follows.

## Necessary correctness and cost boundaries
Different destinations alone do not imply independence. Read sets must include transitive live-reference paths; conditional guards and copies depend on current state. Cache invalidation must cover changed referents, deletion/recreation and cycles. Versioned snapshots must not silently turn live LINK into snapshot COPY. Scheduling, indexing, dependency discovery, invalidation, payload copying and memory overhead belong in end-to-end cost. Dynamic dispatch based on full future traces is an offline advantage unless explicitly exposed equally to controls.

## New 022 draft, separate work
Independently ran its current seven executor tests successfully (including random traces, self-copy, cycles, recreation and cross-bank fallback). Source inspection supports its conservative sufficient condition: all initial references and all COPY/LINK/IF source-destination pairs remain within a bank, preserving bank-local transitive reads, and original instruction order is retained within each bank. Cross-bank programs fall back entirely to serial execution. This is not replication of 001's missing implementation or timing results. The draft's uint24 payload restriction must be explicit, and learned parser batching, registry behavior, compiler/OpenMP/binary pins, malformed-action scoring and full costs still need final review. No scientific fit is admitted by this note.
