# Parallel executor boundary

This is a reconstructed CSM-PARALLEL-022 engineering candidate, not a reproduction of unavailable 001. The pinned `reference019.py` SHA is retained in EXECUTOR_PROVENANCE.json. Exact supplied operations are not learned operation selection.

A state maps identities to Absent, Value(v), or Ref(j), with immutable uint32 payloads, including zero. Read follows references until value/absence; cycles return None. At most N descriptor visits suffice: any longer path on N identities repeats. COPY reads the effective pre-operation source once and stores an independent Value or Absent. LINK preserves identity. SET detaches an outgoing reference; incoming live references remain live. DELETE marks absent; later recreation can again be observed through retained live references. IFSET reads its source in the pre-operation state and performs SET only if its present value equals the expected payload. Queries do not mutate records.

## Parallelism argument

Partition identities by a supplied, input-derived bank function b. Admission scans every initial reference edge and every source-dependent command (COPY, LINK, IFSET). If any edge crosses banks, the entire program uses the serial implementation. Otherwise every initial descriptor and every possible operation dependency remains within a bank. By induction, execution cannot introduce a cross-bank reference: the only instruction creating one is LINK and its endpoints were checked.

Each operation in bank i reads and writes only state in i, and writes its answer to its own distinct original program index. Two operations in distinct banks therefore commute on the product state and disjoint answer slots. Preserve original order within each bank, including all conflicting writes, queries and conditional decisions. Arbitrarily interleaving different banks gives the same final descriptors and answer sequence as the original serial execution. OpenMP executes these independent ordered bank streams concurrently. Same-bank conditions/conflicts are barriers to reordering; cross-bank dependencies cause conservative whole-program fallback, not guessed resolution.

This is an informal mathematical engineering argument, not a new machine-checked theorem. It verifies supplied typed execution only, not learned parsing, arbitrary language, or whole-system correctness. C++ and Python differential tests, pinned019 parity and mutation tests provide bounded implementation evidence.

## Cost boundary

`full_call_seconds` includes Python validation, operation/state conversion, bank-ID compression, C++ whole-input validation, dependency scan, stable bank partition, actual execution, output conversion and state copying. It excludes one-time compilation/library loading, which must be measured separately as cold startup. Input parsing and role binding occur outside this executor and must be added to end-to-end workload timing by the runner. Both serial and parallel calls receive the same validation/semantics. `actual_team_threads` records actual OpenMP team size (0 for serial path); this is not proof of speedup. Cross-bank fallback and low work per bank may make this slower. No performance promotion before measured full-pipeline parity and costs.
