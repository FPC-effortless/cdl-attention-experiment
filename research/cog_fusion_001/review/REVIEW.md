# Independent implementation and experimental-integrity review

## Verdict before final learned evaluation

The reviewed code implements a genuine path from learned event interpretation through a protected typed archive and selected evidence to the original character-output head. Primary strict/hybrid evaluation does not substitute gold operations, routes, supports or answers. The separately named supplied-reference-evidence mode is a diagnostic bypass. Final learned qualification remains determined by the locked six-fit evaluation, not by this engineering review.

The independent corpus audit passes, the evaluator executes on a discarded five-update calibration model, and the independent full-size runtime tests pass. This reviewer has not fitted a model or run TEST, FRESH or LONG model inference. The calibration smoke is not scientific performance evidence.

## Reproduced or source-confirmed defects repaired during the build

| Finding | Repair | Review verification |
|---|---|---|
| `missing` pairs originally contained two identical text histories; distinct splits also shared some canonical semantic histories | Missing worlds now vary an unrelated fact; generator rejects identical pairs and regenerates cross-split semantic duplicates | Independent replay of the final corpus finds zero failures, text duplicates or semantic overlaps across splits |
| Fresh template shift was described too narrowly as syntax-only | Protocol and manifest now identify a combined lexical/syntactic wording shift with fixed entity names | Fresh-only task vocabulary enumerated in `DATA_REVIEW.json` |
| Finished rows continued consuming hidden characters while other batch rows generated | Completed rows are removed from subsequent decoder calls | Independent rerun of engine tests passes; regression includes unequal answer lengths and actual model continuation parity |
| Retrying a full pending-mark/commit workflow resurrected a successfully cleared pending ID | Permanent pending-ID completion history plus early request-receipt lookup | Reproduction changed from a stuck pending fact to idempotent completed replay; independent repair tests pass |
| DEV-only evaluator CLI would still run TEST interventions | Interventions run only if TEST is explicitly selected | Current `run` branch inspected; review smoke calls DEV-only `evaluate_panel` directly |
| Fixed-state interventions used only the first world's stale neural state | Both world origins are constructed, with both evidence variants at each origin | Four generated answers per family are now required for a complete bidirectional intervention success |
| Publication used a mutable session revision and checked ordinary model mutation only before inference | Non-reentrant single-flight request guard, immutable starting revision, and a second model-contract check before publication | Source reviewed; runtime owner's regressions cover reentrancy and in-forward mutation |

## Independent checks actually executed

### Corpus audit

`review_data.py` imports no candidate executor or model module. It independently reconstructs accepted ledgers, query supports, evidence values and answer strings, and checks manifest hashes and partition overlap.

- 1,376 families, 2,752 worlds, 17,856 event positions.
- Every post-event ledger and every final answer/support/evidence label reconciles.
- No identical text-world pairs.
- No cross-split text or canonical semantic-history duplicates.
- Familiar TEST and held-wording evaluation each have 80 answer-changing and 48 invariant/unknown families, rather than 128 answer-changing pairs.
- Corpus SHA256: `3b4133d81db7b13b6779b478373b405a997d6a0f6d7c94c5cad530d7fd308c98`.

Receipt: `DATA_REVIEW.json`.

### Evaluator smoke

The reviewer called `evaluate_panel` directly on the first eight DEV families using only the discarded five-update calibration checkpoint. Strict, acute memory-off, hybrid and supplied-reference-evidence execution completed. Independent counting reconciled exact answers, complete pairs, strict whole-history pairs and the five-changing/three-invariant family denominator. No held TEST/FRESH/LONG model call was made.

Receipt: `EVALUATOR_SMOKE.json`. These results verify execution/counting only and must not be mixed into final candidate results.

### Repair regressions

Two pending/early-replay repair tests and all five engine regression tests were independently rerun: **7 passed**. These include unequal-length batched generation, real native continuation parity, true first-result-dependent second-hop routing in the bounded fixture, and explicit tie/missing results.

### Independent full-size runtime tests

The separate reviewer-owned `tests/test_review.py` uses the actual 1,077,834-parameter discarded calibration model, without replacing its predictions or generator:

1. Instrumented real embedding calls prove exact accounting of input characters, answer-control prefix and generated characters. Replaying a request and restoring a session feed no old characters.
2. A real subprocess terminated with `os._exit` immediately before COMMIT. Reopening and retrying matches uninterrupted sampled text, all recurrent tensors, RNG, counters and a subsequent new answer.
3. The same check terminates immediately after COMMIT. Retry returns the historical receipt once and preserves the latest coherent process continuation.
4. Global pending policy survives reopening, emits no neural answer, consumes no sampling RNG, and rejects a request-ID collision without changing counters.

**4 passed in 12.85 seconds.** Receipt: `RUNTIME_REVIEW.xml`. These demonstrate runtime properties with real full-size neural execution, not learned linguistic correctness.

## Training and fairness observations

The primary trainer uses supervised typed writes/routes during the declared 250-step warmup, then predicted discrete execution for 750 steps. Supervised character teacher forcing is correctly shifted. Both arms receive the same semantic examples, initialization, auxiliary supervision and update count. The full five-event history, query and answer share recurrent state and a backpropagation graph; no reset-event feature cache is used.

The same pretrained native backbone is reused across the three seeds. These are new-interface/data-order replications, not three independently pretrained language models. Nominal stored size is 1,077,834 parameters; active size is 997,674 because 80,160 legacy-board parameters remain stored but disabled. Equal nominal parameters do not imply equal effective capacity or inference cost. The 64-extra-filler panel is an extrapolation diagnostic beyond the five-event training regime.

Hybrid training compares all 32 finite logical identities as supervision. Actual hybrid inference builds an index from the current ledger, classifies over 16 bucket addresses, and scores at most four fetched candidates per hop. The full per-query index-build work remains present and counted. A bounded scored-candidate budget does not make that full query independent of archive size.

## Scope limits that should remain visible

- Fixed entity vocabulary and synthetic event grammar limit the semantic result. Generic SQLite capacity tests with many identities do not establish neural generalization to those identities.
- The strict second hop follows a typed executor rule. The hybrid's second query consumes the actual first selected value. These are different learned-versus-deterministic boundaries.
- The new decoder controller receives retrieved evidence and affects character output. Its state is not directly consumed by the event writer or route-query network, and the legacy read-compute-write board is disabled. No learned averaging or derived durable-write ability has been transferred or tested in this study. Generated characters can indirectly affect later native state, but that does not establish the original continuous-vector computation capability.
- Fixed-state invariant supplied-evidence cases reuse identical evidence packets. They test packet stability; actual irrelevant-edit robustness comes from evaluating the two distinct predicted histories.
- Pending abstention is a deterministic runtime policy on known affected scopes. Correct scope interpretation remains a learned prerequisite. It is not evidence that the language model learned to abstain.
- The durable Session bridge currently uses strict typed resolution. Hybrid retrieval is a separately evaluated in-memory path; it should not be described as an equally qualified durable hybrid session without additional integration.
- The model guard uses parameter identity/version counters plus configuration/training-mode checks before and after inference. Deliberate `.data` mutation and arbitrary host code changes are explicitly outside its contract. The reviewer reproduced the `.data` bypass on a disposable model; no tamper-proof runtime claim is supported.
- Durable source identities and receipts are provenance metadata, not proof that an accepted model proposal is linguistically true.
- Fixed-size neural state and avoidance of historical prompt re-encoding are distinct from bounded total storage. Records, receipts, pending history and execution metadata grow with retained information.

No general attention-replacement, language-quality superiority, or efficiency conclusion follows from these checks. Final learned outcomes and full serving-cost measurements must be reported separately, including failures.

## 23 November 2026 — optional v2 addendum and final evidence review

The review above records the primary **v1** implementation and its scope before final learned evaluation. Its statement that the decoder controller does not directly feed the writer or query network remains correct for the locked v1 campaign. A subsequent, explicitly separate **v2 attachment** now implements those direct connections. It does not change the v1 model, trainer, data, decoder or six fitted endpoints.

### Added architecture and structural qualification

`fusion/full_model.py` composes the existing FusionModel with an eight-value, 32-coordinate RecordComputer. Its persistent state contains both the full base continuation and the new record controller. Actual selected numeric payloads update that controller. Nonzero, bias-free projections connect the controller directly to semantic-write features, the subsequent retrieval key and the original character-output head; a learned readout proposes an eight-dimensional numeric value. The typed runtime supplies source and destination keys and validates publication.

Zero controller state exactly preserves v1 event, routing, language and decode outputs. A nonzero controller is an actual additional input to those paths. Clearing it preserves the base state while removing the record controller's contribution. The numeric-training API freezes the base and the feedback projections, exposing only the 4,296 RecordComputer parameters to the separate numerical fit.

| Full-size optional wrapper quantity | Value |
|---|---:|
| Stored parameters | 1,089,106 |
| Active parameters | 1,008,946 |
| Added parameters over v1 | 11,272 |
| RecordComputer parameters | 4,296 |
| Additional feedback-projection parameters | 6,976 |
| Retained disabled legacy-board parameters | 80,160 |
| Neural continuation tensor bytes | 10,720 |
| Additional record-controller tensor bytes | 128 |

The eight reviewer-owned `tests/test_full_model.py` tests passed in **2.26 seconds**, recorded in `FULL_MODEL_REVIEW.xml`. They check exact zero-controller preservation; causal first-read interventions and nonzero gradients into numeric write, semantic write, subsequent retrieval and original character logits; controller clearing; numeric-only freezing; checkpoint/state round-trip with a nonzero base continuation; actual RecordRuntime publication and restart with the full wrapper; source invalidation; shape and explicit-state errors; and unchanged v1 training-lock hashes. These are structural and engineering qualifications. The numerical experiment did not train the feedback projections or establish the quality of their nonzero effect on language or selection.

### Separately fitted numerical component

The optional component protocol was reviewed and locked before any numeric fit. Three seeds, 1701–1703, each used 1,000 updates and 64,000 training pairs per arm. Within a seed, the connected and separately trained reset-control arms began with identical weights and saw identical examples. The task supplies two independent eight-dimensional vectors from Uniform[-0.75, 0.75] and asks for their mean. All seeds use the same 4,096 fresh evaluation pairs; they are replications on one test set.

| Seed | Connected MSE | Trained reset-control MSE | Acute reset MSE |
|---|---:|---:|---:|
| 1701 | 0.0000478025104 | 0.0469656251 | 0.0477657244 |
| 1702 | 0.0000526335207 | 0.0470454507 | 0.0482849106 |
| 1703 | 0.0000536570406 | 0.0470583439 | 0.0482675061 |

The reviewer independently recomputed every saved-output MSE, checked the exact target arithmetic and shared input hash, and verified both checkpoint hashes for each seed. All recorded numeric gates pass. The trained reset controls approach the theoretical second-only predictor's expected MSE of 0.046875; permutation and acute reset remove the low-error result. The recorded first-input gradients are finite and nonzero. These findings establish useful first-read retention and approximate averaging on this bounded task, without an exact-arithmetic or out-of-distribution guarantee. The six component fits and their recorded evaluation/checkpoint work consumed 10.3976 measured CPU seconds in total.

### Durable numeric execution

The runtime owner's separately specified durable check takes the first 16 held-out pairs per seed, with no outcome-based selection. Actual source records enter the fitted controller and its actual neural proposal is persisted. The reviewer reconciled all **48** recorded input/target/output rows against the saved held-out tensors. The largest discrepancy from batched neural output is **1.1920928955078125e-7**, below the declared 1e-6 tolerance.

The owner receipts report 24 restorations between the two reads, 48 restorations after publication, and 48 controller clears followed by valid derived-record reuse. They also report six source-erasure rejections, three before-commit rollbacks and preservation of unrelated records. The reviewer inspected the verification source and reconciled these receipts, but did not rerun the durable fitted-model campaign. This distinction is separate from the reviewer's independently executed full-wrapper runtime regression above.

Derived-record reuse is qualified through `read_derived`, which validates both source packets and the model/implementation identity. Reading the raw stored envelope alone provides no such current-validity guarantee. Clearing transient controller state and erasing authoritative provenance are different operations: source erasure must invalidate dependent reuse. The source/destination selection is supplied through a typed interface; natural-language input does not automatically invoke numeric computation. A new complete computation resets the record controller, while staged `begin_read`/`finish_read` explicitly persists a partial computation.

### Final primary evidence reconciliation and verdict

`review_final_evidence.py` imports no candidate model, executor or evaluator. It makes no model call and performs no fitting. It independently replays recorded proposed operations, reconstructs every intermediate ledger, derives strict reads from actual final ledgers, checks generated strings and support keys, and recomputes complete-pair, complete-history and per-kind counts across all **3,456** saved primary evaluation rows. It also verifies both state origins in the fixed-state intervention records, reconciles weighted language NLL and live-question counts, and checks all primary and optional fit-lock hashes. The receipt is `FINAL_EVIDENCE_REVIEW.json`.

The main report's familiar, held-wording, long-history, hybrid and live-interaction values agree with these records. Familiar full-history success is 128/128 in every memory fit. Held-wording full-history success is only 15/128, 28/128 and 17/128; live learned questions are 4/7, 5/7 and 3/7. The ordinary-language comparison correctly distinguishes improvement over the admitted local-only parent from degradation relative to the original full COG. The original complete learned-system gate remains failed.

**Independent final verdict:** the familiar-distribution factual-memory connection is supported, and the separately supervised numerical controller plus typed durable publication has passed its bounded component tests. The complete proposed architecture is implemented as an optional composition, with actual direct controller paths, but jointly learned language, source selection, numeric invocation and feedback are not qualified. The new component cannot retroactively upgrade v1's failed wording or multi-turn results. No attention-replacement, general-language, arbitrary-scale or efficiency-superiority claim is supported by this cycle.
