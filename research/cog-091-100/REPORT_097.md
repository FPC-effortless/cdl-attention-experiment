# COG-ATTN-097a — Indexed episodic retrieval plus trainable perception

**Executed:** 2026-11-21 · **Status:** bounded synthetic subsystem study; not end-to-end COG-ATTN qualification.

## Question

Can the previously measured non-attention parallel specialist and multi-label hypothesis model feed a provenance-retaining, append-only raw episode store and retrieve the intended record without an all-to-all history scan? Does it repay its index costs against strong typed and pretokenized/raw search controls?

## Implementation and lineage

- `experiment_097.py` imports the exact `Router`, `make_data`, `vectorize` and `HypothesisNet` definitions in the locally preserved 091–100 study. It retrains *each seed* on the 091–100 training splits and selects checkpoints only on the dev splits. The same held phrases stay out of training, and the ordered operation pairs are disjoint.
- Six seeds: 9107, 9203, 9301, 9409, 9503, 9601. Histories: 1,536, 6,144 and 24,576 raw records, six known operations per entity. At each history size, 60 held requests/seed (48 determinate + 12 explicitly ambiguous). Same held language decisions are **reused at each scale**; they are not 1,080 independent natural-language cases.
- Unambiguous query includes the entity as an external **exact structured input**, not inferred by the model. The raw episodes record one of six known operations, exact object ID, payload and epoch. Queries are artificial utterances from the known templating grammar, including unseen inflections of known stems and negation/revision templates.
- Candidate generation uses model probabilities to rank operation IDs; index lookup visits the six raw records under the externally supplied entity. `top1` is the actual answer; `top2,4,6` measure **candidate coverage only**, not successful downstream disambiguation. A separate 6-dimensional sigmoid head retains two alternatives on ambiguous requests; the decision head defers.
- Source-epoch guards reject a missing or stale entity. A new epoch-2 append is tested while old raw events remain stored. No signed external action is triggered.
- A byte-identical vendored **COG-ATTN-071** `boundary.py`, GitHub blob `f4cccf5ac2f7758fd9cb84516a4d39046b5e56ca`, is exercised independently by `test_097_boundary.py`. This does **not** claim that the synthesized six-action model is correctly grounded in the original eight-operation issue workflow.

## Accuracy (one history size; same language decisions at all three sizes)

| Model | Correct language actions / all | Correct indexed reads / determinate | Correct defer / ambiguous | Correct top-2 candidate coverage / determinate | Ambiguous pair captured |
|---|---:|---:|---:|---:|---:|
| Parallel temporal specialists | 351/360 | 279/288 | 72/72 | 286/288 | not trained |
| 2-layer tiny self-attention | 324/360 | 252/288 | 72/72 | 284/288 | not trained |
| Parallel specialists + separate multi-label hypothesis head | **359/360** | **287/288** | **72/72** | **288/288** | **63/72** |

The top-2 numbers are **oracle set membership**, not a 288/288 actual answer score: no learned or verified second-stage resolver has been tested. The multi-label head learned from explicitly provided two-hot training labels. None of the model outputs has write authority.

On the 288 determinate requests, the first-matching familiar action-word control identifies the intended operation 72/288 times. The single last-slot-per-entity control contains the right operation on 53/288 (n=1,536 and 6,144) and 35/288 (n=24,576). It drops other actions by design. The direct typed **(entity, action)** index is a much stronger control: it retrieves exactly the same payload as the raw episode index when the action decision is correct, and it is faster. This demonstrates that an episodic index is not needed merely to implement known-schema routing.

## Median local single-thread CPU microbenchmarks

These are medians over six seeds, not hardware-normalized confidence bounds. Predictions are timed separately with per-query character encoding and model forward pass. The scan is a full sequential Python loop, not optimized vectorized retrieval or a Transformer KV cache. Index lookup does six-row candidate comparison per predicted action. End-to-end agent authorization, external observations and clarification latency are not included.

| History | Cold index build | Raw full scan/query | Episodic top-1 lookup/query | Typed (entity,action) lookup/query | Multi-head encoder+lookup/query |
|---:|---:|---:|---:|---:|---:|
| 1,536 | 3.81 ms | 32.9 µs | 1.22 µs | 0.38 µs | 480 µs |
| 6,144 | 18.29 ms | 135.7 µs | 1.49 µs | 0.55 µs | 429 µs |
| 24,576 | 88.72 ms | 1,090.4 µs | 4.03 µs | 2.40 µs | 452 µs |

Approximate cold-index break-even versus this unoptimized raw scan: roughly **100–140** queries at 1,536/6,144 records and **82** queries at 24,576 records, ignoring write maintenance and model training. Only 60 queries per seed were scored at each scale, so the build cost was **not** fully amortized in the measured evaluation. A direct typed dictionary is cheaper for these known-schema requests. A single incremental append took approximately 10–15 µs in the measured fixture; this is a microbenchmark, not a write-heavy service profile.

An important correction to an earlier 091 summary: the fairer pretokenized scan baseline is 39.61, 195.64 and 1,145.69 µs/query at 1,536, 6,144 and 24,576 records, respectively. Do not compare the old expensive on-the-fly retokenizing scan (488 and 11,775 µs from an earlier pilot) with the index as if it were the optimized baseline. The 097 scan is differently structured and its timings should not be merged with 091 timings.

## COG signed boundary regression

All ten new tests passed (five index/epoch tests, five tests against byte-identical COG `boundary.py`). They show:

1. A semantic proposal with no external signed token cannot update the local ledger.
2. A signature for another target/nonce cannot be replayed, and duplicate delivery is idempotent.
3. Changing policy epoch makes a previously signed proposal stale.
4. An independently authorized but **semantically mistaken** operation can still commit. The authorization gate does not establish user intent.
5. Provisional ambiguous alternatives can be discarded without creating an execution token.

The fourth point is the crucial open risk. The synthetic 097 retrieval wins do **not** close the semantic commit error rate from 071–080.

## Decisions and next experiment

**Keep:** lossless source-preserving raw archive, bounded candidate access, trainable perception, explicit multi-hypothesis defer, epoch guards, unchanged original COG ledger.

**Do not promote yet:** 097 to general semantic retrieval, context reinterpretation, self-attention equivalence, end-to-end latency reduction, or improved live issue workflow. The known six-action schema, exact entity anchor, direct labels and template language make this a proof-of-connection, not a hard generalization test.

**COG-ATTN-097b admission tests:**

- Train on raw past observations and verified downstream outcomes, not supplied correct operation labels alone; compare frozen vs joint gradients and a true sparse differentiable candidate scorer.
- Hold out **entities, full paraphrase families, corrections and concept aliases**, not just ordered action pairs. Require original MultiWOZ held dialogues or appropriately licensed authentic issue transcripts, as available; keep verification sources and authority disjoint.
- Give the model only raw language (no external exact entity key), preserve earlier conflicting mentions, and ask long-range elliptical questions. Test 128→32,768 turns; include deliberate index misses and high-frequency anchors. Require an explicit fallback budget and account for its use.
- Compare direct typed records, query-dependent raw scan, BM25-style lexical retrieval, sparse learned retrieval, and a pretrained or properly budget-matched Transformer on the same decisions. Match accuracy/coverage before comparing time; include ingestion, cold index build, writes, re-index, verification, rollback, and source-epoch repair.
- Treat retrieved top-k as *candidate coverage* unless a separate, measured resolver converts it into correct final decisions. Report unsupported/wrong authorized actions independently from unauthorized writes.
- Freeze protocol and training choices **before** exposing a reserved authenticity set; publish independent audit and exact source snapshots.

## Reproduce

Python 3.10+, PyTorch 2.x and pytest. From the directory containing the study:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m pytest -q test_097.py test_097_boundary.py
python experiment_097.py --out 097-results.json
sha256sum --check SHA256SUMS
```

`097-results.json` records per-seed training costs, every history scale, model decision counts, candidate probes, build costs, scan controls and request hashes. Exact timing values are host-dependent; the categorical decisions are reproducible with matching torch and CPU settings. A separate researcher should audit all frozen source files and rerun fresh seeds before promotion.
