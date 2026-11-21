# COG-ATTN-097b — Episodic reinterpretation without supplied entity identifiers

**Date:** 2026-11-21  
**Status:** completed, local CPU, synthetic subsystem diagnostic only. Not an agent, not authentic MultiWOZ and not a demonstrated replacement for self-attention.

## Question and changes from 097a

097a handed its reader the exact entity ID and six known action classes. Here the reader receives only a raw conversation string, extracts the mentioned entity from an append-only source-preserving archive, and ranks seven possible observed episode fields, including one field unseen during query-training. Elliptical inputs mention the entity in the preceding utterance rather than in the final question. Ambiguous two-entity contexts and missing entities must not manufacture an answer. Recorded revisions retain their previous raw episode while current readout selects the latest version.

The learned dual-encoder receives **witnessed query-to-document positive pairs**, not evaluator field IDs as model inputs. Its 1,024-feature hashed character n-gram query/document input projections (32 dimensions each) are optimized with cross entropy against within-entity candidate documents. No Transformer, pretrained language knowledge or external model is used. Training selects an epoch only with 36 development queries; all 90 held-out entity names per seed are disjoint. The dev set contained familiar language, and its 100% correctness did **not** predict held-out paraphrase performance.

Separately, a tiny **support-conditioned witness reader** receives 14 extra verified query/evidence examples (two for each of seven relations) from training entities before held-out scoring. It recognizes supported clauses through engineered string normalization and char-TFIDF matching, then transfers the associated **raw evidence text** to a new entity using a local document similarity search. The 14 examples intentionally include the held paraphrase family and the previously unseen relation, making this **few-shot support-conditioned transfer, not zero-shot transfer**. A wrong-link witness control circularly shifts evidence by one entire relation to preserve support count and phrase distribution while breaking semantic associations.

## Frozen scored split, counted correctly

Six seeds: 1103, 1117, 1129, 2111, 2129, 2137; 8,192 raw records per seed. Each seed contributes 90 familiar/held queries + 23 new-relation queries = **113 answerable queries**, plus **18 ambiguous or missing-reference queries** that should defer. Across six seeds there are **678 answerable** and **108 unsupported** contexts. The 113 queries are *related* templates and source-language patterns across seeds; they should not be treated as 678 independent human utterances. Full text and direct target values are synthetic. Entity names are never passed as structured query inputs.

## Final retrieval accuracy at 8,192 records

| Arm | Correct exact top-1 episode / 678 | Correct % | Median scorer per-query, CPU |
|---|---:|---:|---:|
| Zero-fit word overlap | 552 | 81.42% | 44 µs |
| Word TF-IDF (fit on seven candidates/query) | 581 | 85.69% | 1,432 µs |
| Char TF-IDF (fit on seven candidates/query) | 588 | 86.73% | 2,644 µs |
| Frozen random dual encoder | 83 | 12.24% | 1,280 µs |
| Learned contrastive dual encoder | 431 | 63.57% | 1,332 µs |
| Uncalibrated dual + lexical fusion | 435 | 64.16% | 2,863 µs |
| **14 verified support pairs + lexical fallback** | **671** | **98.97%** | **2,776 µs** |
| Shuffled-support negative control | 353 | 52.06% | 2,808 µs |
| Oracle field+entity selector (not usable NLP) | 678 | 100.00% | 6 µs |

All arms, including the deliberately weak ones, defer in **108/108** contexts with no unique entity match because entity-resolution logic is shared. This is not an evaluation of model-calibrated ambiguity. The results are about *which* existing record is selected from a known bounded candidate pool, not spontaneous goal discovery. Top-7 reaches 100% by construction because all seven valid fields are supplied to the ranking stage; top-k coverage alone is not a correct answer.

### By query type

| Arm | Familiar (360) | Held paraphrase (180) | Unseen field (138) |
|---|---:|---:|---:|
| Word overlap | 353 | 90 | 138 |
| Char TF-IDF | 360 | 90 | 138 |
| Learned dual | 359 | 72 | **0** |
| Witness transfer | 353 | **180** | **138** |
| Shuffled witness | 353 | **0** | **0** |

**Interpretation:** The end-to-end contrastive learner improves over a frozen encoder (431 vs 83), but does not beat robust lexical retrieval (588) and fails completely on the unseen relation. The support-conditioned reader repairs the missed held phrases *only after their exact phrases are represented in verified examples*; it slightly harms familiar queries (353 vs 360 char TF-IDF). Seven wrong cases remain. The control supports a causal role for correctly bound witness examples, not an ability to infer unseen semantics independently. A normal language model's distributional understanding is not replicated.

## Scaling and full-cost qualification

A three-seed check used the **same 339 questions** at each archive size, so the unchanged accuracy is a repeated-measure check, not independent generalization. The archive builder retains all raw revisions and adds distractors; the query's entity has seven live fields. The latest-by-alias map and rarity-aware name postings avoid a full-history candidate loop. Summary medians on the measured host:

| Raw records | Cold index build | Entity resolution | Oracle raw scan (with known entity/field) | Zero-fit overlap scoring | Witness scorer |
|---:|---:|---:|---:|---:|---:|
| 2,048 | 1.40 ms | 116 µs | 85 µs | 46 µs | 2,852 µs |
| 8,192 | 11.75 ms* | 113 µs | 612 µs | 44 µs | 2,776 µs |
| 32,768 | 159.92 ms | 116 µs | 3,273 µs | 44 µs | 2,932 µs |

*8,192 medians are from six seeds; the other two sizes use three seeds. Cold build can fluctuate substantially; one 8,192 seed took 164 ms. Median scorer times exclude entity resolution, cold setup, support acquisition and neural training. The query loop's total wall time includes all compared arms and independent evaluator checks, and is **not** a single-arm latency estimate. One learned-encoder training fit took around 143–170 ms on this host. The support reader's two-example-per-relation acquisition is not free; its construction time was not separately recorded. It is therefore **not scientifically supported** to claim end-to-end speed superiority. In fact, the 14-witness method's per-query scoring is much slower than the cheap overlap control, while having higher accuracy.

## COG authorization, provenance and correctness boundary

- The original COG `boundary.py` is unchanged and preserved by the earlier 097a report and tests. This study invokes **no external execution gateway** and never mints authorizing credentials.
- The seven-field candidate inventory and entity formatting are engineered, not acquired from unrestricted natural language. Stored raw strings and version metadata are simulated. `signed-simulated` is merely a fixture string; no cryptographic origin verification is performed in this 097b runner.
- Revisions are retained, and retrieval uses the latest version for the specified entity/field. No silent-shift detection is demonstrated.
- Name resolution relies on an explicit full name in the current turn or earlier context. It **does not** solve anonymous pronoun coreference, unseen entity aliases, cross-episode identity binding, or unindexed raw textual discovery beyond the seven fields. Unknown and two-name references simply defer.
- The 14 held-phrase supports deliberately expose the exact held phrasing before testing; they form a supervised demonstration cohort distinct from the zero-shot learner's training set. These results cannot establish zero-shot semantics.
- There is no matched-quality pretrained Transformer, no authentic MultiWOZ or GitHub comment evaluation and no complete COG-ATTN controller integration.

## Source and reproducibility

From the same directory as `experiment_097b.py` and `test_097b.py`:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m pytest -q test_097b.py test_097.py test_097_boundary.py
python experiment_097b.py --out 097b-results.json
python experiment_097b.py --seeds 1103,2111,2137 --size 2048 --out 097b-scale-2048.json
python experiment_097b.py --seeds 1103,2111,2137 --size 32768 --out 097b-scale-32768.json
sha256sum --check SHA256SUMS
```

The combined archive contains raw per-case predictions (text SHA-256 rather than the full generated utterance), all per-seed results, tests, packages and provenance. `test_097b.py` has six tests; the earlier 097a tests add ten. Sixteen pass together. Scoring choices were developed during this pilot; this **was not prospectively preregistered**, and holdout-family exposure from witness acquisition is intentional and disclosed. Independent researcher replication is outstanding.

## Next falsification (097c)

Work on original user utterances and completely new paraphrase families, not exact repeated support phrases. Allow a bounded trainable sparse retriever to request novel episodic candidates without a known seven-field schema, and compare against optimized BM25, typed lookup and an appropriately pretrained Transformer on the same tasks. Include semantic miscommit errors, source-epoch repair, continual witnessed learning, amortized index rebuild, and a full accounting of clarification/verification/recovery costs. Keep all authoritative writes on the unchanged signed execution boundary.

**Decision: retain a gated lexical baseline and verified witness acquisition; reject claims of general semantic equivalence or economic promotion for 097b.**
