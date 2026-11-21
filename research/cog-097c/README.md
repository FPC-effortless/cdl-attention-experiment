# COG-ATTN-097c · removing supplied answers and schemas

**Status:** executed zero-label source-first retrieval **preflight** on independent, human-authored Python standard-library documentation. This is a **negative result**, not a trained COG-ATTN agent or proof of parity with self-attention. Experiment design and its choices were exploratory, not prospectively preregistered.

The user requested an end to supervised tricks where the learner is supplied the entity IDs, relation inventory, candidate list, intended action, or demonstrated solution that should be learned. This study deliberately does **none** of those things. It also does not disguise the remaining boundary: this is a text-retrieval preflight, **not autonomous skill acquisition, dialogue understanding or closed-loop reasoning**.

## Environment and blind interface

1. Read 18 actual Python 3.13.5 standard-library source modules and extract 219 authentic long docstrings. For each, the model's memory contains **only the prose after the opening summary**; the source summary and source-function label are never supplied to the retriever. All 219 bodies are indexed globally, with no preselected candidate pool.
2. The evaluator privately keeps the original author-written first summary as the zero-shot retrieval query and the original source association as a target. No engineered prompts, reference solutions, gold positive pairs, source IDs, operations, field schemas or query paraphrases reach model code.
3. Independently take 71 natural author-written summaries from 7 other standard-library modules whose passages are **entirely absent** from memory; the system should decline to fabricate evidence.
4. Compare BM25, word TF-IDF, character TF-IDF and unsupervised latent semantic indexing. None is trained with evaluator linkage. No general purpose pretrained LM and no full self-attention is loaded.

**Important qualification:** The test targets a source-summary-to-passage task. Document segmentation, natural author summaries and global corpus indexing are still an engineered *evaluation contract*. This cannot establish generic comprehension, and the Python standard library is not a substitute for unseen human conversations.

## Scored result

| Method | Correct passage (219) | Median query (ms) | Index build (ms) | Cold extraction + build + 219 queries (ms) | Incorrect answers on 71 genuinely missing queries |
|---|---:|---:|---:|---:|---:|
| BM25 | 83 (37.90%) | 0.167 | 6.81 | 463.87 | 71 |
| Word TF-IDF | 78 (35.62%) | 0.604 | 26.40 | 589.29 | 71 |
| Character TF-IDF | 100 (45.66%) | 2.439 | 193.09 | 1,159.56 | 71 |
| Unsupervised LSA | 72 (32.88%) | 1.332 | 190.60 | 915.49 | 71 |

These are one-host CPU diagnostics for this corpus. Cold extraction is shared across methods (418.22 ms) and counted conservatively in each cold start. The 71 absent-query times are tabulated separately, not included in the 219-query column. None of the readers has a principled unknown/abstention mechanism; their 71/71 false answers are **a design failure**, not measured calibration after a tuned threshold.

## Critical structural finding

The actual 097b `Archive` constructor expects dictionaries with a pre-parsed `alias` and `field`, then builds its seven-field, latest-by-alias candidate list. When given genuine raw prose records without those keys, its constructor raises `KeyError('alias')`. Therefore, a scientific evaluation of the previous 097b learned model on unrestricted documents **cannot currently run as written**. Do not score that unsupported interface as 0%, and do not claim that the 097c BM25/TF-IDF/LSA controls are the complete COG-ATTN model. First build a schema-free observation interface and a trainable controller, then test it on real tasks.

## Execute locally

Python **3.13.5**; NumPy **2.3.5**, scikit-learn **1.8.0**, pytest **9.0.2**:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python experiment.py --out 097c-results.json
python -m pytest -q test_experiment.py
sha256sum -c SHA256SUMS
```

`097c-results.json` contains every source module's SHA-256, the normalized corpus digest, exact per-query correctness and scores (query text is digested), per-module counts, query latency, index building, and missing-query behavior. Python source documentation changes by Python release and environment, so results must be compared under the recorded source hashes. A fresh environment matching the published hashes is needed for exact replay. The source code itself is in the standard Python installation, not vendored into this research folder; this package is therefore **not completely self-contained**.

No real external issue was modified, no model was handed authoritative credentials, no human authorization was inferred, and no result is called a COG-ATTN language win. `PROTOCOL.md` freezes the stricter standards required for the **next** genuine qualification run.
