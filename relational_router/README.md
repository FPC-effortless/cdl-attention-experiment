# LRR-001 — Causal relational token indexing

**Scientific status:** experimental MVP; not evidence of replacing attention in a language model.

This implements the first independently runnable end-to-end **token-position
selector without self-attention** based on the previous CDL / CASM / PNDS
portfolio. The scope is intentionally narrow:

1. **Supplied structure:** facts and queries arrive as
   `<F> entity predicate value </F>` and `<Q> entity predicate ? </Q>`.
   There is no learned parser, entity linker, or language model in this pilot.
2. **Learned relation code:** a small embedding + MLP aligns predicate aliases
   to six labeled relation concepts. Supervised alias labels are used during
   training, and the encoder receives only raw predicate tokens at runtime.
   Aliases are seen in training; unseen *fact/query combinations*, subjects
   and values are evaluated, not unseen vocabulary.
3. **Incremental causal index:** `(entity, learned_relation_code)` is a
   Python hash key. Each prior value-token position is appended to a posting
   list once. An unknown relation abstains rather than guessing.
4. **Bounded retrieval:** query one address and emit up to `k` most recent
   value-token positions. No query-key dot products against past tokens,
   no attention matrix, no cross-token softmax, no weighted value aggregation.

The index is a **routing primitive**. It does not compute a language-model
hidden state. Existing CASM-X19D addresses and X20U local record credit
motivate the design, but their mechanisms are not claimed to be replicated
by this small supervised pilot.

## Run

From the repository root with PyTorch and pytest installed:

```bash
python -m pytest -q relational_router/test_router.py
python -m relational_router.benchmark --lengths 8,64,256,1024 --trials 64 \
  --seed 17 --steps 120 --k 1 --output /tmp/lrr-001.json
```

Example from Python:

```python
from relational_router import fit_coder, CausalRelationIndex

coder, train = fit_coder(seed=17)
index = CausalRelationIndex(coder)
index.observe(token_index=3, entity="mira", predicate="owns", value="violin")
index.observe(token_index=8, entity="bob", predicate="holds", value="lantern")
out = index.select(query_index=10, entity="mira", predicate="possesses", k=1)
assert out.indices == (3,)
assert out.values == ("violin",)
```

## Registered metrics and controls

The benchmark creates one gold fact per document, randomizes all entity/value
symbols, forces a **different predicate alias** in the query, and includes
same-entity/wrong-relation and same-relation/wrong-entity distractors.
All fact/query token positions are materialized in a causal stream. A lookup
receives only an entity and predicate—not the gold index or family.

At 8/64/256/1024 facts, report:

- learned `recall@k` (actual gold token position retained);
- raw exact-predicate `recall@k` (no alias alignment);
- relation-only `recall@k` (ablates entity identity);
- oracle canonical-relation `recall@k` (off-path upper bound);
- number of postings visited, index probes, fact writes;
- median and 95th-percentile CPU query latency (descriptive, hardware bound).

The **cost claim** is separate: writing H facts costs O(H), and a hash-table
query uses expected O(1) address lookup plus O(k) returned postings. Memory
cost remains O(H); Python-hash cost and ingest are included in any total-work
argument. Lookup comparisons are not a GPU kernel speed benchmark.

## Hard limitations / failure boundaries

- Gold relation families are explicitly labeled for training. This does not
  demonstrate discovering semantic relations from end-task supervision.
- Entity identity and fact boundaries are supplied; aliases are from a closed
  six-family vocabulary. Natural language, pronouns, unknown words, nesting,
  ambiguity, composition and cross-sentence binding remain **untested**.
- Correctness on synthetic schemas does not prove a drop-in Transformer
  replacement or language-model perplexity parity.
- The CPU lookup latency measures only retrieval; neural code training,
  tokenization, index-building and language-model computation are not
  included in that query number.
- A supervised category classifier uses cross-entropy during training; that
  is **not** token-pair attention or softmax-weighted memory at runtime.

## Next falsification: LRR-002

Replace the supplied entity/predicate boundaries with a non-attention causal
token encoder (e.g. local convolution + recurrent role construction), and
train its relational addresses using query-answer outcome credit and X20U
local counterfactual intervention. Compare against the current scaffolded
ceiling, a learned token-pair retrieval baseline, exact and oracle indices,
randomized role codes, and a parameter-matched Transformer on real text.
Require attention-removal, hard distractor and long-context causal ablations
before claiming any architecture-level capability.

References:
- `casm_v01/EXPERIMENT_CASM_X20U_RESULT.md` (research branch result)
- `casm_v01/CASM_X19D_RESULTS_2026-09-04.md` (research branch result)
- TAC-OSM: `docs/TACOSM-SELECTIVE-001-RESULT.md`
- CDL Stage A/B: `results/stage-a-paired-2026-08-29.md`,
  `results/stage-b.md`
